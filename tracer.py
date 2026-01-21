from typing import NamedTuple, Any, List, Tuple, Callable, TypeVar, Generic, Dict
import typing
import inspect
from dataclasses import dataclass
import numba
from numba.core import ir
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core import ir_utils
import numba.core.untyped_passes

import diamond_pruning_pass
import inlining


T = TypeVar("T")


class Traceable(Generic[T]):
    pass


class TraceEvent(NamedTuple):
    op: str
    output_var: str
    output_val: Any
    inputs: List[Tuple[str, Any]]  # (var_name, value)
    lineno: int


class TracingIRNode:
    pass


@dataclass
class Expression(TracingIRNode):
    text: str
    value: Any = None

    def __str__(self) -> str:
        return self.text


@dataclass
class Return(TracingIRNode):
    expression: Expression

    def __str__(self) -> str:
        return f"return {self.expression.text}"


@dataclass
class Conditional(TracingIRNode):
    condition: Expression
    value: Return | "Conditional"

    def __str__(self) -> str:
        cond_val_str = (
            f" (={self.condition.value})" if self.condition.value is not None else ""
        )
        parts = [f"if {self.condition.text}{cond_val_str}:"]
        if self.value:
            for line in str(self.value).splitlines():
                parts.append(f"  {line}")
        return "\n".join(parts)


def stringify_constant(val: Any) -> str:
    return f"{val:.4f}" if isinstance(val, float) else str(val)


class Trace:
    def __init__(self, events: List[TraceEvent]) -> None:
        self.events = events

    def to_ir(self) -> Return | Conditional | None:
        var_exprs = {}
        BINARY_OPS = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "truediv": "/",
            "floordiv": "//",
            "mod": "%",
            "pow": "**",
            "gt": ">",
            "ge": ">=",
            "lt": "<",
            "le": "<=",
            "eq": "==",
            "ne": "!=",
            "bitand": "&",
            "bitor": "|",
            "bitxor": "^",
            "lshift": "<<",
            "rshift": ">>",
        }

        def resolve(name: str, val: Any) -> str:
            if name in var_exprs:
                return var_exprs[name]
            if name.startswith("$const") or name.startswith("const("):
                return stringify_constant(val)
            if name.startswith("$") and not name.startswith("$const"):
                return "unknown variable"
            return name

        nodes = []

        for event in self.events:
            if event.op == "branch":
                cond_name, cond_val = event.inputs[0]
                expr_str = resolve(cond_name, cond_val)
                # Skip creating Conditional if condition is a concrete boolean value
                # (True/False) rather than an Expression
                if expr_str in ("True", "False"):
                    # Skip this conditional entirely - it's a concrete value, not an expression
                    continue
                # Create Conditional. value (body) is not yet known.
                nodes.append(Conditional(Expression(expr_str, cond_val), None))

            elif event.op == "return":
                if event.inputs:
                    val_name, val_val = event.inputs[0]
                    expr_str = resolve(val_name, val_val)
                else:
                    expr_str = stringify_constant(event.output_val)

                nodes.append(Return(Expression(expr_str, event.output_val)))

            elif event.op in ("cast", "assign") and len(event.inputs) == 1:
                in_name, in_val = event.inputs[0]
                var_exprs[event.output_var] = resolve(in_name, in_val)

            elif event.op in BINARY_OPS:
                lhs_name, lhs_val = event.inputs[0]
                rhs_name, rhs_val = event.inputs[1]
                lhs_expr = resolve(lhs_name, lhs_val)
                rhs_expr = resolve(rhs_name, rhs_val)
                op_sym = BINARY_OPS[event.op]
                var_exprs[event.output_var] = f"({lhs_expr} {op_sym} {rhs_expr})"

            elif event.op == "bool":
                if event.inputs:
                    name, val = event.inputs[0]
                    var_exprs[event.output_var] = resolve(name, val)
                else:
                    var_exprs[event.output_var] = "False"

            else:
                args_str = ", ".join(resolve(name, val) for name, val in event.inputs)
                var_exprs[event.output_var] = f"{event.op}({args_str})"

        # Link nodes
        if not nodes:
            return None

        for i in range(len(nodes) - 2, -1, -1):
            if isinstance(nodes[i], Conditional):
                nodes[i].value = nodes[i + 1]

        return nodes[0]

    def pretty_print(self) -> str:
        return str(self.to_ir())


# Global context for the current trace
_CURRENT_TRACE: List[TraceEvent] = []
_TRACEABLE_VARS: set = set()


@numba.njit
def _log_trace_tuple(record: Tuple[str, str, Any, str, Any, str, Any, int]) -> None:
    with numba.objmode():
        op_name, out_name, out_val, in1_name, in1_val, in2_name, in2_val, lineno = (
            record
        )
        inputs = []
        if in1_name:
            val_name = in1_name
            if in1_name in _TRACEABLE_VARS:
                pass
            inputs.append((val_name, in1_val))
        if in2_name:
            val_name = in2_name
            if in2_name in _TRACEABLE_VARS:
                pass
            inputs.append((val_name, in2_val))

        # Propagate traceability
        is_traceable = False
        if in1_name and in1_name in _TRACEABLE_VARS:
            is_traceable = True
        if in2_name and in2_name in _TRACEABLE_VARS:
            is_traceable = True

        if is_traceable:
            _TRACEABLE_VARS.add(out_name)

        _CURRENT_TRACE.append(TraceEvent(op_name, out_name, out_val, inputs, lineno))


@register_pass(mutates_CFG=True, analysis_only=False)
class TracingInjectionPass(FunctionPass):
    """Injects tracing code into the function's IR.

    When the function is subsequently compiled, it'll log its key steps into
    a global list of TraceEvents.
    """

    _name = "tracing_injection_pass"

    def __init__(self) -> None:
        FunctionPass.__init__(self)

    def run_pass(self, state: Any) -> bool:
        func_ir = state.func_ir

        # Make the tracer functino available to the function we're modifying.
        func_ir.func_id.func.__globals__["_log_trace_tuple"] = _log_trace_tuple

        # Record variable assignments. The python cmopiler introduces lots of
        # anonymous temporary variables, and we'll have to undo those
        # assignments.
        definitions = {}
        for block in func_ir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, ir.Assign):
                    definitions[stmt.target.name] = stmt.value

        def resolve_name(var: Any) -> str:
            if isinstance(var, ir.Var) and var.name in definitions:
                defn = definitions[var.name]
                if isinstance(defn, ir.Expr) and defn.op == "getattr":
                    return defn.attr
            return getattr(var, "name", str(var))

        for block in func_ir.blocks.values():
            new_body = []
            for stmt in block.body:
                # BinOp (Arithmetic)
                if (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "binop"
                ):
                    new_body.append(stmt)
                    self._inject_log(
                        new_body,
                        stmt.loc,
                        stmt.value.fn,
                        stmt.target,
                        stmt.target,
                        resolve_name(stmt.value.lhs),
                        stmt.value.lhs,
                        resolve_name(stmt.value.rhs),
                        stmt.value.rhs,
                    )

                # InplaceBinOp
                elif (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "inplace_binop"
                ):
                    new_body.append(stmt)
                    self._inject_log(
                        new_body,
                        stmt.loc,
                        stmt.value.fn,
                        stmt.target,
                        stmt.target,
                        resolve_name(stmt.value.lhs),
                        stmt.value.lhs,
                        resolve_name(stmt.value.rhs),
                        stmt.value.rhs,
                    )

                # Cast
                elif (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "cast"
                ):
                    new_body.append(stmt)
                    self._inject_log(
                        new_body,
                        stmt.loc,
                        "cast",
                        stmt.target,
                        stmt.target,
                        resolve_name(stmt.value.value),
                        stmt.value.value,
                        "",
                        None,
                    )

                # Simple Assignment (Var or Const)
                elif isinstance(stmt, ir.Assign) and (
                    isinstance(stmt.value, ir.Var) or isinstance(stmt.value, ir.Const)
                ):
                    new_body.append(stmt)

                    val_to_log = stmt.value
                    if isinstance(stmt.value, ir.Const):
                        val_to_log = stmt.value.value

                    self._inject_log(
                        new_body,
                        stmt.loc,
                        "assign",
                        stmt.target,
                        stmt.target,
                        resolve_name(stmt.value),
                        val_to_log,
                        "",
                        None,
                    )

                # Branch
                elif isinstance(stmt, ir.Branch):
                    self._inject_log(
                        new_body,
                        stmt.loc,
                        "branch",
                        resolve_name(stmt.cond),
                        stmt.cond,
                        resolve_name(stmt.cond),
                        stmt.cond,
                        "",
                        None,
                    )
                    new_body.append(stmt)

                # Return
                elif isinstance(stmt, ir.Return):
                    self._inject_log(
                        new_body,
                        stmt.loc,
                        "return",
                        "return_val",
                        stmt.value,
                        resolve_name(stmt.value),
                        stmt.value,
                        "",
                        None,
                    )
                    new_body.append(stmt)

                # Generic Expr Assignment
                elif isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                    new_body.append(stmt)

                    op = stmt.value.op
                    op_to_log = op
                    in1_name = ""
                    in1_val = None

                    if op == "call":
                        # Try to resolve function name
                        func_var = stmt.value.func
                        func_def = ir_utils.get_definition(func_ir, func_var)
                        if isinstance(func_def, (ir.Global, ir.FreeVar)):
                            func_obj = getattr(func_def, "value", None)
                            if func_obj:
                                if hasattr(func_obj, "__name__"):
                                    op_to_log = func_obj.__name__
                                else:
                                    op_to_log = str(func_obj)

                    if op == "call" and stmt.value.args:
                        in1_name = resolve_name(stmt.value.args[0])
                        in1_val = stmt.value.args[0]
                    else:
                        try:
                            val = stmt.value.value
                            in1_name = resolve_name(val)
                            in1_val = val
                        except (KeyError, AttributeError):
                            pass

                    self._inject_log(
                        new_body,
                        stmt.loc,
                        op_to_log,
                        stmt.target,
                        stmt.target,
                        in1_name,
                        in1_val,
                        "",
                        None,
                    )

                else:
                    new_body.append(stmt)

            block.body = new_body

        # Insert global definition at entry block
        first_label = min(func_ir.blocks.keys())
        entry_block = func_ir.blocks[first_label]
        scope = entry_block.scope
        loc = entry_block.loc

        log_var = ir.Var(scope, "$log_trace_tuple", loc)
        assign_global = ir.Assign(
            ir.Global("_log_trace_tuple", _log_trace_tuple, loc), log_var, loc
        )
        entry_block.body.insert(0, assign_global)

        return True

    def _inject_log(
        self,
        body_list: List[Any],
        loc: ir.Loc,
        op_name: Any,
        out_name: Any,
        out_val_var: Any,
        in1_name: Any,
        in1_val_var: Any,
        in2_name: Any,
        in2_val_var: Any,
    ) -> None:
        # 'log_var' is available as "$log_trace_tuple" in the scope (via dominance)
        scope = (
            out_val_var.scope if hasattr(out_val_var, "scope") else ir.Scope(None, loc)
        )
        log_func = ir.Var(scope, "$log_trace_tuple", loc)

        def ensure_var(val: Any) -> ir.Var:
            if isinstance(val, ir.Var):
                return val

            # Handle non-string op_name (like functions)
            if callable(val) and hasattr(val, "__name__"):
                val = val.__name__

            # Create const var
            v = ir.Var(scope, f"$const_{id(val)}", loc)
            body_list.append(ir.Assign(ir.Const(val, loc), v, loc))
            return v

        args = [
            ensure_var(op_name),
            ensure_var(out_name if isinstance(out_name, str) else str(out_name)),
            ensure_var(out_val_var),
            ensure_var(in1_name if isinstance(in1_name, str) else str(in1_name)),
            ensure_var(in1_val_var),
            ensure_var(in2_name if isinstance(in2_name, str) else str(in2_name)),
            ensure_var(in2_val_var) if in2_val_var is not None else ensure_var(0),
            ensure_var(loc.line),
        ]

        # Build tuple
        tuple_var = ir.Var(scope, f"$tuple_{id(args)}", loc)
        tuple_expr = ir.Expr.build_tuple(args, loc)
        body_list.append(ir.Assign(tuple_expr, tuple_var, loc))

        # Call _log_trace_tuple(tuple_var)
        call_expr = ir.Expr.call(log_func, [tuple_var], (), loc)

        # Dummy return var
        dummy_var = ir.Var(scope, f"$log_ret_{loc.line}_{id(call_expr)}", loc)
        body_list.append(ir.Assign(call_expr, dummy_var, loc))


class TraceCompiler(CompilerBase):
    def define_pipelines(self) -> List[Any]:
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(
            inlining.InlineAllCallsPass, numba.core.untyped_passes.IRProcessing
        )
        pm.add_pass_after(
            numba.core.untyped_passes.DeadBranchPrune, inlining.InlineAllCallsPass
        )
        pm.add_pass_after(
            diamond_pruning_pass.DiamondPruningPass,
            numba.core.untyped_passes.DeadBranchPrune,
        )
        pm.add_pass_after(TracingInjectionPass, diamond_pruning_pass.DiamondPruningPass)
        pm.finalize()
        return [pm]


def get_args_to_trace(
    func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> set[str]:
    bound = inspect.signature(func).bind(*args, **kwargs)
    bound.apply_defaults()
    hints = typing.get_type_hints(func)

    return {
        name
        for name in bound.arguments
        if name in hints and typing.get_origin(hints[name]) is Traceable
    }


def trace(func: Callable[..., Any]) -> Callable[..., Any]:
    compiled_func = numba.njit(pipeline_class=TraceCompiler)(func)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # TODO: don't rely on a global variable. Find a way to attach this to
        # either the thread or to the function being traced.
        global _CURRENT_TRACE
        global _TRACEABLE_VARS
        _CURRENT_TRACE = []  # Reset
        _TRACEABLE_VARS = get_args_to_trace(func, args, kwargs)

        res = compiled_func(*args, **kwargs)
        wrapper.trace = Trace(list(_CURRENT_TRACE))
        return res

    return wrapper
