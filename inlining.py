from typing import Any
from numba.core import ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.inline_closurecall import inline_closure_call
from numba.core import ir_utils

@register_pass(mutates_CFG=True, analysis_only=False)
class InlineAllCallsPass(FunctionPass):
    """Inlines all function calls

    We don't want function calls in the traces we produce.  Numba has a built-in
    ``InlineClosureCallPass``, but it's too conservative for our needs; it
    inlines closures for type inference. This pass forces inlining of global
    functions and ``njit``-compiled functions to ensure a flat trace.
    """

    _name = "inline_all_calls_pass"

    def __init__(self) -> None:
        FunctionPass.__init__(self)

    def run_pass(self, state: Any) -> bool:
        func_ir = state.func_ir
        work_list = list(func_ir.blocks.items())

        while work_list:
            label, block = work_list.pop()
            for i, instr in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == "call":

                        def impl(func_ir: Any, block: Any, i: int, expr: Any) -> bool:
                            func_def = ir_utils.get_definition(func_ir, expr.func)
                            func_obj = None

                            if isinstance(func_def, (ir.Global, ir.FreeVar)):
                                func_obj = getattr(func_def, "value", None)

                            # Handle getattr (method calls like Class.method)
                            elif (
                                isinstance(func_def, ir.Expr)
                                and func_def.op == "getattr"
                            ):
                                # Get the object the method is being called on
                                obj_def = ir_utils.get_definition(
                                    func_ir, func_def.value
                                )
                                if isinstance(obj_def, (ir.Global, ir.FreeVar)):
                                    obj = getattr(obj_def, "value", None)
                                    if obj is not None:
                                        # Get the method from the object
                                        attr_name = func_def.attr
                                        func_obj = getattr(obj, attr_name, None)

                            # Handle CPUDispatcher (njit functions)
                            if func_obj and hasattr(func_obj, "py_func"):
                                func_obj = func_obj.py_func

                            if (
                                func_obj
                                and callable(func_obj)
                                and not isinstance(func_obj, type)
                                and hasattr(func_obj, "__code__")
                            ):
                                inline_closure_call(
                                    func_ir,
                                    func_ir.func_id.func.__globals__,
                                    block,
                                    i,
                                    func_obj,
                                    work_list=work_list,
                                )
                                return True

                            if (
                                isinstance(func_def, ir.Expr)
                                and func_def.op == "make_function"
                            ):
                                inline_closure_call(
                                    func_ir,
                                    func_ir.func_id.func.__globals__,
                                    block,
                                    i,
                                    func_def,
                                    work_list=work_list,
                                )
                                return True

                            return False

                        if ir_utils.guard(impl, func_ir, block, i, expr):
                            break

        ir_utils.dead_code_elimination(func_ir, list(func_ir.blocks.keys()))
        return True
