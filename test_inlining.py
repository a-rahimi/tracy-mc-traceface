# %%
from typing import NamedTuple
from numba import njit
from importlib import reload
import tracer

reload(tracer)

# Import TestMakeDecisionHierarchically from the original test file
from test_tracer import TestMakeDecisionHierarchically


class User(NamedTuple):
    metric1: tracer.Float
    metric2: tracer.Float
    metric3: tracer.Int


class TestInlining:
    def test_inlining_and_correctness(self):
        from numba.core.compiler import CompilerBase, DefaultPassBuilder
        from numba.core.compiler_machinery import FunctionPass, register_pass
        from numba.core import ir
        from numba import njit
        import numba

        # VerifyNoCallsPass definition
        @register_pass(mutates_CFG=False, analysis_only=True)
        class VerifyNoCallsPass(FunctionPass):
            _name = "verify_no_calls_pass"

            def __init__(self):
                FunctionPass.__init__(self)

            def run_pass(self, state):
                from numba.core.ir_utils import get_definition

                func_ir = state.func_ir
                for label, block in func_ir.blocks.items():
                    for stmt in block.body:
                        if (
                            isinstance(stmt, ir.Assign)
                            and isinstance(stmt.value, ir.Expr)
                            and stmt.value.op == "call"
                        ):
                            # Check if it's a call to a builtin type (like bool, int) which cannot be inlined
                            expr = stmt.value
                            func_def = get_definition(func_ir, expr.func)
                            is_builtin_type = False

                            if isinstance(func_def, (ir.Global, ir.FreeVar)):
                                val = func_def.value
                                if isinstance(val, type) and val in (
                                    bool,
                                    int,
                                    float,
                                    str,
                                    tuple,
                                    list,
                                    dict,
                                    set,
                                ):
                                    is_builtin_type = True

                            if not is_builtin_type:
                                raise AssertionError(
                                    f"Found remaining call instruction: {stmt.value} (func definition: {func_def})"
                                )
                return True

        class InliningTestCompiler(CompilerBase):
            def define_pipelines(self):
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                # Add InlineAllCallsPass
                pm.add_pass_after(
                    tracer.InlineAllCallsPass, numba.core.untyped_passes.IRProcessing
                )
                # Add Verification pass
                pm.add_pass_after(VerifyNoCallsPass, tracer.InlineAllCallsPass)
                pm.finalize()
                return [pm]

        @njit(pipeline_class=InliningTestCompiler)
        def compiled_func(user):
            return TestMakeDecisionHierarchically.make_decision_hierarchically(user)

        # Test correctness
        u1 = User(metric1=0.9, metric2=0.9, metric3=1)
        assert compiled_func(u1) is True

        u2 = User(metric1=0.5, metric2=0.5, metric3=6)
        assert compiled_func(u2) is True
