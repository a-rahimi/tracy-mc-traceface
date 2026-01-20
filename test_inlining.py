# %%
from numba import njit


@njit
def func_add(x):
    return x + 1


@njit
def func_sub(x):
    return x - 2


@njit
def func_mul(x):
    return x * 3


def outer_func(x):
    a = func_add(x)
    b = func_sub(a)
    return func_mul(b)


class TestInlining:
    def test_inlining_and_correctness(self):
        from numba.core.compiler import CompilerBase, DefaultPassBuilder
        from numba.core.compiler_machinery import FunctionPass, register_pass
        from numba.core import ir
        from numba import njit
        import numba
        import inlining

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
                    inlining.InlineAllCallsPass, numba.core.untyped_passes.IRProcessing
                )
                # Add Verification pass
                pm.add_pass_after(VerifyNoCallsPass, inlining.InlineAllCallsPass)
                pm.finalize()
                return [pm]

        @njit(pipeline_class=InliningTestCompiler)
        def compiled_func(x):
            return outer_func(x)

        # Test correctness
        assert compiled_func(10) == 27
