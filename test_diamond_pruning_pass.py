import pytest
from unittest.mock import MagicMock, patch
import numba
from numba.core import ir, ir_utils, untyped_passes
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from diamond_pruning_pass import DiamondPruningPass


# Define a simple DCE pass for the test
@register_pass(mutates_CFG=True, analysis_only=False)
class DeadCodeEliminationPassForTesting(FunctionPass):
    _name = "test_dead_code_elimination_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        ir_utils.dead_code_elimination(state.func_ir)
        return True


# Define an Inspector pass to capture IR
@register_pass(mutates_CFG=False, analysis_only=True)
class InspectorPass(FunctionPass):
    _name = "inspector_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        TestDiamondPruningIntegration.last_func_ir = state.func_ir
        return True


class TestDiamondPruningIntegration:
    last_func_ir = None

    def setup_method(self):
        TestDiamondPruningIntegration.last_func_ir = None

    def test_integration_identical_branches(self):
        """Test Case 4 (Identical Returns) in a real pipeline"""

        class PruningTestCompiler(CompilerBase):
            def define_pipelines(self):
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(DiamondPruningPass, untyped_passes.IRProcessing)
                pm.add_pass_after(DeadCodeEliminationPassForTesting, DiamondPruningPass)
                pm.add_pass_after(InspectorPass, DeadCodeEliminationPassForTesting)
                pm.finalize()
                return [pm]

        @numba.njit(pipeline_class=PruningTestCompiler)
        def func_identical_returns(x):
            if x > 0:
                return 42
            else:
                return 42

        # Compile and check result correctness
        assert func_identical_returns(1) == 42
        assert func_identical_returns(-1) == 42

        # Check IR for absence of Branch
        func_ir = self.last_func_ir
        assert func_ir is not None

        has_branch = False
        for blk in func_ir.blocks.values():
            if isinstance(blk.terminator, ir.Branch):
                has_branch = True
                break

        assert not has_branch, (
            "Diamond pruning failed to remove the branch in func_identical_returns"
        )

    def test_integration_empty_branches(self):
        """Test Case 1 (Both jump to same target) in a real pipeline"""

        class PruningTestCompiler(CompilerBase):
            def define_pipelines(self):
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(DiamondPruningPass, untyped_passes.IRProcessing)
                pm.add_pass_after(DeadCodeEliminationPassForTesting, DiamondPruningPass)
                pm.add_pass_after(InspectorPass, DeadCodeEliminationPassForTesting)
                pm.finalize()
                return [pm]

        @numba.njit(pipeline_class=PruningTestCompiler)
        def func_empty_branches(x):
            # This pattern should generate empty branches jumping to return
            if x > 0:
                pass
            else:
                pass
            return 100

        # Compile and check result correctness
        assert func_empty_branches(1) == 100
        assert func_empty_branches(-1) == 100

        # Check IR for absence of Branch
        func_ir = self.last_func_ir
        assert func_ir is not None

        has_branch = False
        for blk in func_ir.blocks.values():
            if isinstance(blk.terminator, ir.Branch):
                has_branch = True
                break

        assert not has_branch, (
            "Diamond pruning failed to remove the branch in func_empty_branches"
        )

    def test_unused_branch_optimization(self):
        """Test that unused branches are pruned and simplified to a return"""

        class PruningTestCompiler(CompilerBase):
            def define_pipelines(self):
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(DiamondPruningPass, untyped_passes.IRProcessing)
                pm.add_pass_after(DeadCodeEliminationPassForTesting, DiamondPruningPass)
                pm.add_pass_after(InspectorPass, DeadCodeEliminationPassForTesting)
                pm.finalize()
                return [pm]

        @numba.njit(pipeline_class=PruningTestCompiler)
        def func_with_unused_branch(traceable: float) -> float:
            if traceable > 0.5:
                a = 2
            else:
                a = 0.5
            return 3

        # Compile and check result correctness
        assert func_with_unused_branch(0.6) == 3
        assert func_with_unused_branch(0.4) == 3

        # Check IR for absence of Branch
        func_ir = self.last_func_ir
        assert func_ir is not None

        has_branch = False
        for blk in func_ir.blocks.values():
            if isinstance(blk.terminator, ir.Branch):
                has_branch = True
                break

        assert not has_branch, (
            "Diamond pruning failed to remove the branch in func_with_unused_branch"
        )
