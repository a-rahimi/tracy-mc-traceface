from typing import Any
from numba.core import ir, ir_utils
from numba.core.compiler_machinery import FunctionPass, register_pass


@register_pass(mutates_CFG=True, analysis_only=False)
class DiamondPruningPass(FunctionPass):
    """
    Removes diamond control flow patterns where both branches are empty or jump
    to the same target.
    """

    _name = "diamond_pruning_pass"

    def __init__(self) -> None:
        FunctionPass.__init__(self)

    def run_pass(self, state: Any) -> bool:
        func_ir = state.func_ir
        blocks = func_ir.blocks
        changed = False

        for label, block in list(blocks.items()):
            if isinstance(block.terminator, ir.Branch):
                true_label = block.terminator.truebr
                false_label = block.terminator.falsebr

                true_target = self._get_jump_target(blocks, true_label)
                false_target = self._get_jump_target(blocks, false_label)

                # Case 1: Both jump to M (or match targets)
                if (
                    true_target is not None
                    and false_target is not None
                    and true_target == false_target
                ):
                    block.body[-1] = ir.Jump(true_target, block.terminator.loc)
                    changed = True
                    continue

                # Case 2: True block is effectively a jump to False label
                if true_target is not None and true_target == false_label:
                    block.body[-1] = ir.Jump(false_label, block.terminator.loc)
                    changed = True
                    continue

                # Case 3: False block is effectively a jump to True label
                if false_target is not None and false_target == true_label:
                    block.body[-1] = ir.Jump(true_label, block.terminator.loc)
                    changed = True
                    continue

                # Case 4: Both branches return the same constant
                true_ret = self._get_return_const(blocks, true_label)
                false_ret = self._get_return_const(blocks, false_label)

                if (
                    true_ret is not None
                    and false_ret is not None
                    and true_ret == false_ret
                ):
                    block.body[-1] = ir.Jump(true_label, block.terminator.loc)
                    changed = True
                    continue

        if changed:
            ir_utils.dead_code_elimination(func_ir)

        return True

    def _get_jump_target(self, blocks: Any, label: int) -> Any:
        if label not in blocks:
            return None
        blk = blocks[label]
        if len(blk.body) == 1 and isinstance(blk.body[0], ir.Jump):
            return blk.body[0].target
        return None

    def _get_return_const(self, blocks: Any, label: int) -> Any:
        if label not in blocks:
            return None
        blk = blocks[label]
        if not blk.body:
            return None

        term = blk.body[-1]
        if not isinstance(term, ir.Return):
            return None

        local_defs = {}
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                local_defs[stmt.target.name] = stmt.value

        var = term.value

        # Helper to resolve var to const
        visited = set()
        while isinstance(var, ir.Var):
            if var.name in visited:
                return None
            visited.add(var.name)

            if var.name not in local_defs:
                return None

            val = local_defs[var.name]

            if isinstance(val, ir.Const):
                return val.value
            elif isinstance(val, ir.Expr) and val.op == "cast":
                var = val.value
            elif isinstance(val, ir.Var):
                var = val
            else:
                return None

        if isinstance(var, ir.Const):
            return var.value

        return None
