"""Plan assembly by searching for a disassembly sequence with deterministic DFS.

Keep one initial brick on the board, recursively remove a target with a valid
reference among the remaining bricks, then reverse removals into assembly steps.
Each candidate is evaluated using four criteria:
  Ccon: reference exists and the relative pose/types match an available skill.
  Cobs: only the reference is visible from workspace and wrist camera views in
        the target-removed scene, using approximate geometry in millimeters.
  Cop: the vertical sweep of the target and simplified gripper is collision-free.
  Cstab: stability evaluation depends on Gurobi and is temporarily removed
         (returns None). Integration reference:
         https://github.com/intelligent-control-lab/PhysicsAwareCombinatorialASP

"""
from dataclasses import dataclass

from .brick import Brick
from .structure import Structure

ALLOWED_SKILLS = frozenset({
    (0, 0, 1, 0, 0, 0), (1, 0, 1, 0, 0, 0), (2, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 1, 0), (0, 0, 1, 0, 0, 1),
    (2, 0, 0, 0, 0, 1), (4, 0, 0, 0, 1, 0),
})


@dataclass(frozen=True)
class ConstraintChecks:
    """Paper criteria; None means a criterion has not been evaluated."""

    ccon: bool
    cobs: bool
    cop: bool
    cstab: bool | None

    @property
    def passes_implemented_checks(self) -> bool:
        """Allow planning while the explicitly missing stability check is pending."""
        return self.ccon and self.cobs and self.cop and self.cstab is not False

    @property
    def fully_verified(self) -> bool:
        return self.ccon and self.cobs and self.cop and self.cstab is True


class AssemblyPlanner:
    """Search with Ccon/Cobs/Cop; Cstab awaits the external stability backend.

    Returned plans pass approximate geometry checks, not physical certification.
    """

    def __init__(self, target_structure: Structure):
        if not isinstance(target_structure, Structure):
            raise TypeError("target_structure must be a Structure")
        self.target_structure = target_structure

    def solve(self) -> dict:
        structure = self.target_structure
        if not structure:
            raise ValueError("Cannot plan an empty structure")
        if any(b.ori != 0 for b in structure):
            raise ValueError("The released single-arm skills currently require ori=0")
        anchor = min(structure, key=lambda b: (b.z, b.x, b.y, b.brick_id))
        if anchor.z != 1:
            raise ValueError("Initial brick must be on the board with z=1")
        sequence = self._dfs_search(structure, anchor, set())
        if sequence is None:
            raise ValueError("No plan found with the released skills and geometric constraints")
        return self._build_plan(anchor, sequence)

    def _dfs_search(self, current: Structure, anchor: Brick, failed: set):
        if len(current) == 1:
            return []
        key = frozenset(b.brick_id for b in current)
        if key in failed:
            return None
        for target in sorted(current, key=lambda b: (-b.z, -b.x, b.y, b.brick_id)):
            if target == anchor:
                continue
            remaining = current.without(target.brick_id)
            for ref in sorted(remaining, key=lambda b: (abs(target.z-b.z),
                               abs(target.x-b.x)+abs(target.y-b.y), b.brick_id)):
                if self.is_feasible_step(ref, target, remaining):
                    prefix = self._dfs_search(remaining, anchor, failed)
                    if prefix is not None:
                        return prefix + [(ref, target)]
        failed.add(key)
        return None

    @classmethod
    def is_feasible_step(cls, ref: Brick, target: Brick, remaining: Structure) -> bool:
        """Shared admission checks for search and serialized-plan replay."""
        checks = cls.check_constraints(ref, target, remaining)
        return checks.passes_implemented_checks

    @classmethod
    def check_constraints(cls, ref: Brick, target: Brick, remaining: Structure,
                          tau: tuple[int, ...] | None = None) -> ConstraintChecks:
        """Evaluate a removal candidate; remaining excludes target, includes ref.

        Omit tau to derive it from the brick poses/types. A supplied encoding
        must exactly match those poses/types and belong to the released skills.
        """
        if tau is None:
            tau = target.task_encoding_relative_to(ref)
        return ConstraintChecks(
            ccon=cls._check_connectivity(ref, target, remaining, tau),
            cobs=cls._check_observability(ref, remaining),
            cop=cls._check_operability(target, remaining),
            cstab=cls._check_stability(remaining),
        )

    @staticmethod
    def _check_connectivity(ref: Brick, target: Brick, remaining: Structure,
                            tau: tuple[int, ...]) -> bool:
        """Ccon: an existing reference matches a supported relative task encoding."""
        return (ref in remaining and not any(b.brick_id == target.brick_id for b in remaining)
                and not any((x, y, target.z) in remaining.occupancy_grid
                            for x, y in target.footprint)
                and ref.ori == target.ori == 0
                and all(type(value) is int for value in tau)
                and tuple(tau) == target.task_encoding_relative_to(ref)
                and tuple(tau) in ALLOWED_SKILLS)

    @staticmethod
    def _check_observability(ref: Brick, remaining: Structure) -> bool:
        """Cobs approximation: reference visibility from workspace and wrist views.

        Evaluate the pre-insertion scene (target removed), excluding the reference
        itself from ray obstacles. Target visibility is intentionally not checked.
        This is narrower than the paper's target-and-reference criterion.
        Calibrated camera poses, held-target/gripper occlusion and visibility
        throughout motion require robot integration and are not modeled here.
        """
        return ref in remaining and remaining.is_observable(ref)

    @staticmethod
    def _check_operability(target: Brick, remaining: Structure) -> bool:
        """Cop approximation: vertical swept body and simplified gripper envelope."""
        return remaining.can_extract(target)

    @staticmethod
    def _check_stability(remaining: Structure) -> bool | None:
        return None

    @staticmethod
    def _build_plan(anchor: Brick, sequence: list[tuple[Brick, Brick]]) -> dict:
        steps = []
        for index, (ref, target) in enumerate(sequence):
            steps.append({"step_id": index + 1, "b_ref": ref.to_dict(), "b_tgt": target.to_dict(),
                          "task_encoding": list(target.task_encoding_relative_to(ref)),
                          "current_image": f"images/{index:03d}.png",
                          "current_mask": f"masks/{index:03d}.png"})
        return {"initial_brick": anchor.to_dict(), "steps": steps}
