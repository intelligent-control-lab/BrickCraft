"""Immutable structure states and geometric checks.

Extraction and visibility are geometric approximations, not a dynamics or robot
collision solver. Search branches share immutable bricks, never mutable state.
"""
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from .brick import Brick, STUD_SPACING_MM

# Directions from a surface toward each camera, in physical XYZ coordinates.
# These orthographic views are assumptions, not calibrated camera poses.
WORKSPACE_VIEW_DIRECTION = (0.0, 1.0, 0.0)
WRIST_VIEW_DIRECTION = (1.0, 0.0, 1.0)


def _ray_intersects_box(origin, direction, bounds) -> bool:
    """Intersect an infinite forward ray with the interior of a body box."""
    entry, exit = 0.0, float("inf")
    for position, delta, lower, upper in zip(origin, direction, *bounds):
        if delta == 0:
            if not lower < position < upper:
                return False
        else:
            t1, t2 = (lower - position) / delta, (upper - position) / delta
            entry, exit = max(entry, min(t1, t2)), min(exit, max(t1, t2))
    return exit > entry


@dataclass(frozen=True)
class Structure:
    bricks: tuple[Brick, ...] = ()
    occupancy_grid: Mapping[tuple[int, int, int], int] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        bricks = tuple(self.bricks)
        grid = {}
        ids = set()
        for brick in bricks:
            if not isinstance(brick, Brick):
                raise ValueError("Structure requires Brick instances")
            if brick.brick_id in ids:
                raise ValueError("Duplicate brick_id")
            ids.add(brick.brick_id)
            for x, y in brick.footprint:
                cell = (x, y, brick.z)
                if cell in grid:
                    raise ValueError(f"Brick {brick.brick_id} overlaps another brick")
                grid[cell] = brick.brick_id
        object.__setattr__(self, "bricks", bricks)
        object.__setattr__(self, "occupancy_grid", MappingProxyType(grid))

    @classmethod
    def from_json(cls, data) -> "Structure":
        if not isinstance(data, list) or not data:
            raise ValueError("Structure must be a nonempty JSON array of brick objects")
        return cls(tuple(Brick.from_dict(item) for item in data))

    def to_json(self) -> list[dict]:
        return [brick.to_dict() for brick in self.bricks]

    def __iter__(self):
        return iter(self.bricks)

    def __len__(self):
        return len(self.bricks)

    def without(self, brick_id: int) -> "Structure":
        if not any(b.brick_id == brick_id for b in self.bricks):
            raise KeyError(brick_id)
        return Structure(tuple(b for b in self.bricks if b.brick_id != brick_id))

    def with_brick(self, brick: Brick) -> "Structure":
        return Structure((*self.bricks, brick))

    def can_extract(self, target: Brick) -> bool:
        """Check the entire +Z swept volume of the body and gripper envelope.

        The gripper is approximated by one-stud (16 mm) wide strips on both
        +/-Y sides, spanning the target's X extent, starting at its bottom.
        Every higher layer is checked, including distant overhead obstacles.
        This is not a mesh collision check of the actual fingers, palm or arm.
        The receiver is the remaining structure, without the target.
        """
        xs = {x for x, _ in target.footprint}
        ys = {y for _, y in target.footprint}
        fingers = {(x, y) for x in xs for y in (min(ys) - 1, max(ys) + 1)}
        return not any(b.z >= target.z and b.footprint & (target.footprint | fingers)
                       for b in self.bricks)

    def is_visible_from(self, brick: Brick, direction: tuple[float, float, float]) -> bool:
        """At least one camera-facing surface sample has a clear sight line.

        Samples cover top and side faces at stud centers (side faces use the
        body's mid-height). Including sides lets a covered reference brick
        remain visible. Rays and boxes use millimeters: a (1, 0, 1) direction
        is physically 45 degrees despite the unequal 16/19 mm grid scales.
        Partial visibility does not guarantee detection or visibility throughout
        motion; perspective, field of view and gripper occlusion are not modeled.
        """
        if len(direction) != 3 or not all(isfinite(d) for d in direction) or not any(direction):
            raise ValueError("Camera direction must be a finite, nonzero XYZ vector")
        lower, upper = brick.bounds_mm
        centers = (
            sorted({(x + 0.5) * STUD_SPACING_MM for x, _ in brick.footprint}),
            sorted({(y + 0.5) * STUD_SPACING_MM for _, y in brick.footprint}),
            [(lower[2] + upper[2]) / 2],
        )
        obstacles = [b.bounds_mm for b in self if b.brick_id != brick.brick_id]
        for axis, delta in enumerate(direction):
            if delta == 0:
                continue
            a, b = [i for i in range(3) if i != axis]
            for first in centers[a]:
                for second in centers[b]:
                    point = [0.0, 0.0, 0.0]
                    point[axis] = upper[axis] if delta > 0 else lower[axis]
                    point[a], point[b] = first, second
                    if not any(_ray_intersects_box(point, direction, box) for box in obstacles):
                        return True
        return False

    def is_observable(self, brick: Brick) -> bool:
        """Check the workspace and wrist views in this structure state."""
        return all(self.is_visible_from(brick, direction)
                   for direction in (WORKSPACE_VIEW_DIRECTION, WRIST_VIEW_DIRECTION))
