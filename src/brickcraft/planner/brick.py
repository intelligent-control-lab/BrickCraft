"""Validated, immutable brick geometry and task-encoding conventions."""
from dataclasses import asdict, dataclass

SHAPE_CODES = {"2x2": 0, "2x4": 1}
COLOR_CODES = {"red": "r", "yellow": "y", "blue": "b", "orange": "o"}
STUD_SPACING_MM = 16.0
BRICK_HEIGHT_MM = 19.0


@dataclass(frozen=True)
class Brick:
    brick_id: int
    shape: str
    x: int
    y: int
    z: int
    ori: int
    color: str

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Each brick needs exactly brick_id, shape, x, y, z, ori, color")
        return cls(**data)

    def __post_init__(self):
        for name in ("brick_id", "x", "y", "z", "ori"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.brick_id < 1 or self.z < 1:
            raise ValueError("brick_id and z must be positive")
        if not isinstance(self.shape, str) or self.shape not in SHAPE_CODES or self.ori not in (0, 1):
            raise ValueError("Supported shapes: 2x2, 2x4; ori must be 0 or 1")
        if not isinstance(self.color, str) or self.color not in COLOR_CODES:
            raise ValueError(f"Unsupported color {self.color!r}; use {list(COLOR_CODES)}")
        if any(not (0 <= x < 16 and 0 <= y < 16) for x, y in self.footprint):
            raise ValueError(f"Brick {self.brick_id} falls outside the 16x16 board")

    @property
    def footprint(self):
        sx, sy = (2, 2) if self.shape == "2x2" else ((4, 2) if self.ori == 0 else (2, 4))
        return frozenset((self.x + x, self.y + y) for x in range(sx) for y in range(sy))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def bounds_mm(self):
        """Axis-aligned body box; z=1 rests on the board at z_mm=0.

        Studs, recesses and manufacturing clearances are not modeled.
        """
        xs, ys = zip(*self.footprint)
        return ((min(xs) * STUD_SPACING_MM, min(ys) * STUD_SPACING_MM,
                 (self.z - 1) * BRICK_HEIGHT_MM),
                ((max(xs) + 1) * STUD_SPACING_MM, (max(ys) + 1) * STUD_SPACING_MM,
                 self.z * BRICK_HEIGHT_MM))

    def task_encoding_relative_to(self, reference: "Brick") -> tuple[int, ...]:
        """Return the single-arm 6D encoding of this target relative to reference."""
        return (self.x - reference.x, self.y - reference.y, self.z - reference.z,
                self.ori - reference.ori, SHAPE_CODES[reference.shape], SHAPE_CODES[self.shape])
