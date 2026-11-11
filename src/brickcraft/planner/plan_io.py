"""JSON validation, renderer adaptation, and plan bundle persistence."""
import json
from pathlib import Path

from .brick import Brick, COLOR_CODES
from .structure import Structure
from .assembly_planner import AssemblyPlanner


def validate_plan(plan):
    """Validate a new plan before rendering or loading it; never silently reorder."""
    if not isinstance(plan, dict) or not {"initial_brick", "steps"} <= plan.keys():
        raise ValueError("Expected a plan with initial_brick and steps")
    initial = Brick.from_dict(plan["initial_brick"])
    if initial.z != 1 or initial.ori != 0:
        raise ValueError("Initial brick must be on the board with ori=0")
    if not isinstance(plan["steps"], list):
        raise ValueError("steps must be an array")
    placed = Structure((initial,))
    for index, step in enumerate(plan["steps"]):
        if not isinstance(step, dict) or not {"step_id", "b_ref", "b_tgt", "task_encoding"} <= step.keys():
            raise ValueError(f"Step {index+1}: requires step_id, b_ref, b_tgt, and task_encoding")
        if type(step["step_id"]) is not int or step["step_id"] != index + 1:
            raise ValueError("step_id must run consecutively from 1")
        ref, target = Brick.from_dict(step["b_ref"]), Brick.from_dict(step["b_tgt"])
        if ref not in placed:
            raise ValueError(f"Step {index+1}: b_ref must match an already placed brick")
        next_structure = placed.with_brick(target)
        tau = step["task_encoding"]
        if (not isinstance(tau, list) or any(type(v) is not int for v in tau)
                or tuple(tau) != target.task_encoding_relative_to(ref)):
            expected = list(target.task_encoding_relative_to(ref))
            raise ValueError(
                f"Step {index+1}: task_encoding must match the 6D brick poses/types; "
                f"expected {expected}, got {tau}"
            )
        if not AssemblyPlanner.is_feasible_step(ref, target, placed):
            raise ValueError(f"Step {index+1}: violates released skills/geometric constraints")
        for name, expected in (("current_image", f"images/{index:03d}.png"),
                               ("goal_image", f"images/{index+1:03d}.png"),
                               ("current_mask", f"masks/{index:03d}.png")):
            if step.get(name, expected) != expected:
                raise ValueError(f"Step {index+1}: expected {name}={expected}")
        placed = next_structure
    return tuple(placed)


def plan_to_layout(plan):
    """Adapt to the existing renderer, retaining original IDs in plan.json."""
    validate_plan(plan)
    ordered = [plan["initial_brick"]] + [s["b_tgt"] for s in plan["steps"]]
    ids = {b["brick_id"]: str(i+1) for i, b in enumerate(ordered)}
    layout = {}
    for i, b in enumerate(ordered):
        entry = {**b, "type": b["shape"], "color": COLOR_CODES[b["color"]]}
        entry["ref_brick"] = None if i == 0 else ids[plan["steps"][i-1]["b_ref"]["brick_id"]]
        entry["task_encoding"] = [] if i == 0 else plan["steps"][i-1]["task_encoding"]
        layout[str(i+1)] = entry
    return layout


def layout_to_plan(layout):
    """Convert a historical layout with complete brick metadata into a plan."""
    if not isinstance(layout, dict) or not layout:
        raise ValueError("Layout must be a nonempty keyed brick mapping")
    keys = sorted(layout, key=int)
    colors = {code: name for name, code in COLOR_CODES.items()}
    bricks = {}
    for key in keys:
        entry = layout[key]
        try:
            bricks[key] = Brick.from_dict({
                "brick_id": entry.get("brick_id", int(key)),
                "shape": entry.get("shape", entry.get("type")),
                "x": entry["x"], "y": entry["y"], "z": entry["z"], "ori": entry["ori"],
                "color": colors.get(entry["color"], entry["color"]),
            })
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Layout brick {key}: complete brick metadata is required") from exc
    steps = []
    for index, key in enumerate(keys[1:]):
        entry = layout[key]
        ref_key = str(entry.get("ref_brick"))
        if ref_key not in bricks:
            raise ValueError(f"Layout brick {key}: unknown ref_brick {ref_key!r}")
        steps.append({
            "step_id": index + 1,
            "b_ref": bricks[ref_key].to_dict(), "b_tgt": bricks[key].to_dict(),
            "task_encoding": entry.get("task_encoding", []),
            "current_image": f"images/{index:03d}.png",
            "current_mask": f"masks/{index:03d}.png",
        })
    plan = {"initial_brick": bricks[keys[0]].to_dict(), "steps": steps}
    validate_plan(plan)
    return plan


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_plan(path, plan):
    """Save a validated plan and invalidate numbered renders when it changes.

    Keep unrelated files (including named illustrations) in the bundle intact.
    This prevents a planning-only update from silently using old scene images.
    """
    validate_plan(plan)
    path = Path(path)
    plan = {key: value for key, value in plan.items() if key != "schema_version"}
    plan["steps"] = [{key: value for key, value in step.items() if key != "goal_image"}
                     for step in plan["steps"]]
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if isinstance(previous, dict):
        previous.pop("schema_version", None)
        for step in previous.get("steps", []):
            step.pop("goal_image", None)
    if previous is not None and previous != plan:
        for folder in ("images", "masks"):
            for image in (path.parent / folder).glob("*.png"):
                if image.stem.isdecimal():
                    image.unlink()
        (path.parent / f"{path.parent.resolve().name}.png").unlink(missing_ok=True)
    write_json(path, plan)
