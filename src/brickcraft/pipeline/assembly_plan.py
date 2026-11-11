"""Validated assembly steps with full brick metadata and rendered masks."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from brickcraft.planner.brick import Brick
from brickcraft.planner.plan_io import validate_plan
from brickcraft.utils.task_encoding_utils import validate_task_encoding


@dataclass(frozen=True)
class AssemblyStep:
    """One reference-target pair, its current mask, and a binary goal silhouette."""

    # Step IDs match the plan JSON and start at 1.
    step_id: int
    b_ref: Brick
    b_tgt: Brick
    # current_image: np.ndarray
    # goal_image: np.ndarray
    current_mask: np.ndarray  # [H, W, 3] RGB, with the reference brick highlighted.
    goal_mask: np.ndarray  # [H, W] uint8, 255 on the entire post-assembly structure.
    task_encoding: np.ndarray  # [x, y, z, orientation, reference type, target type].

    def __post_init__(self):
        if type(self.step_id) is not int or self.step_id < 1:
            raise ValueError("step_id must be a positive integer starting at 1")
        if not isinstance(self.b_ref, Brick) or not isinstance(self.b_tgt, Brick):
            raise ValueError("b_ref and b_tgt must be Brick instances")
        if self.b_ref.brick_id == self.b_tgt.brick_id:
            raise ValueError("b_ref and b_tgt must have distinct brick IDs")
        encoding = validate_task_encoding(self.task_encoding, source=f"assembly step {self.step_id}")
        expected = self.b_tgt.task_encoding_relative_to(self.b_ref)
        if not np.array_equal(self.task_encoding, expected):
            raise ValueError(
                f"Assembly step {self.step_id}: task_encoding does not match b_ref/b_tgt; "
                f"expected {list(expected)}, got {self.task_encoding}"
            )
        object.__setattr__(self, "task_encoding", encoding)


PLAN_IMAGE_SIZE = (512, 512)


class AssemblyPlan:
    def __init__(self, plan_path: str):
        self.plan_path = Path(plan_path)
        self.steps = self.load_plan(self.plan_path)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def load_plan(self, plan_path: Path) -> List[AssemblyStep]:
        plan_path = Path(plan_path)
        with open(plan_path / "plan.json", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Expected a plan object with initial_brick and steps")
        new_format = "initial_brick" in data or "steps" in data
        if new_format:
            validated_data = data
            steps = data.get("steps", [])
        else:
            steps = [value for _, value in sorted(data.items(), key=lambda item: int(item[0]))]
            if not steps:
                raise ValueError("Historical keyed plans must contain at least one step")
            normalized_steps = []
            for index, step in enumerate(steps):
                if not isinstance(step, dict) or type(step.get("id")) is not int or step["id"] != index:
                    raise ValueError("Historical plan IDs must be consecutive from 0")
                validate_task_encoding(step["task_encoding"], source=f"assembly step {index}")
                if "b_ref" not in step or "b_tgt" not in step:
                    raise ValueError(
                        f"Historical assembly step {index} needs full b_ref and b_tgt metadata; "
                        "regenerate the plan from its structure"
                    )
                normalized_steps.append({
                    "step_id": index + 1,
                    "b_ref": step["b_ref"], "b_tgt": step["b_tgt"],
                    "task_encoding": step["task_encoding"],
                })
            validated_data = {"initial_brick": steps[0]["b_ref"], "steps": normalized_steps}
        validate_plan(validated_data)
        self.initial_brick = Brick.from_dict(validated_data["initial_brick"])

        def read_image(relative_path, name):
            path = plan_path / relative_path
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Failed to load {name}: {path}. Render the plan first.")
            if image.shape[:2] != PLAN_IMAGE_SIZE:
                raise ValueError(f"{name} size must be {PLAN_IMAGE_SIZE}, got {image.shape[:2]}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # images = [read_image(step.get("current_image", f"images/{index:03d}.png"), "current_image")
        #           for index, step in enumerate(steps)]
        masks = [read_image(step.get("current_mask", f"masks/{index:03d}.png"), "current_mask")
                 for index, step in enumerate(steps)]
        if steps:
            # There is a final state, but no extra assembly step for it.
            # final_image = (f"images/{len(steps):03d}.png" if new_format else
            #                steps[-1].get("goal_image", f"images/{len(steps):03d}.png"))
            # images.append(read_image(final_image, "final image"))
            masks.append(read_image(f"masks/{len(steps):03d}.png", "final structure mask"))

        result = []
        for index, step in enumerate(steps):
            # Include both the white body and the next step's reference highlight.
            goal_mask = np.any(masks[index + 1] > 30, axis=2).astype(np.uint8) * 255
            result.append(AssemblyStep(
                step_id=step["step_id"] if new_format else index + 1,
                # current_image=images[index],
                # goal_image=images[index + 1],
                current_mask=masks[index],
                goal_mask=goal_mask,
                b_ref=Brick.from_dict(step["b_ref"]),
                b_tgt=Brick.from_dict(step["b_tgt"]),
                task_encoding=step["task_encoding"],
            ))
        return result

    def get_step(self, step_id: int) -> AssemblyStep:
        if type(step_id) is not int or not 1 <= step_id <= self.num_steps:
            raise ValueError(f"Invalid step_id {step_id}. It should be in [1, {self.num_steps}].")
        return self.steps[step_id - 1]
