"""Generate a single-arm plan from a JSON array, optionally rendering with Blender.

Usage (run from the repository root):
  python scripts/generate_assembly_plan.py --structure configs/structures/L1.json
  python scripts/generate_assembly_plan.py --structure configs/structures/L1.json --render
Default output: configs/assembly_plans/<structure-name>/plan.json + structures.json.
With --render: also generate images/, masks/, and <plan-directory-name>.png.
Options: --output path/to/plan.json --blender BLENDER_EXECUTABLE --samples 32
Ccon/Cobs/Cop use approximate geometry; Cstab awaits StableLego integration.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brickcraft.planner import AssemblyPlanner, Structure
from brickcraft.planner.plan_io import save_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, type=Path, help="Input structure JSON array")
    parser.add_argument("--output", type=Path, help="Output JSON (default: configs/assembly_plans/<stem>/plan.json)")
    parser.add_argument("--render", action="store_true", help="Also generate images/ and masks/ with Blender")
    parser.add_argument("--blender", default="blender", help="Blender executable")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples (default: 32)")
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    output = args.output or ROOT / "configs" / "assembly_plans" / args.structure.stem / "plan.json"
    try:
        if args.render and shutil.which(args.blender) is None:
            raise ValueError(f"Blender executable not found: {args.blender}")
        structure = Structure.from_json(json.loads(args.structure.read_text(encoding="utf-8")))
        plan = AssemblyPlanner(structure).solve()
        save_plan(output, plan)
        structure_copy = output.parent / "structures.json"
        if args.structure.resolve() != structure_copy.resolve():
            shutil.copyfile(args.structure, structure_copy)
        print(f"Generated {len(plan['steps'])} steps: {output}", flush=True)
        print("Cstab: not evaluated (StableLego integration and solver dependencies pending).",
              file=sys.stderr, flush=True)
        if args.render:
            subprocess.run([args.blender, "-b", "--python-exit-code", "1", "-P",
                            str(ROOT / "scripts" / "render_assembly_plan.py"), "--",
                            "--plan", str(output.resolve()), "--samples", str(args.samples)], check=True)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Planning/rendering failed: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
