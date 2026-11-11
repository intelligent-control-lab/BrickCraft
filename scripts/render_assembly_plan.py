"""Command-line entry point for rendering assembly plans inside Blender.

Usage (run from the repository root; script arguments follow Blender's --):
  blender -b -P scripts/render_assembly_plan.py -- --plan configs/assembly_plans/L1/plan.json
Default output: images/, masks/, and <plan-directory-name>.png beside plan.json.
Options: --output_dir path/to/plan --samples 32 --save_final_blend
         --assets_dir path/to/STL_assets
Legacy input: replace --plan with --layout path/to/layout.json (6D encodings required).
"""

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(REPO_ROOT) / "src"))
DEFAULT_ASSETS_DIR = os.path.join(REPO_ROOT, "src", "brickcraft", "assets")


def parse_args():
    """Parse the arguments after Blender's ``--`` separator."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        prog="blender -b -P scripts/render_assembly_plan.py --",
        description="Render images and reference masks from a generated plan or historical layout",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--plan", help="Generated plan.json (recommended)")
    inputs.add_argument("--layout", help="Historical sequenced layout.json")
    parser.add_argument("--samples", type=int, default=32, help="Positive Cycles sample count")
    parser.add_argument("--output_dir", default=None,
                        help="Plan output directory (default: the input file's directory)")
    parser.add_argument("--assets_dir", default=DEFAULT_ASSETS_DIR,
                        help="Directory with assembly_board.STL, brick_2x2.STL, brick_2x4.STL "
                             "(default: %(default)s)")
    parser.add_argument("--save_final_blend", action="store_true",
                        help="Also save the final assembly as final_assembly.blend")
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    return args


def main():
    """Main function to render plan assembly steps."""
    args = parse_args()
    from brickcraft.blender.assembly_plan_renderer import AssemblyPlanRenderer, RenderingConfig

    source = args.plan or args.layout
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(source))

    renderer = AssemblyPlanRenderer(RenderingConfig(samples=args.samples))
    renderer.render_plan(
        source,
        output_dir,
        os.path.join(args.assets_dir, "assembly_board.STL"),
        os.path.join(args.assets_dir, "brick_2x2.STL"),
        save_final_blend=args.save_final_blend,
        brick_2x4_path=os.path.join(args.assets_dir, "brick_2x4.STL"),
        is_plan=bool(args.plan),
    )

if __name__ == "__main__":
    main()
