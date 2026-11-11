"""Run Kinova assembly with state identification or sequential plan execution.

    python scripts/run_pipeline.py --config configs/pipeline_config.yaml

Set plan_path to the directory containing plan.json and rendered masks, configure
checkpoints and pick poses, and manually install initial_brick before starting.

Execution modes (enable_step_identification in the YAML config):
  false (default): Execute steps 1 through N in order, then exit.
  true: Before each assembly step, observe the current brick structure and match
    it against the structure states in the assembly plan to select the next
    step. Run pick and assembly for that step. A confident match to the
    completed structure ends the run. (Step identification can misclassify structures.)

Interaction:
  A failed pick skips assembly for that attempt. After either skill fails,
    enter y to re-identify (identification mode) or skip the step (sequential
    mode); enter n to stop.
  Press Ctrl+C to stop and release resources.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default="./configs/pipeline_config.yaml",
        help="Pipeline config path (default: %(default)s)",
    )
    args = parser.parse_args()
    if not Path(args.config).is_file():
        parser.error(f"Configuration file not found: {args.config}")

    # Keep hardware imports out of --help and argument validation.
    from brickcraft.pipeline import LongHorizonPipeline, LongHorizonPipelineConfig

    config = LongHorizonPipelineConfig.from_yaml(args.config)
    LongHorizonPipeline(config).run()


if __name__ == "__main__":
    main()
