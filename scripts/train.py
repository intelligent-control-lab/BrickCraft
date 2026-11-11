#!/usr/bin/env python
"""Train script that auto-registers brick_diffusion before calling LeRobot."""

import sys

# Auto-register brick_diffusion into LeRobot's policy factory and config registry.
import brickcraft.policies  # noqa: F401

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    sys.exit(main())
