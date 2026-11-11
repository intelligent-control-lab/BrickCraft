"""Blender implementation for assembly-plan images, masks and final previews.

Run through scripts/render_assembly_plan.py inside Blender. This module requires
bpy; it is deliberately not imported by brickcraft or brickcraft.blender.
"""
import json
import os
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import bpy
import mathutils

from brickcraft.planner.plan_io import layout_to_plan, plan_to_layout, save_plan, write_json

@dataclass
class RenderingConfig:
    board_color: tuple[float, float, float, float] = (0.3, 0.8, 0.2, 1.0) # R, G, B, A
    brick_colors: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        "y": (0.9, 0.75, 0.1, 1.0),
        "r": (0.8, 0.1, 0.1, 1.0),
        "b": (0.0, 0.1, 0.9, 1.0),
        "o": (1.0, 0.36, 0.0, 1.0),
    })
    board_size: list[float] = field(default_factory=lambda: [256, 256, 1.5]) # brick board size in mm
    board_location: list[float] = field(default_factory=lambda: [128, 128, 0])
    stud_spacing: float = 16.0  # Distance between studs in mm
    brick_height: float = 19.0  # Height of a standard brick in mm

    # rendering settings
    samples: int = 32
    resolution_x: int = 512
    resolution_y: int = 512
    exposure: float = 1.0

    # camera settings
    camera_angle_x: float = 69 # realsense D415
    camera_angle_y: float = 42
    camera_position: tuple[float, float, float] = (128, 400, 75)
    camera_target: tuple[float, float, float] = (128, 0, 75)

    # lighting settings
    lights: dict[str, dict] = field(default_factory=lambda: {
        "top_spotlight": {
            "type": "SPOT",
            "energy": 20000.0,
            "location": (128, 128, 400),
            "rotation": (0, 0, 0)  # degrees
        },
        "side_light": {
            "type": "AREA",
            "energy": 20000.0,
            "size": 100.0,
            "location": (-128, 128, 256),
            "rotation": (0, -45, 0)
        },
    })

    # background settings
    background_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # R, G, B, A
    background_strength: float = 0.5


class AssemblyPlanRenderer:
    def __init__(self, config: RenderingConfig = RenderingConfig()):
        self.config = config
        self.board = None
        self.brick_templates = {}
        self.materials_by_code = {}
        self.placed_bricks = {}

    def _clear_scene(self):
        """Clear all objects in the scene."""
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

    def _configure_units(self):
        """Configure Blender units to millimeters."""
        scene = bpy.context.scene
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.length_unit = "MILLIMETERS"
        scene.unit_settings.scale_length = 0.001  # 1 BU = 1 mm

    def _import_stl(self, path, name):
        """Import STL file and name the object."""
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=path)
        else:
            try:
                bpy.ops.preferences.addon_enable(module="io_mesh_stl")
            except Exception:
                pass
            bpy.ops.import_mesh.stl(filepath=path)
        obj = bpy.context.selected_objects[0]
        obj.name = name
        return obj

    def _get_or_create_material(self, name, color):
        """Get or create a material with the given name and color."""
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            if bsdf is not None:
                # Set base color
                bsdf.inputs["Base Color"].default_value = color
                # Set metallic property for slight metallic look
                bsdf.inputs["Metallic"].default_value = 0.1
                # Adjust other properties for better appearance
                bsdf.inputs["Roughness"].default_value = 0.3
                bsdf.inputs["Specular IOR Level"].default_value = 1.5
            # Set viewport display color
            mat.diffuse_color = color
        return mat

    def _apply_materials(self, obj, mat):
        """Apply material to object."""
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    def _add_camera(self, location: tuple[float, float, float], target: tuple[float, float, float]):
        """Add and configure camera."""
        cam_data = bpy.data.cameras.new(name="Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)
        cam.location = location

        direction = mathutils.Vector(target) - cam.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam.rotation_euler = rot_quat.to_euler()
        cam.data.angle_x = math.radians(self.config.camera_angle_x)
        cam.data.angle_y = math.radians(self.config.camera_angle_y)
        bpy.context.scene.camera = cam
        return cam

    def _add_lightings(self):
        """Add lightings based on config."""
        created_lights = []

        for light_name, light_config in self.config.lights.items():
            light_data = bpy.data.lights.new(
                name=light_name,
                type=light_config.get("type")
            )
            light_data.energy = light_config.get("energy")

            if "size" in light_config:
                light_data.size = light_config.get("size")

            light_obj = bpy.data.objects.new(name=light_name, object_data=light_data)
            bpy.context.collection.objects.link(light_obj)

            location = light_config.get("location")
            light_obj.location = location

            rotation = light_config.get("rotation")
            light_obj.rotation_euler = (
                math.radians(rotation[0]),
                math.radians(rotation[1]),
                math.radians(rotation[2])
            )

            created_lights.append(light_obj)
            print(f"Created light: {light_name} at {location}")

        return created_lights

    def _render_image(self, output_path: str):
        """Render a image."""
        scene = bpy.context.scene
        scene.use_nodes = True
        scene.render.engine = "CYCLES"
        scene.render.resolution_x = self.config.resolution_x
        scene.render.resolution_y = self.config.resolution_y
        scene.render.resolution_percentage = 100

        # Cycles settings
        scene.cycles.samples = self.config.samples
        scene.cycles.use_denoising = True  # Enable denoising

        scene.view_settings.view_transform = "Standard"
        scene.view_settings.exposure = self.config.exposure

        # Setup color space
        scene.sequencer_colorspace_settings.name = "sRGB"

        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

    @staticmethod
    def _get_brick_type(entry: dict):
        """Return the brick type while keeping legacy layouts as 2x2."""
        brick_type = entry.get("type", "2x2")
        if brick_type not in {"2x2", "2x4"}:
            raise ValueError(f"Unsupported brick type: {brick_type}")
        return brick_type

    @staticmethod
    def _get_brick_orientation(entry: dict):
        """Return the brick orientation index (0 or 1)."""
        orientation = int(entry.get("ori", 0))
        if orientation not in {0, 1}:
            raise ValueError(
                f"Unsupported brick orientation: {orientation}. Expected 0 or 1."
            )
        return orientation

    def _get_brick_position(self, entry: dict):
        """
        Calculate the STL origin position from the lower-left footprint stud.

        A 2x4 brick occupies 4x2 studs for ori=0 and 2x4 studs for ori=1.
        Legacy entries without type/ori remain 2x2 bricks at their old positions.
        """
        brick_type = self._get_brick_type(entry)
        orientation = self._get_brick_orientation(entry)

        if brick_type == "2x4" and orientation == 0:
            size_x, size_y = 4, 2
        elif brick_type == "2x4":
            size_x, size_y = 2, 4
        else:
            size_x, size_y = 2, 2

        x_mm = (entry["x"] + size_x / 2) * self.config.stud_spacing
        y_mm = (entry["y"] + size_y / 2) * self.config.stud_spacing
        z_mm = (entry["z"] - 1) * self.config.brick_height
        return (x_mm, y_mm, z_mm)

    def _place_brick(self, entry: dict, key, *, name_suffix="", z_override=None):
        """Copy and place the template selected by an entry's brick type."""
        brick_type = self._get_brick_type(entry)
        template = self.brick_templates.get(brick_type)
        if template is None:
            raise ValueError(f"No template loaded for brick type: {brick_type}")

        brick = template.copy()
        brick.data = template.data.copy()
        brick.name = f"Brick_{brick_type}_{key}{name_suffix}"
        bpy.context.collection.objects.link(brick)
        brick.hide_viewport = False
        brick.hide_render = False

        mat = self.materials_by_code.get(entry.get("color"))
        if mat is not None:
            self._apply_materials(brick, mat)

        x_mm, y_mm, z_mm = self._get_brick_position(entry)
        brick.location = (x_mm, y_mm, z_mm if z_override is None else z_override)
        orientation = self._get_brick_orientation(entry)
        if brick_type == "2x4":
            # The 2x4 STL's long axis is Y at zero rotation.
            brick.rotation_euler = (0.0, 0.0, math.radians(90.0 * (orientation - 1)))
        else:
            brick.rotation_euler = (0.0, 0.0, math.radians(90.0 * orientation))
        return brick

    def _create_black_background(self):
        """Create or get a world with black background for mask rendering."""
        world = bpy.data.worlds.get("Mask_World")
        if world is None:
            world = bpy.data.worlds.new("Mask_World")
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
        bg = nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        output = nodes.new("ShaderNodeOutputWorld")
        links.new(bg.outputs["Background"], output.inputs["Surface"])
        return world

    def _create_white_mask(self):
        """Create or get a white emissive material for mask rendering."""
        mat = bpy.data.materials.get("Mask_White")
        if mat is None:
            mat = bpy.data.materials.new(name="Mask_White")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            output = nodes.new("ShaderNodeOutputMaterial")
            links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return mat

    def _render_combined_mask(self, all_bricks: list, ref_brick, output_path: str):
        """
        Render a combined mask where:
        - Background is black
        - All placed bricks are white
        - Reference brick is filled with red color, when provided

        Args:
            all_bricks: List of all brick objects placed so far
            ref_brick: The reference brick object (or None)
            output_path: Path to save the mask image
        """
        scene = bpy.context.scene

        # Hide board and all non-brick objects
        original_hide = {}
        for obj in scene.objects:
            if obj.type == "MESH":
                original_hide[obj] = obj.hide_render
                if obj not in all_bricks:
                    obj.hide_render = True

        # Save original world and materials
        original_world = scene.world
        scene.world = self._create_black_background()

        white_mat = self._create_white_mask()
        original_mats = {}

        # Create red emissive material for reference brick
        ref_brick_color = (1.0, 0.0, 0.0, 1.0)  # red
        red_mat = bpy.data.materials.new(name=f"Ref_Mask_Red")
        red_mat.use_nodes = True
        nodes = red_mat.node_tree.nodes
        links = red_mat.node_tree.links
        nodes.clear()
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = ref_brick_color
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(emission.outputs["Emission"], output.inputs["Surface"])

        # Apply materials: white for all bricks, red for reference brick
        for brick in all_bricks:
            original_mats[brick] = [slot.material for slot in brick.material_slots]

            # Choose material based on whether it's the reference brick
            target_mat = red_mat if brick == ref_brick else white_mat

            if brick.material_slots:
                for slot in brick.material_slots:
                    slot.material = target_mat
            else:
                brick.data.materials.append(target_mat)

        # Emission-only masks need no denoising or path-tracing noise.
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.exposure = 0
        scene.render.resolution_x = self.config.resolution_x
        scene.render.resolution_y = self.config.resolution_y
        scene.render.resolution_percentage = 100
        # Render the combined mask in one pass
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        # Restore original materials and visibility
        for obj, was_hidden in original_hide.items():
            obj.hide_render = was_hidden

        for brick in all_bricks:
            if brick in original_mats:
                for slot, mat in zip(brick.material_slots, original_mats[brick]):
                    slot.material = mat

        scene.world = original_world

        # Clean up the temporary red material
        bpy.data.materials.remove(red_mat)

    def _setup_scene(self, board_path: str, brick_path: str, brick_2x4_path: str = None):
        """
        Setup the initial scene with assembly board and brick template. Setup lighting and camera.
        Args:
            board_path: Path to the board STL file
            brick_path: Path to the 2x2 brick STL file
            brick_2x4_path: Path to the 2x4 brick STL file. Defaults to the
                brick_2x4.STL next to brick_path.
        """
        self._configure_units()
        self._clear_scene()

        # Import and setup Assembly board
        self.board = self._import_stl(board_path, "Assembly_Board")
        self.board.location = (
            self.config.board_location[0],
            self.config.board_location[1],
            self.config.board_location[2],
        )
        board_mat = self._get_or_create_material("Assembly_Board_Green", self.config.board_color)
        self._apply_materials(self.board, board_mat)

        # Import and setup brick templates
        if brick_2x4_path is None:
            brick_2x4_path = os.path.join(os.path.dirname(brick_path), "brick_2x4.STL")
        brick_paths = {
            "2x2": brick_path,
            "2x4": brick_2x4_path,
        }
        self.brick_templates = {}
        for brick_type, path in brick_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Brick STL not found for {brick_type}: {path}")
            template = self._import_stl(path, f"Brick_{brick_type}_template")
            template.location = (0, 0, 0)
            template.hide_viewport = True
            template.hide_render = True
            self.brick_templates[brick_type] = template

        # Create materials for bricks
        self.materials_by_code = {
            code: self._get_or_create_material(f"Lego_Brick_{code}", color)
            for code, color in self.config.brick_colors.items()
        }

        # Setup lighting and camera
        self._add_lightings()
        self._add_camera(
            mathutils.Vector(self.config.camera_position),
            mathutils.Vector(self.config.camera_target),
        )

        # Setup world background
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs["Color"].default_value = self.config.background_color
            bg_node.inputs["Strength"].default_value = self.config.background_strength

    def render_images(self, layout: dict, output_dir: str):
        """
        Render colored images of each assembly step based on the layout.
        Args:
            layout: A dictionary where each key is a step number (as string) and the value is a dict with:
                - x: X position in blocks (starts from 0)
                - y: Y position in blocks (starts from 0)
                - z: Z level (starts from 1)
                - color: Color code of the brick "r", "g", "b", "y"...
                - ref_brick: Reference brick id for this step (can be "None")
        """
        os.makedirs(output_dir, exist_ok=True)
        self.placed_bricks = {}
        ordered_keys = sorted(layout.keys(), key=lambda k: int(k))


        for idx, key in enumerate(ordered_keys):
            entry = layout[key]
            brick = self._place_brick(entry, key)
            self.placed_bricks[str(key)] = brick
            self._render_image(os.path.join(output_dir, f"{idx:03d}.png"))

    def render_images_addbrick(self, layout: dict, output_dir: str):
        """
        Render colored images of each assembly step with the next brick floating above.
        The next brick is shown hovering 2 brick heights above its target position.
        Args:
            layout: A dictionary where each key is a step number (as string) and the value is a dict with:
                - x: X position in blocks (starts from 0)
                - y: Y position in blocks (starts from 0)
                - z: Z level (starts from 1)
                - color: Color code of the brick "r", "g", "b", "y"...
                - ref_brick: Reference brick id for this step (can be "None")
        """
        os.makedirs(output_dir, exist_ok=True)
        self.placed_bricks = {}
        ordered_keys = sorted(layout.keys(), key=lambda k: int(k))

        for idx, key in enumerate(ordered_keys):
            entry = layout[key]

            # Add the current brick to the scene
            brick = self._place_brick(entry, key)
            self.placed_bricks[str(key)] = brick

            # If there's a next brick, show it floating above its target position
            next_brick_obj = None
            if idx < len(ordered_keys) - 1:
                next_key = ordered_keys[idx + 1]
                next_entry = layout[next_key]
                _, _, next_z_mm = self._get_brick_position(next_entry)

                # Add 2 brick heights to make it float
                floating_z = next_z_mm + 2 * self.config.brick_height

                next_brick_obj = self._place_brick(
                    next_entry,
                    next_key,
                    name_suffix="_floating",
                    z_override=floating_z,
                )

            # Render the image with the floating next brick
            self._render_image(os.path.join(output_dir, f"{idx:03d}.png"))

            # Remove the floating brick after rendering
            if next_brick_obj is not None:
                bpy.data.objects.remove(next_brick_obj, do_unlink=True)


    def render_masks(self, layout: dict, output_dir: str):
        """
        Render every structure state, including the completed assembly.
        Current-state masks highlight the next step's reference brick in red;
        the final mask has only white bricks on black, with no reference highlight.

        Args:
            layout: A dictionary where each key is a step number (as string) and the value is a dict with:
                - x: X position in blocks (starts from 0)
                - y: Y position in blocks (starts from 0)
                - z: Z level (starts from 1)
                - color: Color code of the brick "r", "g", "b", "y"...
                - ref_brick: Reference brick id for this step (can be "None")
        """
        os.makedirs(output_dir, exist_ok=True)
        self.placed_bricks = {}
        ordered_keys = sorted(layout.keys(), key=lambda k: int(k))

        # First, place the first brick
        if len(ordered_keys) > 0:
            first_key = ordered_keys[0]
            first_entry = layout[first_key]
            brick = self._place_brick(first_entry, first_key)
            self.placed_bricks[str(first_key)] = brick

        # For step i, show the structure from step i-1 and highlight the reference brick
        for mask_idx, key in enumerate(ordered_keys[1:]):
            entry = layout[key]

            # Get reference brick for this step (from already placed bricks)
            ref_brick_key = entry.get("ref_brick")
            ref_brick_obj = None
            if ref_brick_key not in (None, "", "None"):
                ref_brick_obj = self.placed_bricks.get(str(ref_brick_key))

            # Render combined mask with current placed bricks and reference brick
            mask_path = os.path.join(output_dir, f"{mask_idx:03d}.png")
            self._render_combined_mask(list(self.placed_bricks.values()), ref_brick_obj, mask_path)

            # After rendering the mask, add the current brick to placed_bricks for next iteration
            brick = self._place_brick(entry, key)
            self.placed_bricks[str(key)] = brick

        if ordered_keys:
            final_path = os.path.join(output_dir, f"{len(ordered_keys) - 1:03d}.png")
            self._render_combined_mask(list(self.placed_bricks.values()), None, final_path)

    def write_plan_json(self, layout: dict, output_path: str):
        """Write a validated plan with full brick metadata, keeping fresh renders."""
        write_json(output_path, layout_to_plan(layout))
        print(f" Plan JSON saved to: {output_path}")

    def _save_blend_file(self, filepath: str):
        """Save the current scene as a .blend file.

        Args:
            filepath: Path to save the .blend file (should end with .blend)
        """
        # Ensure the directory exists
        dir_path = os.path.dirname(os.path.abspath(filepath))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Add .blend extension if not present
        if not filepath.endswith('.blend'):
            filepath += '.blend'

        # Save the file
        bpy.ops.wm.save_as_mainfile(filepath=filepath)
        print(f"Saved Blender file: {filepath}")
        return filepath

    def render_plan(
        self,
        layout_path,
        output_dir,
        board_path,
        brick_path,
        save_final_blend=False,
        brick_2x4_path=None,
        is_plan=False,
    ):
        """
        Render step images, reference masks and a final preview from a plan or legacy layout.

        Args:
            layout_path: Path to a generated plan or historical layout JSON file
            output_dir: Directory to save the plan
            board_path: Path to the board STL file
            brick_path: Path to the 2x2 brick STL file
            save_final_blend: If True, saves the final assembly as a .blend file
            brick_2x4_path: Optional path to the 2x4 brick STL file
            is_plan: True for generated plans, False for historical layouts
        """
        # Load layout
        with open(layout_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        layout = plan_to_layout(data) if is_plan else data
        if not is_plan:
            layout_to_plan(layout)
        if is_plan:
            plan_json_path = (Path(layout_path) if Path(output_dir).resolve() == Path(layout_path).resolve().parent
                              else Path(output_dir) / "plan.json")
            save_plan(plan_json_path, data)
            structure_source = Path(layout_path).parent / "structures.json"
            structure_copy = Path(output_dir) / "structures.json"
            if structure_source.is_file() and structure_source.resolve() != structure_copy.resolve():
                shutil.copyfile(structure_source, structure_copy)

        # Render assembly images
        self._setup_scene(board_path, brick_path, brick_2x4_path)
        images_dir = os.path.join(output_dir, "images")
        self.render_images(layout, images_dir)

        # Save the final assembly as a .blend file
        if save_final_blend:
            blend_path = os.path.join(output_dir, "final_assembly.blend")
            self._save_blend_file(blend_path)
            print(f"Final assembly saved as: {blend_path}")

        # Reset the scene and generate combined masks
        self._setup_scene(board_path, brick_path, brick_2x4_path)
        masks_dir = os.path.join(output_dir, "masks")
        self.render_masks(layout, masks_dir)

        # Keep a convenient final-structure preview alongside the plan.
        final_image = Path(images_dir) / f"{len(layout) - 1:03d}.png"
        shutil.copyfile(final_image, Path(output_dir) / f"{Path(output_dir).resolve().name}.png")

        # Historical layouts still need conversion to runtime JSON.
        if not is_plan:
            self.write_plan_json(layout, os.path.join(output_dir, "plan.json"))

        # Preserve the floating-brick views for historical layout inputs.
        if not is_plan:
            self._setup_scene(board_path, brick_path, brick_2x4_path)
            images_addbrick_dir = os.path.join(output_dir, "images_addbrick")
            self.render_images_addbrick(layout, images_addbrick_dir)
