import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import numpy as np
import json
import random
import cv2
import bpy
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path

bproc.init()

# ==========================================
# 1. INITIALIZATION & PATHS
# ==========================================
BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / "data" / "surgical_tools_models"
CAM_FILE = BASE_DIR / "data" / "camera.json"
ROOT_OUTPUT_DIR = BASE_DIR / "output_synthetic"

# Set camera intrinsics 
# Set camera intrinsics and explicit resolution
with open(CAM_FILE, "r") as file:
    camera_params = json.load(file)

cam_w, cam_h = camera_params["width"], camera_params["height"]
K = np.array([
    [camera_params["fx"], 0, camera_params["cx"]], 
    [0, camera_params["fy"], camera_params["cy"]], 
    [0, 0, 1]
])

# Explicitly set both resolution and intrinsics
bproc.camera.set_resolution(cam_w, cam_h)
CameraUtility.set_intrinsics_from_K_matrix(K, cam_w, cam_h)

bproc.renderer.set_output_format(enable_transparency=True)
bproc.renderer.set_max_amount_of_samples(100)


def project_pt_to_2d(pt_3d):
    """Projects 3D world coordinates to 2D pixel coordinates."""
    scene = bpy.context.scene
    cam = bpy.context.scene.camera
    co_ndc = world_to_camera_view(scene, cam, pt_3d)
    return [co_ndc.x * cam_w, (1.0 - co_ndc.y) * cam_h]


# ==========================================
# 2. MAIN GENERATION LOOP
# ==========================================
def generate_dataset(tool_name, num_samples=250):
    tool_dir = MODELS_DIR / tool_name
    obj_files = list(tool_dir.glob("*.obj"))

    # Create specific subdirectory for this tool
    tool_output_dir = ROOT_OUTPUT_DIR / tool_name
    tool_output_dir.mkdir(parents=True, exist_ok=True)

    annotations = []

    # Verified Blender Vertex IDs
    if tool_name == "tweezers":
        idx_tip1 = 183922
        idx_tip2 = 113535
        idx_joint = 245256
    else:  # needle_holder
        idx_tip1 = 146985
        idx_tip2 = 78113
        idx_joint = 157620

    for i in range(num_samples):
        bproc.clean_up(clean_up_camera=False)

        # Load random articulation state version 
        selected_obj_path = random.choice(obj_files)
        tool_obj = bproc.loader.load_obj(str(selected_obj_path))[0] 
        tool_obj.set_location([0, 0, 0])

        # Domain Randomization: Perturb Materials 
        mat = tool_obj.get_materials()[0] 
        mat.set_principled_shader_value("Specular", random.uniform(0, 1)) 
        mat.set_principled_shader_value("Roughness", random.uniform(0.05, 0.4)) 
        mat.set_principled_shader_value("Metallic", 1.0) 

        # Domain Randomization: Add Point Light 
        light = bproc.types.Light() 
        light.set_type("POINT") 
        light.set_location(bproc.sampler.shell(
            center=tool_obj.get_location(),
            radius_min=1, radius_max=5,
            elevation_min=1, elevation_max=89
        )) 
        light.set_energy(random.uniform(100, 1000)) 

        # Camera Placement using Shell Sampling (Optimized Distance) 
        location = bproc.sampler.shell(
            center=tool_obj.get_location(),
            radius_min=4.0, radius_max=8.0,
            elevation_min=-90, elevation_max=90
        ) 
        lookat_point = tool_obj.get_location() + np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]) 
        rotation_matrix = bproc.camera.rotation_from_forward_vec(lookat_point - location,
                                                                 inplane_rot=np.random.uniform(-0.5, 0.5)) 
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix) 
        bproc.camera.add_camera_pose(cam2world_matrix) 

        bpy.context.view_layer.update()

        # Keypoint Projection
        bpy_obj = tool_obj.blender_obj
        p3d_tip1 = bpy_obj.matrix_world @ bpy_obj.data.vertices[idx_tip1].co
        p3d_tip2 = bpy_obj.matrix_world @ bpy_obj.data.vertices[idx_tip2].co
        p3d_joint = bpy_obj.matrix_world @ bpy_obj.data.vertices[idx_joint].co

        # Render 
        data = bproc.renderer.render() 
        color_img = data["colors"][0] 

        if color_img.shape[-1] == 4:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGRA)
        else:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        img_filename = f"{tool_name}_{i:04d}.png"
        cv2.imwrite(str(tool_output_dir / img_filename), color_img)

        annotations.append({
            "image_id": img_filename,
            "tool": tool_name,
            "keypoints": {
                "tip_1": project_pt_to_2d(p3d_tip1),
                "tip_2": project_pt_to_2d(p3d_tip2),
                "joint_or_base": project_pt_to_2d(p3d_joint)
            }
        })
        print(f"[{tool_name}] Generated {i + 1}/{num_samples}")

    with open(tool_output_dir / f"{tool_name}_annotations.json", "w") as f:
        json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    generate_dataset("tweezers", 250)
    generate_dataset("needle_holder", 250)