import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import numpy as np
import json
import csv
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
ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Path for the global statistics CSV file
CSV_STATS_PATH = ROOT_OUTPUT_DIR / "generation_stats.csv"

# Set camera intrinsics 
with open(CAM_FILE, "r") as file:
    camera_params = json.load(file)

cam_w, cam_h = camera_params["width"], camera_params["height"]
K = np.array([
    [camera_params["fx"], 0, camera_params["cx"]], 
    [0, camera_params["fy"], camera_params["cy"]], 
    [0, 0, 1]
])
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
# 2. MAIN GENERATION LOOP WITH STATS & SAFE BATCHING
# ==========================================
def generate_dataset(tool_name, num_samples=250):
    tool_dir = MODELS_DIR / tool_name
    obj_files = list(tool_dir.glob("*.obj"))
    
    tool_output_dir = ROOT_OUTPUT_DIR / tool_name
    tool_output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = tool_output_dir / f"{tool_name}_annotations.json"
    
    # Load existing annotations safely to prevent duplicate index names across batches
    if json_path.exists():
        with open(json_path, "r") as f:
            annotations = json.load(f)
        start_idx = len(annotations)
    else:
        annotations = []
        start_idx = 0

    # Verified Blender Vertex IDs
    if tool_name == "tweezers":
        idx_tip1 = 183922
        idx_tip2 = 113535
        idx_joint = 245256 
    else: # needle_holder
        idx_tip1 = 146985
        idx_tip2 = 78113
        idx_joint = 157620

    # Prepare or append to statistics CSV records
    csv_exists = CSV_STATS_PATH.exists()
    csv_file = open(CSV_STATS_PATH, mode="a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    
    # Write header if file is brand new
    if not csv_exists:
        csv_writer.writerow([
            "image_name", "tool", "source_obj", "light_energy", 
            "specular", "roughness", "blur_kernel", "noise_std"
        ])

    for i in range(num_samples):
        bproc.clean_up(clean_up_camera=False)
        current_idx = start_idx + i
        
        # Load random articulation state 
        selected_obj_path = random.choice(obj_files)
        tool_obj = bproc.loader.load_obj(str(selected_obj_path))[0] 
        tool_obj.set_location([0, 0, 0])
        
        # Domain Randomization: Materials 
        specular_val = random.uniform(0, 1)
        roughness_val = random.uniform(0.05, 0.4)
        mat = tool_obj.get_materials()[0] 
        mat.set_principled_shader_value("Specular", specular_val) 
        mat.set_principled_shader_value("Roughness", roughness_val) 
        mat.set_principled_shader_value("Metallic", 1.0) 
        
        # Domain Randomization: Lighting 
        light_energy = random.uniform(100, 1000)
        light = bproc.types.Light() 
        light.set_type("POINT") 
        light.set_location(bproc.sampler.shell(
            center=tool_obj.get_location(),
            radius_min=1, radius_max=5,
            elevation_min=1, elevation_max=89
        )) 
        light.set_energy(light_energy) 
        
        # Camera Placement (Optimized Distance -6.0 range) 
        location = bproc.sampler.shell(
            center=tool_obj.get_location(),
            radius_min=4.0, radius_max=8.0,
            elevation_min=-90, elevation_max=90
        ) 
        lookat_point = tool_obj.get_location() + np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]) 
        rotation_matrix = bproc.camera.rotation_from_forward_vec(lookat_point - location, inplane_rot=np.random.uniform(-0.5, 0.5)) 
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
            
        # Image-Space Augmentations
        blur_kernel = 0
        if random.random() > 0.5:
            blur_kernel = random.choice([3, 5])
            color_img = cv2.GaussianBlur(color_img, (blur_kernel, blur_kernel), 0)
            
        noise_std = 0.0
        if random.random() > 0.5:
            noise_std = random.uniform(5, 15)
            noise = np.random.normal(0, noise_std, color_img.shape).astype(np.int16)
            color_img = np.clip(color_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Save Image
        img_filename = f"{tool_name}_{current_idx:04d}.png"
        cv2.imwrite(str(tool_output_dir / img_filename), color_img)
        
        # Record Annotations
        annotations.append({
            "image_id": img_filename,
            "tool": tool_name,
            "keypoints": {
                "tip_1": project_pt_to_2d(p3d_tip1),
                "tip_2": project_pt_to_2d(p3d_tip2),
                "joint_or_base": project_pt_to_2d(p3d_joint)
            }
        })

        # Record CSV Statistics row
        csv_writer.writerow([
            img_filename, 
            tool_name, 
            selected_obj_path.name, 
            round(light_energy, 2), 
            round(specular_val, 4), 
            round(roughness_val, 4), 
            blur_kernel, 
            round(noise_std, 2)
        ])

        print(f"[{tool_name}] Generated sample {current_idx + 1}")

    csv_file.close()
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=4)

if __name__ == "__main__":
    generate_dataset("tweezers", 250)
    generate_dataset("needle_holder", 250)
