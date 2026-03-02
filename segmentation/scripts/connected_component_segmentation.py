import argparse
import shutil
from pathlib import Path
import numpy as np
from natsort import natsorted
import pyvista as pv
import pandas as pd

import networkx as nx
from scipy.spatial import cKDTree
import os
import glob
import trimesh
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors

import numpy as np
import trimesh
from scipy.interpolate import interp1d
import cv2

from scipy.spatial import KDTree



def thin_and_display_mask(mask_path: Path):
    """
    Loads a binary mask, thins it to a 1-pixel skeleton, 
    and displays the result to verify the spiral path.
    """
    # 1. Load the mask in grayscale
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask at {mask_path}")
        return

    # 2. Ensure the mask is strictly binary (0 or 255)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. Perform Thinning (Skeletonization)
    # This reduces the ribbon to a single-pixel-wide line
    skeleton = cv2.ximgproc.thinning(binary)

    return skeleton

def sample_mask_raster(mask_path: Path, n_steps: float=0.2, label: str='dummy', display_output=False):
    intersections = {}
    # 1. Load the mask and skeletonize to get a 1-pixel precision line
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Thinning ensures we only hit 1 pixel per coil crossing
    skeleton = cv2.ximgproc.thinning(binary)
    
    # Create a color version for drawing
    display = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    h, w = skeleton.shape
    
    # 2. Define the 20% height intervals (Start at 20%, end at 80%)
    # This gives us 4 distinct scan lines (20, 40, 60, 80)
    y_steps = np.arange(0.0, 1.0, n_steps) * h
    
    for i, y_float in enumerate(y_steps):
        y = int(y_float)
        
        # 3. Calculate Color Gradient (Red at top -> Blue at bottom)
        # OpenCV BGR: (Blue, Green, Red)
        fraction = i / (len(y_steps) - 1)
        color = (int(255 * fraction), 0, int(255 * (1 - fraction)))
        
        # 4. Draw the Raster Scan Line
        cv2.line(display, (0, y), (w, y), (50, 50, 50), 1)
        
        # 5. Find intersections (where the skeleton is white)
        # We look across the entire row at height 'y'
        row_pixels = skeleton[y, :]
        x_intersects = np.where(row_pixels > 0)[0]
        
        # 6. Draw circles at every intersection
        for x in x_intersects:
            cv2.circle(display, (int(x), y), 1, color, -1)
            if i not in intersections:
                intersections[i] = []
            intersections[i].append((int(x), y))

    # 7. Show in OpenCV
    if display_output:
        cv2.imshow(f"Raster Sampled Intersections for label {label}", display)
        print(f"Sampled {len(y_steps)} rows. Press any key to continue...")
        cv2.waitKey(0)
    return intersections

def match_lens(list_a, list_b):
    """
    Makes two lists the same length by repeating the last element 
    of the shorter list.
    """
    target_len = max(len(list_a), len(list_b))
    
    # Create copies so we don't change the original lists
    a = list(list_a)
    b = list(list_b)
    
    # Repeat the last ID of list_a until it matches the target
    while len(a) < target_len:
        a.append(a[-1])
        
    # Repeat the last ID of list_b until it matches the target
    while len(b) < target_len:
        b.append(b[-1])
        
    return a, b

def get_clamped_pairs(num_points):
    # This works for 11 points, 100 points, or 2 points.
    indices = []
    for i in range(num_points):
        # The first index is always i
        idx1 = i
        # The second index is i+1, but we cap it at the last available index
        idx2 = min(i + 1, num_points - 1)
        
        indices.append((idx1, idx2))
    return indices

def raster_masks_to_points(curr_mask_path, next_mask_path, curr_z, next_z, obje_file_writer, obj_idx, n_steps=0.2):
    curr_raster = sample_mask_raster(curr_mask_path, n_steps=n_steps, label="curr")
    next_raster = sample_mask_raster(next_mask_path, n_steps=n_steps, label="next")
    # print(f"Curr raster: {curr_raster}")
    # print(f"Next raster: {next_raster}")
    curr_line_keys = sorted(curr_raster.keys())
    next_line_keys = sorted(next_raster.keys())
    # print(f"Curr line keys: {len(curr_line_keys)}")
    # print(f"Next line keys: {len(next_line_keys)}")
    # Now, for each line we will match. Assuming that same number of lines yhave intersected atleast a sigle point on the film roll.
    for line_key_curr, line_key_next in zip(curr_line_keys, next_line_keys):
        list_points_curr = curr_raster[line_key_curr]
        list_points_next = next_raster[line_key_next]
        # print(f"Line {line_key_curr}: {len(list_points_curr)} points in curr, {len(list_points_next)} points in next")
        # We will match points in the current line to the next line. We will use a simple nearest neighbor approach for this.
        curr_idx_pairs = get_clamped_pairs(len(list_points_curr))
        next_idx_pairs = get_clamped_pairs(len(list_points_next))
        # print(f"Curr idx pairs: {curr_idx_pairs}, {len(curr_idx_pairs)}")
        # print(f"Next idx pairs: {next_idx_pairs}, {len(next_idx_pairs)}")
        curr_idx_pairs_normalized, next_idx_pairs_normalized = match_lens(curr_idx_pairs, next_idx_pairs)
        # print(f"After normalization, curr idx pairs: {curr_idx_pairs_normalized}, {len(curr_idx_pairs_normalized)}")
        # print(f"After normalization, next idx pairs: {next_idx_pairs_normalized}, {len(next_idx_pairs_normalized)}")
        for curr_idx_pair, next_idx_pair in zip(curr_idx_pairs_normalized, next_idx_pairs_normalized):
            curr_point1 = list_points_curr[curr_idx_pair[0]] # v1
            curr_point2 = list_points_curr[curr_idx_pair[1]] # v2
            next_point1 = list_points_next[next_idx_pair[0]] # v3
            next_point2 = list_points_next[next_idx_pair[1]] # v4
            # We will write the points to the obj file in the format: v x y z
            # Assuming you are writing a single Quad (2 triangles) between 4 points
            obje_file_writer.write(f"v {curr_point1[0]} {curr_point1[1]} {curr_z}\n") # ID: obj_idx
            obje_file_writer.write(f"v {curr_point2[0]} {curr_point2[1]} {curr_z}\n") # ID: obj_idx + 1
            obje_file_writer.write(f"v {next_point1[0]} {next_point1[1]} {next_z}\n") # ID: obj_idx + 2
            obje_file_writer.write(f"v {next_point2[0]} {next_point2[1]} {next_z}\n") # ID: obj_idx + 3

            # Standard "Z" or "N" pattern for a Quad:
            # Triangle 1: curr1 -> curr2 -> next1
            # obje_file_writer.write(f"f {obj_idx} {obj_idx+1} {obj_idx+2}\n")
            # # Triangle 2: curr2 -> next2 -> next1 (Notice the order change!)
            # obje_file_writer.write(f"f {obj_idx+1} {obj_idx+2} {obj_idx+3}\n")

            obj_idx += 4
        # obj_idx += 1
        # break
    
    return obj_idx


def mask_to_points(mask_path, z_value, spacing=1):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Optional: Skeletonize here to get the center line
    # mask = skeletonize(mask) 

    # Find coordinates of all white pixels
    y_coords, x_coords = np.where(mask > 0)
    
    # Sub-sample to keep the file size down
    x_coords = x_coords[::spacing]
    y_coords = y_coords[::spacing]
    
    # Create the Nx3 array [x, y, z]
    points = np.column_stack((x_coords, y_coords))
    return points


def parse_arguments():
    parser = argparse.ArgumentParser(description='Connected Component Segmentation')
    parser.add_argument('--input-dir', type=str, required=True, help='Path to the input image directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the output segmentation')
    return parser.parse_args()


def read_coordinates(coord_file, z_value):
    coordinates = []
    with open(coord_file, 'r') as f:
        next(f)
        for line in f:
            coords = tuple(map(float, line.strip().split(',')))
            coordinates.append((coords[0], coords[1], float(z_value)))

    return coordinates


def pre_process_files(input_dir, slice_img_dir, slice_coord_dir, slice_mask_dir):
    for slice in input_dir.iterdir():
        print(f"Processing slice: {slice.name}")
        vol_id = slice.name.split('_')[0]
        print(f"Extracted volume ID: {vol_id}")
        z_value = slice.name.split('_')[-1].split('.')[0]
        print(f"Extracted z-value: {z_value}")
        segmentation_instance = [i for i in slice.iterdir() if i.is_dir()]
        cluster_instance = [i for i in segmentation_instance[0].iterdir() if i.is_dir()]
        coordinate_instance = [i for i in cluster_instance[0].glob('*.txt')]
        print(f"Found {len(coordinate_instance)} coordinate files for z-value {z_value}.")
        coordinates = []
        for coord_file in coordinate_instance:
            coordinates.extend(read_coordinates(coord_file, z_value))
        print(f"Read {len(coordinates)} coordinates for z-value {z_value}.")
        with open(slice_coord_dir / f"{z_value}.txt", 'w') as f:
            f.write("x,y,z\n")
            for coord in coordinates:
                f.write(f"{coord[0]},{coord[1]},{coord[2]}\n")
        mask_files = glob.glob(str(cluster_instance[0] / 'total_segmentation_mask*'))
        print(mask_files)
        shutil.copy(Path(cluster_instance[0] / 'original_image.jpg'), slice_img_dir / f"{z_value}.jpg")
        skeleton = thin_and_display_mask(Path(mask_files[0]))
        cv2.imwrite(slice_mask_dir / f"{z_value}.png", skeleton)

def main():
    args = parse_arguments()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")

    SLICE_IMG_DIR = Path(output_dir / 'slice_images')
    SLICE_COORD_DIR = Path(output_dir / 'slice_coordinates')
    SLICE_MASK_DIR = Path(output_dir / 'slice_masks')

    SLICE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    SLICE_COORD_DIR.mkdir(parents=True, exist_ok=True)
    SLICE_MASK_DIR.mkdir(parents=True, exist_ok=True)

    # pre_process_files(input_dir, SLICE_IMG_DIR, SLICE_COORD_DIR, SLICE_MASK_DIR)

    all_slices = natsorted(list(SLICE_COORD_DIR.glob('*.txt')))
    all_masks = natsorted(list(SLICE_MASK_DIR.glob('*.png')))
    all_images = natsorted(list(SLICE_IMG_DIR.glob('*.jpg')))

    obj_idx = 1

    with open(output_dir / 'output.obj', 'w') as obj_file:
        for i in range(5):
            curr_z = float(all_slices[i].stem)
            next_z = float(all_slices[i+1].stem)
            obj_idx = raster_masks_to_points(all_masks[i], all_masks[i+1], curr_z, next_z, obj_file, obj_idx, n_steps=0.0001)


if __name__ == "__main__":
    main()