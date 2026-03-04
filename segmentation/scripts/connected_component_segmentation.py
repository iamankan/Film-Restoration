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
import pickle

from scipy.spatial import distance_matrix
from itertools import groupby


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
    pen = 0
    with open(coord_file, 'r') as f:
        next(f)
        for line in f:
            coords = tuple(map(float, line.strip().split(',')))
            coordinates.append((coords[0], coords[1], float(z_value), pen))
            pen = 1

    return coordinates

def calculate_euclidian_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def decide_winding(s1, s2, e1, e2):
    winding_needed = False
    s1s2 = calculate_euclidian_distance(s1[0], s1[1], s2[0], s2[1])
    s1e2 = calculate_euclidian_distance(s1[0], s1[1], e2[0], e2[1])
    e1s2 = calculate_euclidian_distance(e1[0], e1[1], s2[0], s2[1])
    e1e2 = calculate_euclidian_distance(e1[0], e1[1], e2[0], e2[1])
    min_dist = min(s1s2, s1e2, e1s2, e1e2)
    print(f"s1s2: {s1s2}, s1e2: {s1e2}, e1s2: {e1s2}, e1e2: {e1e2}, min_dist: {min_dist}")
    if min_dist == s1s2 or min_dist == e1e2:
        winding_needed = True
    return winding_needed


def build_automata_sequence(successor_map, total_nodes):
    """
    Builds a deterministic sequence from a transition table (successor_map).
    
    Args:
        successor_map (dict): e.g., {0: 1, 1: 2}
        total_nodes (int): Total number of segments in the forest
    """
    # 1. Find the Start State (q0)
    # The start is a node that is a 'Key' but never a 'Value'
    all_keys = set(successor_map.keys())
    all_values = set(successor_map.values())
    
    # Potential starts are nodes that no one points to
    start_candidates = list(all_keys - all_values)
    
    # If it's a perfect chain, there's 1 candidate. 
    # If it's a loop, we default to the first available key.
    start_node = start_candidates[0] if start_candidates else next(iter(all_keys))
    
    # 2. Execute the Automaton
    sequence = [start_node]
    visited = {start_node}
    
    # While the current state has a transition defined
    while sequence[-1] in successor_map:
        next_state = successor_map[sequence[-1]]
        
        # Safety: check for infinite loops (cycles)
        if next_state in visited:
            print(f"Loop detected at {next_state}. Terminating sequence.")
            break
            
        sequence.append(next_state)
        visited.add(next_state)
        
    return sequence


def correct_for_winding(segment_list):
    print(f'Segment list length before winding correction: {len(segment_list)}')
    n = len(segment_list)
    if n == 1 : return segment_list
    for i in range(n - 1):
        seg1 = segment_list[i]
        seg2 = segment_list[i + 1]
        # print(f'i: {i}')
        s1 = np.array(seg1[0])
        e1 = np.array(seg1[-1])
        s2 = np.array(seg2[0])
        e2 = np.array(seg2[-1])
        print(f's1: {s1}, e1: {e1}, s2: {s2}, e2: {e2}')
        if decide_winding(s1, s2, e1, e2):
            segment_list[i + 1] = segment_list[i+1][::-1]
            print(f"Winding correction applied between segments {i+1} according to {i}")
    M = np.full((n,  n), np.inf)

    for i in range(n): # Start
        for j in range(n): # End
            if i == j: continue
            start = segment_list[i][0]
            end = segment_list[j][-1]
            M[i][j] = calculate_euclidian_distance(start[0], start[1], end[0], end[1])
    print(f'M: {M}')
    # Get the top n-1 distances
    dist_list_idx = np.argsort(M, axis=None)
    row, col = np.unravel_index(dist_list_idx, M.shape)
    r, c = row[:n-1],col[:n-1]
    # print(r, c)
    final_idx = []
    successor = {}
    for i, j in zip(r,c):
        final_idx.append(int(j))
        final_idx.append(int(i))
        print(f's{j}[e{j}s{i}]e{i}')
        successor[int(j)] = int(i) # Successor of j is i. It means ej->si is a rule.
    print(f'Successor:\n{successor}')
    automata_sequence = build_automata_sequence(successor_map=successor, total_nodes=n)
    print(f'Sequence from automata: {automata_sequence}')
    # print(final_idx)
    # final_list = [i for i,_ in groupby(final_idx)]
    # print(f"final order list idx: {final_list}")
    # print(f'final_list: {final_list}')
    tmp = []
    for i in automata_sequence:
        tmp.append(segment_list[i])
    return tmp


class Segment:
    def __init__(self, idx, pointset):
        self.idx = idx
        self.points = pointset
        self.start = pointset[0]
        self.end = pointset[-1]
        self.start_out = None
        self.start_in = None
        self.end_out = None
        self.end_in = None
        self.visited = False
        self.traversed = False
    
    def get_idx(self):
        return self.idx
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end
    
    def extrapolate_vector(self, side='end', b=15, window=10):
        if side == 'end':
            pts = self.points[-window:]
            dir = np.array(pts[-1]) - np.array(pts[0])
            a = self.end
        elif side == 'start':
            pts = self.points[:window]
            dir = np.array(pts[0]) - np.array(pts[-1])
            a = self.start
        
        norm = np.linalg.norm(dir)
        if norm == 0: return a
        
        u_hat = dir / norm
        
        return a + (u_hat * b)

def build_kdtree(segments):
    tree_coords = []
    metadata = []

    for seg in segments:
        tree_coords.append(seg.start)
        metadata.append({'obj': seg, 'type': 'start'})

        tree_coords.append(seg.end)
        metadata.append({'obj': seg, 'type': 'end'})
    
    tree_coords = np.array(tree_coords)
    kdtree = KDTree(data=tree_coords)

    return kdtree, metadata

def search_in_kdtree(kdtree:KDTree, metadata, curr_seg:Segment, b, window, search_radius, k:int=1):
    if curr_seg.visited:
        return # This segment is already fully connected, skip the KD-Tree search
    
    curr_seg.visited = True
    start_extrapolation = curr_seg.extrapolate_vector(side='start', b=b, window=window)
    end_extrapolation = curr_seg.extrapolate_vector(side='end', b=b, window=window)

    dist_start, idx_start = kdtree.query(x=start_extrapolation, k=k, distance_upper_bound=search_radius)
    if dist_start != float('inf'):
        next_metadata = metadata[idx_start]
        next_seg = next_metadata['obj']
        next_side = next_metadata['type'] # start/end

        if next_seg.get_idx() != curr_seg.get_idx():
            curr_seg.start_out = (next_seg.get_idx(), next_side)
            if next_side == 'end':
                next_seg.end_in = (curr_seg.get_idx(), 'start')
            elif next_side == 'start':
                next_seg.start_in = (curr_seg.get_idx(), 'start')

            

    dist_end, idx_end = kdtree.query(x=end_extrapolation, k=k, distance_upper_bound=search_radius)
    if dist_end != float('inf'):
        next_metadata = metadata[idx_end]
        next_seg = next_metadata['obj']
        next_side = next_metadata['type'] # start/end

        if next_seg.get_idx() != curr_seg.get_idx():
            curr_seg.end_out = (next_seg.get_idx(), next_side)
            if next_side == 'end':
                next_seg.end_in = (curr_seg.get_idx(), 'end')
            elif next_side == 'start':
                next_seg.start_in = (curr_seg.get_idx(), 'end')


def order_with_winding(segment_list):
    print(f'Order with winding logic....')
    n = len(segment_list)
    if n==1 : return segment_list
    segments = [Segment(idx=i, pointset=segment_list[i]) for i in range(n)]
    # Building the KDTree
    kdtree, metadata = build_kdtree(segments=segments)
    for curr_seg in segments:
        search_in_kdtree(kdtree=kdtree, metadata=metadata, curr_seg=curr_seg, b=10, window=15, search_radius=20)
    for segment in segments:
        print(f'Printing segments from KDTREE ====================')
        print(segment.__dict__)

def pre_process_files(input_dir, slice_img_dir, slice_coord_dir, slice_mask_dir):
    all_segs = {'slices': {}}
    for it, slice in enumerate(natsorted(input_dir.iterdir())):
        if it > 1: break
        print(f"Processing slice: {slice.name}")
        vol_id = slice.name.split('_')[0]
        print(f"Extracted volume ID: {vol_id}")
        z_value = slice.name.split('_')[-1].split('.')[0]
        print(f"Extracted z-value: {z_value}")
        all_segs['slices'][str(z_value)] = []
        segmentation_instance = [i for i in slice.iterdir() if i.is_dir()]
        cluster_instance = [i for i in segmentation_instance[0].iterdir() if i.is_dir()]
        coordinate_instance = [i for i in natsorted(cluster_instance[0].glob('*.txt'))]
        print(f"Found {len(coordinate_instance)} coordinate files for z-value {z_value}.")
        coordinates = []
        tmp = []
        for coord_file in coordinate_instance:
            coord_list = read_coordinates(coord_file, z_value)
            tmp.append([t[:2] for t in coord_list])
            coordinates.extend(coord_list)
        order_with_winding(tmp)
        tmp = correct_for_winding(tmp)
        all_segs['slices'][str(z_value)] = tmp
        print(f"Read {len(coordinates)} coordinates for z-value {z_value}.")
        with open(slice_coord_dir / f"{z_value}.txt", 'w') as f:
            f.write("x,y,z,t\n")
            for coord in coordinates:
                f.write(f"{coord[0]},{coord[1]},{coord[2]},{coord[3]}\n")
        mask_files = glob.glob(str(cluster_instance[0] / 'total_segmentation_mask*'))
        print(mask_files)
        shutil.copy(Path(cluster_instance[0] / 'original_image.jpg'), slice_img_dir / f"{z_value}.jpg")
        skeleton = thin_and_display_mask(Path(mask_files[0]))
        cv2.imwrite(slice_mask_dir / f"{z_value}.png", skeleton)
    return all_segs

def save_windings(all_segs, slice_img_dir, slice_winding_dir):
    for z_value, segments in all_segs['slices'].items():
        slice_img = cv2.imread(str(slice_img_dir / f"{z_value}.jpg"))
        # print(f'Length of segments: {len(segments)}')
        merged_list = [item for sublist in segments for item in sublist]
        # print(f'Length of merged list: {len(merged_list)}')
        for i, p in enumerate(merged_list):
            cv2.circle(slice_img, (int(p[0]), int(p[1])), 3, (0, int((1-(i/len(merged_list)))*255), int((i/len(merged_list))*255)), -1) # BGR format
            if i < len(merged_list) - 1:
                # print(f'p: {p}, p+1: {merged_list[i+1]}')
                cv2.line(slice_img, (int(p[0]), int(p[1])), (int(merged_list[i+1][0]), int(merged_list[i+1][1])), (255,255,255),1)
        # for segment_points in segments:
        #     for i, segment_point in enumerate(segment_points):
        #         cv2.circle(slice_img, (int(segment_point[0]), int(segment_point[1])), 3, (0, int((1-(i/len(segment_points)))*255), int((i/len(segment_points))*255)), -1) # BGR format
        #         if i < len(segment_points) - 1:
        #             cv2.line(slice_img, (int(segment_point[0]), int(segment_point[1])), (int(segment_points[i+1][0]), int(segment_points[i+1][1])),
        #                      (255,255,255), 1)
        cv2.imwrite(str(slice_winding_dir / f"{z_value}.jpg"), slice_img)

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

    all_segs = pre_process_files(input_dir, SLICE_IMG_DIR, SLICE_COORD_DIR, SLICE_MASK_DIR)

    with open(output_dir / 'all_segs.pkl', 'wb') as f:
        pickle.dump(all_segs, f)

    # all_slices = natsorted(list(SLICE_COORD_DIR.glob('*.txt')))
    # all_masks = natsorted(list(SLICE_MASK_DIR.glob('*.png')))
    # all_images = natsorted(list(SLICE_IMG_DIR.glob('*.jpg')))

    # obj_idx = 1

    # with open(output_dir / 'output.obj', 'w') as obj_file:
    #     for i in range(5):
    #         curr_z = float(all_slices[i].stem)
    #         next_z = float(all_slices[i+1].stem)
    #         obj_idx = raster_masks_to_points(all_masks[i], all_masks[i+1], curr_z, next_z, obj_file, obj_idx, n_steps=0.0001)
    # read pickle file
    with open(output_dir / 'all_segs.pkl', 'rb') as f:
        all_segs = pickle.load(f)
    SLICE_WINDING_DIR = Path(output_dir / 'slice_winding')
    SLICE_WINDING_DIR.mkdir(parents=True, exist_ok=True)
    save_windings(all_segs, SLICE_IMG_DIR, SLICE_WINDING_DIR)





if __name__ == "__main__":
    main()