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

from pprint import pprint
import math
import imageio.v3 as iio

import json

from scipy.spatial import Delaunay

import matplotlib.pyplot as plt

import trimesh

def read_coordinates(coord_file, z_value):
    coordinates = []
    with open(coord_file, 'r') as f:
        next(f)
        for line in f:
            coords = tuple(map(float, line.strip().split(',')))
            coordinates.append((coords[0], coords[1]))
    return coordinates


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
    
    def get_num_points(self):
        return len(self.points)
    
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
    
    def __repr__(self):
        display_dict = {k: v for k, v in self.__dict__.items() if k != 'points'}
        display_dict['points_count'] = len(self.points)
        attrs = ", ".join(f"{k}={v}" for k, v in display_dict.items())
        return f"Segment({attrs})"

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

def traverse_segments(segments):
    valid_segments = []
    invalid_segments = []
    terminal_segments = []

    globally_ordered_slice_segment = []
    for seg in segments:
        if seg.start_out == None and seg.start_in == None and seg.end_out == None and seg.end_in == None:
            invalid_segments.append(seg)
        else:
            valid_segments.append(seg)
    
    print(f'# of valid segments: {len(valid_segments)}, # of invalid segments: {len(invalid_segments)}')
    if len(valid_segments) == 0 and len(invalid_segments) == len(segments):
        segLen = []
        print(f'Length of invalid segment: {len(invalid_segments)}')
        for seg in invalid_segments:
            print(seg.get_num_points)
            segLen.append(seg.get_num_points())
        print(f'SegLen: {segLen}')
        seg_idx = np.argmax(np.array(segLen))
        print(f'Seg-idx: {seg_idx}')
        return invalid_segments[seg_idx].points


    

    for seg in valid_segments:
        if (seg.start_out == None and seg.start_in == None) or (seg.end_out == None and seg.end_in == None):
            terminal_segments.append(seg)

    
    if len(terminal_segments) < 2 or len(valid_segments) == 0:
        print("No valid segments...Maybe its a cycle...Maybe a loop...")
        return
    
    head = terminal_segments[0]
    tail = terminal_segments[1]

    start = True
    end = False

    while not tail.traversed:
        curr_segment = head
        curr_segment.traversed = True
        next_reverse_order = False

        if start:
            curr_reverse_order = False

            if curr_segment.start_out is not None:
                curr_reverse_order = True
                next_segment_idx, next_segment_recieving_type = curr_segment.start_out
                if next_segment_recieving_type == 'end':
                    next_reverse_order = True

            if curr_segment.end_out is not None:
                next_segment_idx, next_segment_recieving_type = curr_segment.end_out
                if next_segment_recieving_type == 'end':
                    next_reverse_order = True
        else:
            if curr_reverse_order:
                if curr_segment.start_out:
                    next_segment_idx, next_segment_recieving_type = curr_segment.start_out
                    if next_segment_recieving_type == 'end':
                        next_reverse_order = True
                else:
                    end = True
            else:
                if curr_segment.end_out:
                    next_segment_idx, next_segment_recieving_type = curr_segment.end_out
                    if next_segment_recieving_type == 'end':
                        next_reverse_order = True
                else:
                    end = True
        
        next_segment = segments[next_segment_idx]
        if start:
            globally_ordered_slice_segment.extend(curr_segment.points[::-1] if curr_reverse_order else curr_segment.points)
        
        if not end:
            globally_ordered_slice_segment.extend(next_segment.points[::-1] if next_reverse_order else next_segment.points)
        head = next_segment
        start = False
        print(f'Reverse order: Current: {curr_reverse_order}, Next: {next_reverse_order}')
        curr_reverse_order = next_reverse_order
    
    return globally_ordered_slice_segment
        

def order_with_winding(segment_list, b=40, window=50, search_radius=50):
    print(f'Order with winding logic....')
    n = len(segment_list)
    if n==1 : return segment_list[0]
    segments = [Segment(idx=i, pointset=segment_list[i]) for i in range(n)]
    # Building the KDTree
    kdtree, metadata = build_kdtree(segments=segments)
    for curr_seg in segments:
        search_in_kdtree(kdtree=kdtree, metadata=metadata, curr_seg=curr_seg, b=b, window=window, search_radius=search_radius)
    pprint(segments)
    return traverse_segments(segments=segments)
    
    

def pre_process_files(input_dir, slice_img_dir, b=40, window=50, search_radius=50):
    ordered_segments = {'slices': {}}
    for it, slice in enumerate(natsorted(input_dir.iterdir())):
        # if it > 10: break
        print(f"Processing slice: {slice.name}")
        vol_id = slice.name.split('_')[0]
        print(f"Extracted volume ID: {vol_id}")
        z_value = slice.name.split('_')[-1].split('.')[0]
        print(f"Extracted z-value: {z_value}")
        
        segmentation_instance = [i for i in slice.iterdir() if i.is_dir()]
        cluster_instance = [i for i in segmentation_instance[0].iterdir() if i.is_dir()]
        coordinate_instance = [i for i in natsorted(cluster_instance[0].glob('*.txt'))]

        print(f"Found {len(coordinate_instance)} coordinate files for z-value {z_value}.")

        tmp = []
        for coord_file in coordinate_instance:
            coord_list = read_coordinates(coord_file, z_value)
            tmp.append(coord_list)
        globally_ordered_slice_segment = order_with_winding(tmp,  b=b, window=window, search_radius=search_radius)
        print("Ordered segments before putting side pickel")
        print(f"Len: {len(globally_ordered_slice_segment)}")
        ordered_segments['slices'][str(z_value)] = [globally_ordered_slice_segment]
        print(f"After putting in the dict, the length is {len(ordered_segments['slices'][str(z_value)])}")
        shutil.copy2(Path(f'{cluster_instance[0]}/original_image.jpg'), Path(f'{slice_img_dir}/{z_value}.jpg'))
        
        
    return ordered_segments




def save_windings(all_segs, slice_img_dir, slice_winding_dir):
    for z_value, segments in all_segs['slices'].items():
        slice_img = cv2.imread(str(slice_img_dir / f"{z_value}.jpg"))
        print(f'Length of segments: {len(segments)}')
        if len(segments) > 1:
            merged_list = [item for sublist in segments for item in sublist]
        elif len(segments) == 1:
            merged_list = segments[0]
        elif len(segments) == 0:
            print(f'Sorry no segments present. Cannot draw.')
            continue
        print(f'Length of merged list: {len(merged_list)}')
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


def thin_and_display_mask(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask at {mask_path}")
        return

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    skeleton = cv2.ximgproc.thinning(binary)

    return skeleton


def match_lengths(list_a, list_b):
    target_len = max(len(list_a), len(list_b))
    a = list(list_a)
    b = list(list_b)
    while len(a) < target_len:
        a.append(a[-1])
    while len(b) < target_len:
        b.append(b[-1])
        
    return a, b

def get_clamped_pairs(num_points):
    indices = []
    for i in range(num_points):
        idx1 = i
        idx2 = min(i + 1, num_points - 1)
        indices.append((idx1, idx2))
    return indices


def calculate_normal(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    n = np.cross(v1, v2)
    return (n / np.linalg.norm(n)).tolist()

def calculate_vector(a,b):
    return (np.array(b) - np.array(a)).tolist()

def meshify(list_curr_idx, list_next_idx, fid, curr_coords, next_coords, z_value, curr_start, next_start):
    for p in next_coords:
        fid.write(f'v {p[0]} {p[1]} {z_value + 1}\n')
        
    for i in range(len(list_curr_idx) - 1):
        
        c1, c2 = list_curr_idx[i][0] + curr_start, list_curr_idx[i][1] + curr_start
        n1, n2 = list_next_idx[i][0] + next_start, list_next_idx[i][1] + next_start
        
        if c1 != n1 and n1 != c2 and c1 != c2:
            fid.write(f'f {c1} {n1} {c2}\n')
            
        if c2 != n1 and n1 != n2 and c2 != n2:
            fid.write(f'f {c2} {n1} {n2}\n')


        
def make_meshes(segmentation_file, mesh_file, h, w):

    with open(segmentation_file, 'rb') as f:
        ordered_segments = pickle.load(f)
    
    z_slices = sorted(ordered_segments['slices'].keys())
    
    with open(mesh_file, 'w') as fid:
        
        first_z = z_slices[0]
        
        first_coords = ordered_segments['slices'][first_z][0]
        
        for p in first_coords:
            fid.write(f'v {p[0]} {p[1]} {float(first_z)}\n')
            
        curr_start = 1 
        
        
        for i in range(len(z_slices) - 1):
            curr_z = z_slices[i]
            next_z = z_slices[i+1]
            
            curr_coords = ordered_segments['slices'][curr_z][0]
            next_coords = ordered_segments['slices'][next_z][0]
            
            list_curr_idx = get_clamped_pairs(len(ordered_segments['slices'][curr_z][0]))
            list_next_idx = get_clamped_pairs(len(ordered_segments['slices'][next_z][0]))

            list_curr_idx, list_next_idx = match_lengths(list_curr_idx, list_next_idx)
            
            next_start = curr_start + len(curr_coords)
            
            meshify(list_curr_idx, list_next_idx, fid, 
                    curr_coords, next_coords, 
                    float(curr_z), curr_start, next_start)
            
            curr_start = next_start

    print(f"Mesh saved to {mesh_file}")


def calculate_euclidian_distance(x1, x2, y1, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def populate_uv(uv_mapping, z_idx, delta, segment, mid_idx, z_val):
    num_points = len(segment)
    start_idx = 0
    uv_mapping[z_idx][start_idx] = (int(z_val),segment[mid_idx])
    # March left
    left_idx = mid_idx - delta
    how_far = -1
    while left_idx >= 0:
        uv_mapping[z_idx][how_far] = (int(z_val), segment[left_idx])
        left_idx = left_idx - delta
        how_far = how_far - 1
    
    # March right
    right_idx = mid_idx + delta
    how_far = 1
    while right_idx < num_points:
        uv_mapping[z_idx][how_far] = (int(z_val), segment[right_idx])
        right_idx = right_idx + delta
        how_far = how_far + 1

def write_obj_uv(filename, vertices, faces):
    with open(filename, 'w') as f:
        # Write all vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} 0.000000\n")
        
        # Write all faces (1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def write_obj_xyz(filename, vertices, faces, uv_map):
    with open(filename, 'w') as f:
        # Write all vertices
        for v in vertices:
            slice_no, xy_coord = uv_map[v[1]][v[0]]
            f.write(f"v {float(xy_coord[0])} {float(xy_coord[1])} {float(slice_no)}\n")
        
        # Write all faces (1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def create_uv_mesh(segmentation_file, delta=5):
    with open(segmentation_file, 'rb') as f:
        ordered_segments = pickle.load(f)
    
    ordered_segments = ordered_segments['slices']
    
    slice_list = sorted(list(ordered_segments.keys()))

    uv_mapping = {int(i):{} for i, _ in enumerate(slice_list)}
    
    delta = delta
    z_idx = 0
    z_val = slice_list[0]
    segment = ordered_segments[z_val][0]

    num_points = len(segment)
    mid_idx = num_points//2

    populate_uv(uv_mapping, z_idx, delta, segment, mid_idx, z_val)
    prev_mid_idx = mid_idx
    prev_segment = segment

    for z_idx, z_val in enumerate(slice_list):
        if z_idx == 0: continue
        print(f'i: {z_idx}, z_val: {z_val}')
        curr_segment = ordered_segments[z_val][0]
        segment_np = np.array(curr_segment)
        segment_tree = KDTree(data=segment_np)
        _, curr_mid_idx = segment_tree.query(x=np.array(uv_mapping[z_idx-1][0][1]), k =1)
        print(f'Nearest point to {uv_mapping[z_idx-1][0]} found is {curr_segment[curr_mid_idx]} at index {curr_mid_idx}')
        left_curr = curr_segment[curr_mid_idx-delta]
        left_prev = prev_segment[prev_mid_idx-delta]
        right_prev = prev_segment[prev_mid_idx+delta]
        # Global winding logic to fic, if needed
        ll = calculate_euclidian_distance(x1=left_curr[0], y1=left_curr[1], x2=left_prev[0], y2=left_prev[1])
        lr = calculate_euclidian_distance(x1=left_curr[0], y1=left_curr[1], x2=right_prev[0], y2=right_prev[1])
        print(f'll: {ll}, lr: {lr}')
        if ll >= lr:
            print(f'Winding wrong!')
            ordered_segments[z_val][0] = curr_segment[::-1]
            print(f'Fixed winding!')
        populate_uv(uv_mapping=uv_mapping, z_idx=z_idx, delta=delta, segment=curr_segment, mid_idx=curr_mid_idx, z_val=z_val)
        prev_mid_idx = curr_mid_idx
        prev_segment = curr_segment
    
    # for i, _ in enumerate(slice_list):
    #     print(f'i: {i}, [{i}][0]: {uv_mapping[i][0]}, [{i}][-1]: {uv_mapping[i][-1]}, [{i}][1]: {uv_mapping[i][1]} Total points: {len(list(uv_mapping[i].keys()))}, Delta: {delta}\n')
    u_list = []
    v_list = list(uv_mapping.keys())
    for i, _ in enumerate(slice_list):
        u_val = list(uv_mapping[i].keys())
        u_list.extend(u_val)
        key_min = min(u_val)
        key_max = max(u_val)
        print(f'(u,v): min: {key_min,i}, max: {key_max,i}')
    u = min(u_list)
    v = min(v_list)
    U = max(u_list)
    V = max(v_list)
    print(f'The uv-range is from {u,v} to {U,V} with a step size of {delta}')
    print(f'The size of array after normalization is {V-v} x {U-u}')
    normalized_uv_mapping = {}
    normalized_uv_array = []
    for v_key in v_list:
        normalized_uv_mapping[v_key-v]={}
        u_keys = list(uv_mapping[v_key])
        for u_key in u_keys:
            normalized_uv_mapping[v_key-v][u_key-u]=uv_mapping[v_key][u_key]
            normalized_uv_array.append([u_key-u,v_key-v])
    normalized_uv_array_np = np.array(normalized_uv_array)
    print(f'Normalized array shape: {normalized_uv_array_np.shape}')

    print("Started Delauny")
    tri = Delaunay(normalized_uv_array_np)
    print("Finished Delauny")
    filtered_simplices = tri.simplices


    tri_points = normalized_uv_array_np[tri.simplices]
    diff_y = np.max(tri_points[:, :, 1], axis=1) - np.min(tri_points[:, :, 1], axis=1)
    mask = diff_y <= 1.0
    filtered_simplices = tri.simplices[mask]


    # plt.triplot(normalized_uv_array_np[:,0], normalized_uv_array_np[:,1], filtered_simplices)
    # plt.plot(normalized_uv_array_np[:,0], normalized_uv_array_np[:,1], 'o')
    # plt.show()


    print(filtered_simplices) # Indices for vertices
    for i, simp in enumerate(filtered_simplices):
        triangle0 = np.unravel_index(simp[0], normalized_uv_array_np.shape)
        triangle1 = np.unravel_index(simp[1], normalized_uv_array_np.shape)
        triangle2 = np.unravel_index(simp[2], normalized_uv_array_np.shape)
        u0,v0 = triangle0
        u0,v0 = int(u0),int(v0)
        u1,v1 = triangle1
        u1,v1 = int(u1), int(v1)
        u2,v2 = triangle2
        u2,v2 = int(u2), int(v2)
        print(f'{i}: {u0,v0} {u1,v1} {u2,v2}')
        break

    flattened_vertices = normalized_uv_array_np.reshape(-1, 2)
    write_obj_uv(filename='/localdisk0/test-segmentation/delauny_uv.obj', vertices=flattened_vertices, faces=filtered_simplices)
    write_obj_xyz(filename='/localdisk0/test-segmentation/delauny_original.obj', vertices=flattened_vertices, faces=filtered_simplices,
                  uv_map=normalized_uv_mapping)





def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters for file handling')
    file_parser = parser.add_argument_group(title='FILE HANDLING', description="Parameters to handle I/O for files")
    file_parser.add_argument('--input-dir', type=Path, required=True, help='Path to the input image directory')
    file_parser.add_argument('--output-dir', type=Path, required=True, help='Directory to save the output segmentation')
    file_parser.add_argument('--segmentation-file-name', type=Path, default="ordered_segments.pkl")

    ordering_parser = parser.add_argument_group(title='KD-Tree ARGUMENTS', description="Parameters to describe KD-Tree")
    ordering_parser.add_argument('--displacement', type=float, default=40, help="How far to go to find the next point of a wrap?")
    ordering_parser.add_argument('--window', type=int, default=50, help="How many points to consider to find the next point?")
    ordering_parser.add_argument('--search-radius', type=float, default=50, help="Around what radius should I search in KD-Tree?")

    mesh_parser = parser.add_argument_group(title='MESH GENERATION', description="Parameters for generating meshes")
    mesh_parser.add_argument('--mesh-file-name', type=Path, default="mesh.obj", help="File name to save the mesh in. It is OBJ file.")

    uv_parser = parser.add_argument_group(title='UV-Mapping ARGUMENTS')
    uv_parser.add_argument('--uv-delta', type=int, default=1)


    return parser.parse_args()


def main():
    args = parse_arguments()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    segmentation_file_name = Path(args.segmentation_file_name)
    displacement = args.displacement
    window = args.window
    search_radius = args.search_radius
    mesh_file_name = args.mesh_file_name
    uv_delta = args.uv_delta
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")

    SLICE_IMG_DIR = Path(output_dir / 'slice_images')
    SLICE_WINDING_DIR = Path(output_dir / 'slice_winding')

    SLICE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    SLICE_WINDING_DIR.mkdir(parents=True, exist_ok=True)

    MESH_DIR = Path(output_dir / mesh_file_name)

    # ordered_segments = pre_process_files(input_dir, SLICE_IMG_DIR, b=displacement, window=window, search_radius=search_radius)

    output_segmentation_fname = Path(output_dir / f'{segmentation_file_name}')

    # with open(output_segmentation_fname, 'wb') as f:
    #     pickle.dump(ordered_segments, f)
    
    # with open(output_segmentation_fname, 'rb') as f:
    #     ordered_segments = pickle.load(f)
    
    # # save_windings(ordered_segments, SLICE_IMG_DIR, SLICE_WINDING_DIR)

    # # calculate the height and width of a slice to map uv-coord in mesh file
    # for i in SLICE_IMG_DIR.iterdir():
    #     img = iio.imread(i)
    #     h, w, _ = img.shape

    # make_meshes(segmentation_file=output_segmentation_fname, mesh_file=MESH_DIR, h=h, w=w)

    create_uv_mesh(segmentation_file=output_segmentation_fname, delta=uv_delta)





if __name__ == "__main__":
    main()