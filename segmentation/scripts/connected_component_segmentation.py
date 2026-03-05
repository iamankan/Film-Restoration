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
def meshify(list_curr, list_next, f, c, z_value, curr_coord_list, next_coord_list):
    # 1. Write ALL vertices first
    for curr_layer in curr_coord_list:
        f.write(f'v {curr_layer[0]} {curr_layer[1]} {z_value}\n')
    for next_layer in next_coord_list:
        f.write(f'v {next_layer[0]} {next_layer[1]} {z_value + 1}\n')

    top_start = c
    bot_start = c + len(curr_coord_list)

    for i, (v_curr, v_next) in enumerate(zip(list_curr, list_next)):
        t1, t2 = v_curr[0] + top_start, v_curr[1] + top_start
        b1, b2 = v_next[0] + bot_start, v_next[1] + bot_start
        
        # Only write the face if all three indices are different!
        if t1 != b1 and b1 != t2 and t1 != t2:
            f.write(f'f {t1} {b1} {t2}\n')
            
        if t2 != b1 and b1 != b2 and t2 != b2:
            f.write(f'f {t2} {b1} {b2}\n')

    return c + len(curr_coord_list) + len(next_coord_list)

def make_meshes(segmentation_file, mesh_file):
    with open(segmentation_file, 'rb') as f:
        ordered_segments = pickle.load(f)
    
    z_slices = list(ordered_segments['slices'].keys())

    line_number = 1

    fmesh = open(mesh_file, 'w')
    fmesh.close()

    for i, curr_z in enumerate(z_slices[:-1]):
        next_z = z_slices[i+1]
        curr_n = len(ordered_segments['slices'][curr_z][0])
        next_n = len(ordered_segments['slices'][next_z][0])

        curr_idx_pairs = get_clamped_pairs(curr_n)
        next_idx_pairs = get_clamped_pairs(next_n)

        curr_idx_pairs_norm, next_idx_pairs_norm = match_lengths(curr_idx_pairs, next_idx_pairs)

        with open(mesh_file, 'a') as fmesh:
            line_number = meshify(list_curr=curr_idx_pairs_norm, list_next=next_idx_pairs_norm, f=fmesh, 
                                  c=line_number, z_value=float(curr_z),
                                  curr_coord_list=ordered_segments['slices'][curr_z][0], 
                                  next_coord_list=ordered_segments['slices'][next_z][0]
                                  )


    


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
    
    with open(output_segmentation_fname, 'rb') as f:
        ordered_segments = pickle.load(f)
    
    # save_windings(ordered_segments, SLICE_IMG_DIR, SLICE_WINDING_DIR)

    make_meshes(segmentation_file=output_segmentation_fname, mesh_file=MESH_DIR)





if __name__ == "__main__":
    main()