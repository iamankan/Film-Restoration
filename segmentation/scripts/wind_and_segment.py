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


class Point:
    def __init__(self, x:float, y:float, z:float=0):
        self.x:float = x
        self.y:float = y
        self.z:float = z
        self.used_up:bool = False
        self.used_down:bool = False
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_z(self):
        return self.z
    
    def is_used_up(self):
        return self.used_up
    
    def is_used_down(self):
        return self.used_down
    
    def set_used_up(self, up:bool):
        self.used_up = up
        return self
    
    def set_used_down(self, down:bool):
        self.used_down = down
        return self

    def get_point(self):
        return [self.x, self.y, self.z]


class Pointset:
    def __init__(self, pointset):
        self.pointset = pointset

    
    # Delegate standard list operations
    def __len__(self): return len(self.pointset)
    def __getitem__(self, i): return self.pointset[i]
    def __iter__(self): return iter(self.pointset)
    
    def get_pointset(self):
        return self.pointset
    
    def reverse_pointset(self):
        self.pointset = self.pointset[::-1]
        return self.pointset
    
    def get_kdtree(self):
        tree_array = []
        metadata = []
        for pt in self.pointset:
            tree_array.append(pt.get_point())
            metadata.append({'Point': pt})
        kdtree = KDTree(data=np.array(tree_array))
        return kdtree, metadata



class Segment:
    def __init__(self, idx, pointset):
        self.idx = idx
        self.points = pointset
        self.start = self.points[0]
        self.end = self.points[-1]
        self.start_out = None
        self.start_in = None
        self.end_out = None
        self.end_in = None
        self.visited = False
        self.traversed = False

    def reverse_segment(self):
        self.points = self.points.reverse_pointset()
        self.start = self.points[0]
        self.end = self.points[-1]
        return self

    
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
            dir = np.array(pts[-1].get_point()) - np.array(pts[0].get_point())
            a = self.end.get_point()
        elif side == 'start':
            pts = self.points[:window]
            dir = np.array(pts[0].get_point()) - np.array(pts[-1].get_point())
            a = self.start.get_point()
        
        norm = np.linalg.norm(dir)
        if norm == 0: return a
        
        u_hat = dir / norm
        
        return a + (u_hat * b)
    
    def __repr__(self):
        display_dict = {k: v for k, v in self.__dict__.items() if k != 'points'}
        display_dict['points_count'] = len(self.points)
        attrs = ", ".join(f"{k}={v}" for k, v in display_dict.items())
        return f"Segment({attrs})"



def read_coordinates(coord_file, z_value)->Pointset:
    coordinates = []
    with open(coord_file, 'r') as f:
        next(f)
        for line in f:
            coord = tuple(map(float, line.strip().split(',')))
            coordinates.append(Point(x=float(coord[0]), y=float(coord[1]), z=float(z_value)))
    return Pointset(coordinates)


def test_delaunay():
    print("Testing Delanay")
    arr = np.array([[-3,0], [-2,0], [-1,0], [0,0], [1,0], [2,0], [3,0],
                                    [-1,1], [0,1], [1,1],
                                            [0,2], [1,2], [3,2],
                                    [-1,3], [0,3]
                    ])
    tri = Delaunay(arr)
    filtered_simplices = tri.simplices

    tri_points = arr[tri.simplices]
    diff_y = np.max(tri_points[:, :, 1], axis=1) - np.min(tri_points[:, :, 1], axis=1)
    mask = diff_y <= 1.0
    filtered_simplices = tri.simplices[mask]


    plt.triplot(arr[:,0], arr[:,1], filtered_simplices)
    plt.plot(arr[:,0], arr[:,1], 'o')
    plt.show()


def preprocess(input_dir:Path, slice_img_dir:Path, slice_segmentation_dir:Path):
    ordered_segmentations = {}
    for slice_id in natsorted(input_dir.iterdir()):
        z_value = (slice_id.stem).split('_')[1]
        print(f'Pre-processing slice# {z_value}')
        segmentation_instance = [i for i in slice_id.iterdir() if i.is_dir()]
        cluster_instance = [i for i in segmentation_instance[0].iterdir() if i.is_dir()]
        total_segmentation = [i for i in cluster_instance[0].glob('total_colored_segmentation_*.jpg')]
        all_segmentation_coordinate_files = [i for i in natsorted(cluster_instance[0].glob('segmented_component_*.txt'))]
        slice_image = Path(cluster_instance[0] / 'original_image.jpg')

        # Let's copy th images and segmentations
        shutil.copy2(total_segmentation[0], Path(slice_segmentation_dir / f'{z_value}.jpg'))
        shutil.copy2(slice_image, Path(slice_img_dir / f'{z_value}.jpg'))

        ordered_segmentations[z_value] = []
        for coord_file in all_segmentation_coordinate_files:
            ordered_segmentations[z_value].append(read_coordinates(coord_file=coord_file, z_value=z_value))
    
    return ordered_segmentations

def build_kdtree(segments):
    tree_coords = []
    metadata = []
    for segment in segments:
        tree_coords.append(segment.get_start().get_point())
        metadata.append({"obj": segment, "type":'start'})
        tree_coords.append(segment.get_end().get_point())
        metadata.append({"obj": segment, "type":'end'})
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
    # print(f'idx: {curr_seg.get_idx()}, start: {dist_start, idx_start}')
    if dist_start != float('inf'):
        next_metadata = metadata[idx_start]
        # print(f'next-metadata: {next_metadata}, curr-idx: {curr_seg.get_idx()}')
        next_seg = next_metadata['obj']
        next_side = next_metadata['type'] # start/end

        if next_seg.get_idx() != curr_seg.get_idx():
            curr_seg.start_out = (next_seg.get_idx(), next_side)
            if next_side == 'end':
                next_seg.end_in = (curr_seg.get_idx(), 'start')
            elif next_side == 'start':
                next_seg.start_in = (curr_seg.get_idx(), 'start')

            

    dist_end, idx_end = kdtree.query(x=end_extrapolation, k=k, distance_upper_bound=search_radius)
    # print(f'idx: {curr_seg.get_idx()}, end: {dist_end, idx_end}')
    if dist_end != float('inf'):
        next_metadata = metadata[idx_end]
        # print(f'next-metadata: {next_metadata}')
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
            segLen.append(seg.get_num_points())
        seg_idx = np.argmax(np.array(segLen))
        return invalid_segments[seg_idx].points


    

    for seg in valid_segments:
        if (seg.start_out == None and seg.start_in == None) or (seg.end_out == None and seg.end_in == None):
            terminal_segments.append(seg)

    
    if len(terminal_segments) < 2 or len(valid_segments) == 0:
        print(f'Number of terminal: {len(terminal_segments)}, number of valid segments: {len(valid_segments)}')
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
            if curr_reverse_order:
                curr_segment.reverse_segment()
            globally_ordered_slice_segment.extend(curr_segment.points)
        
        if not end:
            if next_reverse_order:
                next_segment.reverse_segment()
            globally_ordered_slice_segment.extend(next_segment.points)
        
        head = next_segment
        start = False
        # print(f'Reverse order: Current: {curr_reverse_order}, Next: {next_reverse_order}')
        curr_reverse_order = next_reverse_order
    
    return Pointset(globally_ordered_slice_segment)


def join_segments(ordered_pointsets: list):
    print(f'Recieved {len(ordered_pointsets)} ordered pointsets')
    if len(ordered_pointsets) == 1: return ordered_pointsets[0]


    segments = [Segment(idx=i, pointset=ordered_pointsets[i]) for i in range(len(ordered_pointsets))]
    kdtree, kdmetadata = build_kdtree(segments=segments)
    for curr_segment in segments:
        search_in_kdtree(kdtree=kdtree, metadata=kdmetadata, curr_seg=curr_segment, b=40, window=50, search_radius=50)
    globally_ordered_slice_segment = traverse_segments(segments=segments)
    return globally_ordered_slice_segment

def process_segments(ordered_segmentations):
    z_list = list(ordered_segmentations.keys())
    for z_idx, z_value in enumerate(z_list):
        ordered_pointsets = ordered_segmentations[z_value] # list
        number_of_segments = len(ordered_pointsets)
        print(f'Z: {z_value}, Number of segments: {number_of_segments}')
        globally_ordered_slice_segment = join_segments(ordered_pointsets=ordered_pointsets)
        ordered_segmentations[z_value] = Pointset(globally_ordered_slice_segment)
    return ordered_segmentations


def draw_winding(ordered_segments, slice_img_dir, slice_winding_dir):
    z_values = list(ordered_segments.keys())
    white = (255, 255, 255)
    for z_value in z_values:
        print(f'Drawing Slice# {z_value}')
        img = iio.imread(Path(slice_img_dir / f'{z_value}.jpg'))
        ordered_pointset = ordered_segments[z_value]
        n = len(ordered_pointset)
        print(f'n: {n}')
        for i, op in enumerate(ordered_pointset):
            color = (255 * i/(n-1), 255 * (1 - (i/(n-1))), 0) # RGB
            x, y, _ = op.get_point()
            cv2.circle(img=img, center=(int(x),int(y)), radius=2, color=color, thickness=1)
            if i < n-1:
                next_x, next_y, _ = ordered_pointset[i+1].get_point()
                cv2.line(img=img, pt1=(int(x),int(y)), pt2=(int(next_x), int(next_y)), color=white, thickness=1)
        iio.imwrite(Path(slice_winding_dir / f'{z_value}.jpg'), image=img)

def euclidean_distance(p1:Point, p2:Point):
    p1 = p1.get_point()
    p2 = p2.get_point()
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)


def prepare_for_meshify(ordered_segmentations, delta=10):
    z_values = list(ordered_segmentations.keys())
    vu_map = {int(i):{} for i,_ in enumerate(z_values)}
    for z_idx, z_value in enumerate(z_values):
        curr_z_idx = z_idx
        curr_z_value = z_value
        curr_pointset = ordered_segmentations[curr_z_value]
        curr_n = len(curr_pointset)
        print(f'CURR N: {curr_n}')

        # Make the mapping for vu-map
        print(f'Starting Make the mapping for vu-map')
        if z_idx == 0:
            print(f'UV right')
            curr_mid_idx = curr_n//2
            u = 0
            vu_map[z_idx][u] = (curr_z_value, curr_mid_idx, curr_pointset[curr_mid_idx].get_point())
            i = curr_mid_idx + delta
            u += 1
            while i < curr_n - 1:
                vu_map[z_idx][u] = (curr_z_value, i, curr_pointset[i].get_point())
                i += delta
                u += 1
            
            print(f'UV left')
            u=0
            j = curr_mid_idx - delta
            u -= 1
            while j >= 0:
                vu_map[z_idx][u] = (curr_z_value, j, curr_pointset[j].get_point())
                j -= delta
                u -= 1
            continue


        print(f'Starting Make the mapping for vu-map for slices after first')


        prev_z_idx = z_idx - 1
        prev_z_value = z_values[prev_z_idx]
        prev_u_list = sorted(list(vu_map[prev_z_idx].keys()))
        prev_min_u = min(prev_u_list)
        prev_max_u = max(prev_u_list)
        prev_left_list = prev_u_list[prev_min_u:0]
        prev_right_list = prev_u_list[1:]
        prev_pointset = ordered_segmentations[prev_z_value]
        prev_n = len(prev_pointset)
        _, prev_mid_idx, _ = vu_map[prev_z_idx][0]
        prev_mid_coord = ordered_segmentations[prev_z_value][int(prev_mid_idx)]
        curr_kd_tree, curr_kd_metadata = curr_pointset.get_kdtree()
        _, curr_mid_idx = curr_kd_tree.query(x=prev_mid_coord.get_point(), k=1) # find the nearest point to the previous index
        u = 0
        vu_map[curr_z_idx][u] = (int(curr_z_value), int(curr_mid_idx), curr_pointset[curr_mid_idx].get_point())

        # Now check left and right for winding direction
        curr_left_idx = curr_mid_idx - delta
        curr_right_idx = curr_mid_idx + delta

        prev_left_idx = prev_mid_idx - delta
        prev_right_idx = prev_mid_idx + delta

        cl_pl = euclidean_distance(curr_pointset[int(curr_left_idx)], prev_pointset[int(prev_left_idx)])
        cl_pr = euclidean_distance(curr_pointset[int(curr_left_idx)], prev_pointset[int(prev_right_idx)])

        cr_pl = euclidean_distance(curr_pointset[int(curr_right_idx)], prev_pointset[int(prev_left_idx)])
        cr_pr = euclidean_distance(curr_pointset[int(curr_right_idx)], prev_pointset[int(prev_right_idx)])

        if cl_pl > cl_pr or cr_pr > cr_pl:
            print(f'Winding is wrong..Fixing it...')
            curr_pointset.reverse_pointset() # Reversing the pointset
            ordered_segmentations[z_value]=curr_pointset
            curr_mid_idx = curr_n - curr_mid_idx - 1 # adjust the mid-index after reversing the segment
        
        # populate the right side of uv map
        for prev_u_right in prev_right_list:
            _, prev_right_ordered_segmentation_idx, _ = vu_map[prev_z_idx][prev_u_right]
            if prev_right_ordered_segmentation_idx > curr_n - 1: # The previous segment finished before the current segment
                break
            prev_right_coord = ordered_segmentations[prev_z_value][int(prev_right_ordered_segmentation_idx)]
            _, curr_right_ordered_Segmentation_idx = curr_kd_tree.query(x=[prev_right_coord.get_point()], k=1)
            vu_map[curr_z_idx][prev_u_right] = (curr_z_value, int(curr_right_ordered_Segmentation_idx), curr_pointset[int(curr_right_ordered_Segmentation_idx)].get_point())
            ordered_segmentations[prev_z_value][int(prev_right_ordered_segmentation_idx)].set_used_down(True)
            ordered_segmentations[curr_z_value][int(curr_right_ordered_Segmentation_idx)].set_used_up(True)
        # populate the left side of the uv map
        for prev_u_left in prev_left_list:
            _, prev_left_ordered_segmentation_idx, _ = vu_map[prev_z_idx][prev_u_left]
            if prev_left_ordered_segmentation_idx <=0: # The current segment finished before the previous segment
                break
            prev_left_coord = ordered_segmentations[prev_z_value][int(prev_left_ordered_segmentation_idx)]
            _, curr_left_ordered_Segmentation_idx = curr_kd_tree.query(x=[prev_left_coord.get_point()], k=1)
            vu_map[curr_z_idx][prev_u_left] = (curr_z_value, int(curr_left_ordered_Segmentation_idx), curr_pointset[int(curr_left_ordered_Segmentation_idx)].get_point())
            ordered_segmentations[prev_z_value][int(prev_left_ordered_segmentation_idx)].set_used_down(True)
            ordered_segmentations[curr_z_value][int(curr_left_ordered_Segmentation_idx)].set_used_up(True)
        
    # clean up the ordered segmentations
    for z_value in z_values:
        print(f'Cleaning slice# {z_value}')
        ordered_pointset = ordered_segmentations[z_value].get_pointset()
        print(f'Type of ordered_pointset: {type(ordered_pointset)}')
        unused_point_indices = []
        for i, pt in enumerate(ordered_pointset):
            if pt.used_down == False and pt.used_up == False:
                unused_point_indices.append(i)
        ordered_pointset_cleaned = [p for i, p in enumerate(ordered_pointset) if i not in unused_point_indices]
        ordered_segmentations[z_value] = ordered_pointset_cleaned
        print(f'Number of points after cleaning: {len(ordered_segmentations[z_value])}')
    
    return ordered_segmentations, vu_map

def perform_delaunay(grid):
    print("Performing Delanay")
    tri = Delaunay(grid)
    filtered_simplices = tri.simplices

    tri_points = grid[tri.simplices]
    diff_y = np.max(tri_points[:, :, 1], axis=1) - np.min(tri_points[:, :, 1], axis=1)
    mask = diff_y <= 1.0
    filtered_simplices = tri.simplices[mask]
    # plt.triplot(grid[:,0], grid[:,1], filtered_simplices)
    # plt.plot(grid[:,0], grid[:,1], 'o')
    # plt.show()
    return filtered_simplices

def meshify(ordered_segmentations_cleaned, vu_map, mesh_file):
    print(f'Meshifying using cleaned segments and uv-maps')
    v_list = list(vu_map.keys())
    grid = []
    for v in v_list:
        u_list = list(vu_map[v].keys())
        for u in u_list:
            grid.append([u,v])
    grid = np.array(grid)
    filtered_simplices = perform_delaunay(grid=grid)
    print(f'filtered_simplices: {filtered_simplices}')
    flattened_vertices = grid.reshape(-1, 2)
    with open(mesh_file, 'w') as fmesh:
        for u,v in flattened_vertices:
            _,_, vertex = vu_map[v][u]
            fmesh.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
        for face in filtered_simplices:
            fmesh.write(f'f {face[0]} {face[1]} {face[2]}\n')







def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters for file handling')
    file_parser = parser.add_argument_group(title='FILE HANDLING', description="Parameters to handle I/O for files")
    file_parser.add_argument('--input-dir', type=Path, required=True, help='Path to the input image directory')
    file_parser.add_argument('--output-dir', type=Path, required=True, help='Directory to save the output segmentation')
    file_parser.add_argument('--segmentation-file-name', type=Path, default="ordered_segments.pkl")


    mesh_parser = parser.add_argument_group(title='MESH GENERATION', description="Parameters for generating meshes")
    mesh_parser.add_argument('--mesh-file-name', type=Path, default="mesh.obj", help="File name to save the mesh in. It is OBJ file.")
    mesh_parser.add_argument('--mesh-output-dir', type=Path, help="Output directory for mesh. If not provided, then outuput directory will be used.")
    mesh_parser.add_argument('--delta', type=int, default=10, help="Set delta for sample during meshing")


    return parser.parse_args()


def main():
    args = parse_arguments()
    input_dir = args.input_dir
    output_dir = args.output_dir
    segmentation_file_name = args.segmentation_file_name
    mesh_file_name = args.mesh_file_name
    mesh_output_dir = args.mesh_output_dir
    mesh_output_dir = mesh_output_dir if mesh_output_dir else output_dir
    delta = args.delta

    SLICE_IMG_DIR = Path(output_dir / 'slice_images')
    SLICE_IMG_DIR.mkdir(parents=True, exist_ok=True)

    SLICE_WINDING_DIR = Path(output_dir / 'slice_winding_images')
    SLICE_WINDING_DIR.mkdir(parents=True, exist_ok=True)

    SLICE_SEGMENTATION_DIR = Path(output_dir / 'slice_segmentation_images')
    SLICE_SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)

    ordered_segmentations = preprocess(input_dir=input_dir, slice_img_dir=SLICE_IMG_DIR, slice_segmentation_dir=SLICE_SEGMENTATION_DIR)

    with open(output_dir / segmentation_file_name, 'wb') as fseg:
        pickle.dump(ordered_segmentations, fseg)

    # with open(output_dir / segmentation_file_name, 'rb') as fseg:
    #     ordered_segmentations = pickle.load(fseg)
    
    for i in ordered_segmentations.keys():
        print(type(ordered_segmentations[i])) # list
        print(type(ordered_segmentations[i][0])) # pointset
        print(type(ordered_segmentations[i][0][0])) # point
        break

    ordered_segmentations = process_segments(ordered_segmentations=ordered_segmentations)

    with open(output_dir / segmentation_file_name, 'wb') as fseg:
        pickle.dump(ordered_segmentations, fseg)
    
    draw_winding(ordered_segments=ordered_segmentations, slice_winding_dir=SLICE_WINDING_DIR, slice_img_dir=SLICE_IMG_DIR)

    ordered_segmentations_cleaned, vu_map = prepare_for_meshify(ordered_segmentations=ordered_segmentations, delta=delta)
    
    with open(output_dir / f'ordered_segments_cleaned.pkl', 'wb') as fseg:
        pickle.dump(ordered_segmentations_cleaned, fseg)
    
    SLICE_GLOBAL_WINDING_CLEANED = Path(output_dir / 'slice_global_winding_cleaned')
    SLICE_GLOBAL_WINDING_CLEANED.mkdir(parents=True, exist_ok=True)
    draw_winding(ordered_segments=ordered_segmentations_cleaned, slice_winding_dir=SLICE_GLOBAL_WINDING_CLEANED, slice_img_dir=SLICE_IMG_DIR)

    meshify(ordered_segmentations_cleaned=ordered_segmentations_cleaned, vu_map=vu_map, mesh_file=Path(mesh_output_dir / mesh_file_name))



    




if __name__ == "__main__":
    main()