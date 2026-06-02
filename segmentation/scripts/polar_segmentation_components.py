from pathlib import Path
import imageio.v3 as iio
import argparse
from natsort import natsorted
import shutil
import numpy as np
import cv2
from scipy.spatial import KDTree
import json
import pickle
from scipy.spatial import Delaunay
from tqdm import tqdm
from typing import TypedDict
from matplotlib import pyplot as plt
import networkx as nx
import trimesh
import uuid
import pymeshlab

class Point:
    _registry = {}
    def __init__(self, x:float, y:float, z:int):
        self._x = x
        self._y = y
        self._z = z
        self._idx = str(uuid.uuid4())

        Point._registry[self._idx] = self
    
    @classmethod
    def get_by_idx(cls, idx: str):
        """Look up a point using Point.get_by_idx(idx_string)"""
        return cls._registry.get(idx)
    
    def get_idx(self)->str:
        return self._idx

    def get_point(self) -> np.ndarray:
        return np.array([self._x, self._y, self._z])
    
    def get_polar(self) -> list:
        r = np.hypot(self._x, self._y)
        t = np.arctan2(self._y, self._x)
        return r, t
    
    def __add__(self, other):
        if other == 0:
            return self
        if isinstance(other, Point):
            return Point(self._x + other._x, 
                        self._y + other._y, 
                        self._z + other._z)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)
    
    def __truediv__(self, scalar):
        """Allows Point / number"""
        if isinstance(scalar, (int, float)):
            return Point(self._x / scalar, 
                         self._y / scalar, 
                         self._z / scalar)
        return NotImplemented
    
    def __sub__(self, other):
        """Allows Point - Point (Calculates a vector)"""
        if isinstance(other, Point):
            return Point(self._x - other._x, 
                         self._y - other._y, 
                         self._z - other._z)
        return NotImplemented

    def __mul__(self, scalar):
        """Allows Point * scalar (Scaling a vector)"""
        if isinstance(scalar, (int, float)):
            return Point(self._x * scalar, 
                         self._y * scalar, 
                         self._z * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __repr__(self):
        return f"Point({self._x}, {self._y}, {self._z})"


class Segment:
    def __init__(self, all_points:list[Point], idx: int, z_value: int):
        self._all_points = all_points
        self._idx = idx
        self._z_value = z_value
        self._winding = None
    
    def get_winding(self)->str:
        return self._winding
    
    def set_winding(self, winding:str)->None:
        print(f'Setting winding to {winding}')
        self._winding = winding
    
    def get_seg_idx(self)->int:
        return self._idx
    
    def get_z_value(self)->int:
        return self._z_value
    
    def get_point_at(self, idx:int)->Point:
        _p = self._all_points[idx]
        return _p.get_point()
    
    def get_len(self)->int:
        return len(self._all_points)
    
    def get_segment(self)->list[Point]:
        return self._all_points
    
    def reverse(self)->None:
        print(f'Reversing Segment!')
        self._all_points = self._all_points[::-1]

    def get_start_point(self)->Point:
        return self._all_points[0]
    
    def get_end_point(self)->Point:
        return self._all_points[-1]
    
    def get_centroid(self)->Point:
        _centroid = [_points.get_point() for _points in self._all_points]
        mean_coords = np.mean(np.array(_centroid), axis=0)
        return Point(x=mean_coords[0], y=mean_coords[1], z=int(mean_coords[2]))


class Slice:
    def __init__(self, z_value:int, idx:int, all_segments:list[Segment]):
        self._z_value = z_value
        self._all_segments = all_segments
        self._idx = idx
        self._kd_metadata = []
    
    def _build_kd_tree(self):
        _segmentation_points = []
        for _segment in self._all_segments:
            for _point in _segment.get_segment():
                _point_np = _point.get_point()
                self._kd_metadata.append(
                    {
                        'segmentation_index': _segment.get_seg_idx(),
                        'point': {
                            'index': _point.get_idx(),
                            'coordinate': _point_np.tolist()
                        }
                    }
                    )
                _segmentation_points.append(_point_np)
        return KDTree(data=np.array(_segmentation_points))
        
    def get_kd_tree(self)->tuple[KDTree,list]:
        return self._build_kd_tree(), self._kd_metadata


    def get_total_segments(self) -> int:
        return len(self._all_segments)
    
    def get_segment_at(self, idx:int) -> Segment:
        return self._all_segments[idx]
    
    def update_segment_at(self, idx:int, new_segment:Segment):
        self._all_segments[idx] = new_segment
    
    def get_z_value(self)->int:
        return self._z_value
    
    def get_idx(self):
        return self._idx
    
    def get_all_segments(self)->list[Segment]:
        return self._all_segments
    
    def get_centroid(self)->Point:
        _centroid = [_seg.get_centroid().get_point() for _seg in self._all_segments]
        mean_coords = np.mean(np.array(_centroid), axis=0)
        return Point(x=mean_coords[0], y=mean_coords[1], z=int(mean_coords[2]))


class Volume:
    def __init__(self, all_slices: list[Slice]):
        self._all_slices = all_slices
        # Build the KD-tree for the whole volume
        print(f'Building KD-Tree')
        self._vol_kd = {}
        for _slice in self._all_slices:
            _slice_kd_tree, _slice_kd_metadata = _slice.get_kd_tree()
            self._vol_kd[_slice.get_idx()] = {
                'tree': _slice_kd_tree,
                'metadata': _slice_kd_metadata
            }
    
    def get_kd_tree(self)->dict:
        return self._vol_kd
    
    def get_slice_at(self, idx)->Slice:
        return self._all_slices[idx]
    
    def update_slice_at(self, idx, new_slice:Slice):
        self._all_slices[idx] = new_slice
    
    def get_all_slices(self)->list[Slice]:
        return self._all_slices
    
    def get_centroid(self)->Point:
        _centroid = [_slc.get_centroid().get_point() for _slc in self._all_slices]
        mean_coords = np.mean(np.array(_centroid), axis=0)
        return Point(x=mean_coords[0], y=mean_coords[1], z=int(mean_coords[2]))


def make_alignment_tree(volume:Volume, kd_match_radius:float|int=1):
    all_slices = volume.get_all_slices()
    total_slices = len(all_slices)
    all_kd_trees = volume.get_kd_tree()
    curr_slice = all_slices[0]
    curr_slice_index = curr_slice.get_idx()
    curr_kd = all_kd_trees[curr_slice_index]
    slice_matches = {}
    alignment_tree = {}
    segmentation_tree = {}
    point_indices_set = set()
    for next_slice in all_slices[1:]:
        print(f'Processing slice {curr_slice_index}/{total_slices-1} for mesh alignment')
        next_slice_index = next_slice.get_idx()
        next_kd = all_kd_trees[next_slice_index]
        kd_matches = curr_kd['tree'].query_ball_tree(other=next_kd['tree'], r=kd_match_radius)
        slice_matches[curr_slice_index] = kd_matches
        segmentation_tree[curr_slice_index]={}
        alignment_tree[curr_slice_index] = {}
        target_seg_idx = {}
        for _i, _m in enumerate(kd_matches):
            if len(_m) == 0:
                continue
            segment_idx = curr_kd['metadata'][_i]['segmentation_index']
            if segment_idx not in alignment_tree[curr_slice_index].keys():
                alignment_tree[curr_slice_index][segment_idx] = []
            if segment_idx not in target_seg_idx.keys():
                target_seg_idx[segment_idx] = set()

            alignment_tree[curr_slice_index][segment_idx].append(
                    {
                        'from': {
                            'segment_index': segment_idx,
                            'point': curr_kd['metadata'][_i]['point']
                        },
                        'to': {
                            'segment_index': next_kd['metadata'][_m[0]]['segmentation_index'],
                            'point': next_kd['metadata'][_m[0]]['point']
                        }
                    }
                )
            point_indices_set.add(curr_kd['metadata'][_i]['point']['index'])
            point_indices_set.add(next_kd['metadata'][_m[0]]['point']['index'])
            target_seg_idx[segment_idx].add(next_kd['metadata'][_m[0]]['segmentation_index'])
        segmentation_tree[curr_slice_index] = {key: list(val) for key, val in target_seg_idx.items()}
            
        curr_kd = next_kd
        curr_slice = next_slice
        curr_slice_index = next_slice_index

    return alignment_tree, segmentation_tree, list(point_indices_set)

def make_alignment_tree_query(volume: Volume, kd_match_radius: float|int=1):
    all_slices = volume.get_all_slices()
    total_slices = len(all_slices)
    all_kd_trees = volume.get_kd_tree()
    curr_slice = all_slices[0]
    curr_slice_index = curr_slice.get_idx()
    curr_kd = all_kd_trees[curr_slice_index]
    
    alignment_tree = {}
    segmentation_tree = {}
    point_indices_set = set()
    
    for next_slice in all_slices[1:]:
        print(f'Processing slice {curr_slice_index}/{total_slices-1} for mesh alignment')
        next_slice_index = next_slice.get_idx()
        next_kd = all_kd_trees[next_slice_index]
        
        alignment_tree[curr_slice_index] = {}
        segmentation_tree[curr_slice_index] = {}
        target_seg_idx_tracker = {}
        
        # Track which target nodes in the next slice have already been claimed
        claimed_global_indices = set()
        
        # Loop through every single ordered point in the current slice's metadata
        for i, curr_meta in enumerate(curr_kd['metadata']):
            src_coord = curr_meta['point']['coordinate']
            curr_seg_idx = curr_meta['segmentation_index']
            
            # Use the pre-existing KD-tree from your class structure to query 1-to-1
            dist, j = next_kd['tree'].query(src_coord, k=1)
            
            # 1. Discard if it exceeds your spatial search radius limit
            if dist > kd_match_radius:
                continue
                
            # 2. Discard if this target index has already been claimed by a previous point
            if j in claimed_global_indices:
                continue
                
            # Lock it in so no other node can claim it
            claimed_global_indices.add(j)
            
            # Retrieve the matching target metadata from your class structures
            matched_target_meta = next_kd['metadata'][j]
            next_seg_idx = matched_target_meta['segmentation_index']
            
            if curr_seg_idx not in alignment_tree[curr_slice_index]:
                alignment_tree[curr_slice_index][curr_seg_idx] = []
            if curr_seg_idx not in target_seg_idx_tracker:
                target_seg_idx_tracker[curr_seg_idx] = set()
            
            alignment_tree[curr_slice_index][curr_seg_idx].append({
                'from': {
                    'segment_index': curr_seg_idx,
                    'point': curr_meta['point']
                },
                'to': {
                    'segment_index': next_seg_idx,
                    'point': matched_target_meta['point']
                }
            })

            point_indices_set.add(curr_meta['point']['index'])
            point_indices_set.add(matched_target_meta['point']['index'])
            
            target_seg_idx_tracker[curr_seg_idx].add(next_seg_idx)
            
        segmentation_tree[curr_slice_index] = {
            key: list(val) for key, val in target_seg_idx_tracker.items()
        }
            
        curr_kd = next_kd
        curr_slice = next_slice
        curr_slice_index = next_slice_index

    return alignment_tree, segmentation_tree, list(point_indices_set)


def find_segment_components(segmentation_tree):
    G = nx.Graph()
    
    # 1. Build the graph from the segmentation tree
    # Nodes are identified by a tuple: (slice_idx, segment_idx)
    for curr_slice_idx, segments in segmentation_tree.items():
        next_slice_idx = curr_slice_idx + 1 # Alignment is always to next slice
        for src_seg_idx, target_segs in segments.items():
            for dest_seg_idx in target_segs:
                # Add an edge between matching segments in adjacent slices
                G.add_edge((curr_slice_idx, src_seg_idx), (next_slice_idx, dest_seg_idx))
                
    # 2. Extract isolated connected components
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} independent connected components in the volume.")
    return components


def make_mesh_old(alignment_tree, segmentation_tree, point_indices_list):
    uuid_to_obj_idx = {
        idx_str: (i + 1) for i, idx_str in enumerate(point_indices_list)
    }

    global_faces = []
    
    all_slice_indices = list(alignment_tree.keys())
    for curr_slice_index in all_slice_indices:
        all_segmentation_indices = list(alignment_tree[curr_slice_index].keys())
        for curr_segmentation_index in all_segmentation_indices:
            aligned_point_list = alignment_tree[curr_slice_index][curr_segmentation_index]
            pointing_to = segmentation_tree[curr_slice_index][curr_segmentation_index]
            
            for to_seg_idx in pointing_to:
                # Filter down to the matching paired nodes for this specific segment track
                active_links = [
                    link for link in aligned_point_list 
                    if link['to']['segment_index'] == to_seg_idx
                ]
                
                # We need at least 2 consecutive links (4 points) to form a quad-strip
                if len(active_links) < 2:
                    continue

                print(f'Stitching strip manually for Slice: {curr_slice_index}, From Seg: {curr_segmentation_index} -> To Seg: {to_seg_idx}')

                # March down the ladder and stitch adjacent pairs into 2 triangles
                for _i in range(len(active_links) - 1):
                    link_curr = active_links[_i]
                    link_next = active_links[_i + 1]

                    # Get the 4 corners of our quad cell
                    v_from_curr = link_curr['from']['point']['index']  # Top-Left
                    v_to_curr   = link_curr['to']['point']['index']    # Bottom-Left
                    
                    v_from_next = link_next['from']['point']['index']  # Top-Right
                    v_to_next   = link_next['to']['point']['index']    # Bottom-Right

                    # Map to 1-based OBJ indices
                    a = uuid_to_obj_idx[v_from_curr]
                    b = uuid_to_obj_idx[v_to_curr]
                    c = uuid_to_obj_idx[v_from_next]
                    d = uuid_to_obj_idx[v_to_next]

                    # Triangle 1: Top-Left, Bottom-Left, Bottom-Right
                    global_faces.append([a, b, d])
                    
                    # Triangle 2: Top-Left, Bottom-Right, Top-Right
                    global_faces.append([a, d, c])
                    
    return uuid_to_obj_idx, global_faces

def make_connected_mesh(alignment_tree, segmentation_tree, point_indices_list):
    import networkx as nx

    # 1. Map UUID strings to 1-based OBJ vertex indices
    uuid_to_obj_idx = {
        idx_str: (i + 1) for i, idx_str in enumerate(point_indices_list)
    }

    # 2. Build a structural tracking graph from the segmentation tree
    # Nodes are tuples: (slice_index, segment_index)
    G = nx.Graph()
    for curr_slice_index, segments in segmentation_tree.items():
        next_slice_index = curr_slice_index + 1
        for src_seg_idx, target_segs in segments.items():
            for dest_seg_idx in target_segs:
                G.add_edge((curr_slice_index, src_seg_idx), (next_slice_index, dest_seg_idx))

    # 3. Extract the independent connected components and sort them by size (largest first)
    components = sorted(list(nx.connected_components(G)), key=len, reverse=True)
    print(f"Graph analysis: Found {len(components)} independent connected components.")

    # Create a quick-lookup map from a segment node to its component ID index
    node_to_comp_id = {}
    for comp_id, nodes in enumerate(components):
        for node in nodes:
            node_to_comp_id[node] = comp_id

    # 4. Initialize face containers for each independent component
    component_faces = [[] for _ in range(len(components))]
    
    all_slice_indices = list(alignment_tree.keys())
    for curr_slice_index in all_slice_indices:
        all_segmentation_indices = list(alignment_tree[curr_slice_index].keys())
        for curr_segmentation_index in all_segmentation_indices:
            aligned_point_list = alignment_tree[curr_slice_index][curr_segmentation_index]
            pointing_to = segmentation_tree[curr_slice_index][curr_segmentation_index]
            
            # Identify which component the source segment track belongs to
            src_node = (curr_slice_index, curr_segmentation_index)
            comp_id = node_to_comp_id.get(src_node)
            
            if comp_id is None:
                continue # Skip if this node somehow didn't register in the topology graph

            for to_seg_idx in pointing_to:
                # Filter down to the matching paired nodes for this specific segment track
                active_links = [
                    link for link in aligned_point_list 
                    if link['to']['segment_index'] == to_seg_idx
                ]
                
                # We need at least 2 consecutive links (4 points) to form a quad-strip
                if len(active_links) < 2:
                    continue

                # March down the ladder and stitch adjacent pairs into 2 triangles
                for _i in range(len(active_links) - 1):
                    link_curr = active_links[_i]
                    link_next = active_links[_i + 1]

                    # Get the 4 corners of our quad cell
                    v_from_curr = link_curr['from']['point']['index']  # Top-Left
                    v_to_curr   = link_curr['to']['point']['index']    # Bottom-Left
                    
                    v_from_next = link_next['from']['point']['index']  # Top-Right
                    v_to_next   = link_next['to']['point']['index']    # Bottom-Right

                    # Map to 1-based OBJ indices
                    a = uuid_to_obj_idx[v_from_curr]
                    b = uuid_to_obj_idx[v_to_curr]
                    c = uuid_to_obj_idx[v_from_next]
                    d = uuid_to_obj_idx[v_to_next]

                    # Append triangles directly to their specific isolated component container
                    # Triangle 1: Top-Left, Bottom-Left, Bottom-Right
                    component_faces[comp_id].append([a, b, d])
                    
                    # Triangle 2: Top-Left, Bottom-Right, Top-Right
                    component_faces[comp_id].append([a, d, c])
                    
    return uuid_to_obj_idx, component_faces



def save_winding(volume: Volume, slice_winding_dir: Path, slice_img_dir: Path):
    all_slices = volume.get_all_slices()
    vol_cg = volume.get_centroid().get_point()
    for slc in all_slices:
        dest_fname = f'{slice_winding_dir}/{slc.get_z_value():04d}.jpg'
        original_fname = f'{slice_img_dir}/{slc.get_z_value():04d}.jpg'
        img = iio.imread(original_fname)
        for seg in slc.get_all_segments():
            seg_pts = seg.get_segment()
            n = seg.get_len()
            print(f'Segment winding: {seg.get_winding()}')
            for i, pt in enumerate(seg_pts):
                my_pt = pt.get_point()
                x,y = int(my_pt[0]), int(my_pt[1])
                color = (0, 255 * (1 - (i/(n-1))), 255 * i/(n-1))
                cv2.circle(img=img, center=(x,y), radius=2, color=color, thickness=2)
        slc_cg = slc.get_centroid().get_point()
        cv2.circle(img=img, center=(int(vol_cg[0]), int(vol_cg[1])), radius=2, color=(255,255,0), thickness=2)
        cv2.circle(img=img, center=(int(slc_cg[0]), int(slc_cg[1])), radius=2, color=(0,255,255), thickness=2)
        print(f'Saving {dest_fname}')
        cv2.imwrite(dest_fname, img=img)

def determine_winding(volume: Volume) -> Volume:
    all_slices = volume.get_all_slices()
    vol_cg = volume.get_centroid()
    
    for slc in all_slices:
        print(f'Slice#{slc.get_z_value()}')
        segs = slc.get_all_segments()
        
        for seg in segs:
            total_del_t = 0
            points = seg.get_segment() 
            
            for i in range(len(points) - 1):
                p_curr = points[i] - vol_cg
                p_next = points[i+1] - vol_cg
                
                _, t_curr = p_curr.get_polar()
                _, t_next = p_next.get_polar()
                
                dt = t_next - t_curr
                
                if dt > np.pi:
                    dt -= 2 * np.pi
                elif dt < -np.pi:
                    dt += 2 * np.pi
                
                total_del_t += dt
                
            seg.set_winding("+" if total_del_t > 0 else "-")
            print(f'Seg#{seg.get_seg_idx()}, Winding: {seg.get_winding()} (Accumulated: {total_del_t:.4f})')
            
    return volume

def fix_winding(volume:Volume, target_winding:str)->Volume:
    all_slices = volume.get_all_slices()
    for slc in all_slices:
        print(f'Slice#{slc.get_z_value()}')
        segs = slc.get_all_segments()
        for seg in segs:
            print(f'Before: Segment#{seg.get_seg_idx()}, Winding: {seg.get_winding()}, [{seg.get_start_point()}, {seg.get_end_point()}]')
            if seg.get_winding() != target_winding:
                seg.reverse()
                seg.set_winding(winding=target_winding)
                print(f'After: Segment#{seg.get_seg_idx()}, Winding: {seg.get_winding()}, [{seg.get_start_point()}, {seg.get_end_point()}]')
    return volume

# def get_sampled_coordinates(coord: list, sampling_rate: int=1):
#     sampled_coordinates = coord[::sampling_rate]
#     if sampled_coordinates[-1] != coord[-1]:
#         sampled_coordinates.append(coord[-1])
#     return sampled_coordinates


# def read_segmentation_file(coord_file:Path, seg_idx:int, z_value: int, sampling_rate: int):
#     coordinates = []
#     with open(coord_file, 'r') as f:
#         next(f)
#         for line in f:
#             xy = list(map(float, line.strip().split(',')))
#             coordinates.append(Point(x=xy[0], y=xy[1], z=z_value))
#     return Segment(all_points=get_sampled_coordinates(coord=coordinates, sampling_rate=sampling_rate), idx=seg_idx, z_value=z_value)

def get_sampled_coordinates(coord: list, sampling_rate: int=1):
    sampled_coordinates = coord[::sampling_rate]
    # Simple check for lists of primitive types / tuples
    if sampled_coordinates[-1] != coord[-1]:
        sampled_coordinates.append(coord[-1])
    return sampled_coordinates


def read_segmentation_file(coord_file: Path, seg_idx: int, z_value: int, sampling_rate: int):
    raw_coords = []
    with open(coord_file, 'r') as f:
        next(f) # Skip header
        for line in f:
            xy = list(map(float, line.strip().split(',')))
            raw_coords.append((xy[0], xy[1])) # Store as light tuple
            
    # 1. Downsample the raw data BEFORE turning them into heavy class objects
    sampled_raw = get_sampled_coordinates(coord=raw_coords, sampling_rate=sampling_rate)
    
    # 2. Only instantiate Point objects for the final, optimized subset
    final_points = [Point(x=pt[0], y=pt[1], z=z_value) for pt in sampled_raw]
    
    return Segment(all_points=final_points, idx=seg_idx, z_value=z_value)

def preprocess(input_dir:Path, slice_img_dir:Path, slice_segmentation_dir:Path, sampling_rate: int) -> Volume:
    all_slices = []
    for slice_idx, slice_id in enumerate(natsorted(input_dir.iterdir())):
        all_segments = []
        z_name = (slice_id.stem).split('_')[1]
        z_value = int(z_name)
        print(f'Pre-processing slice# {z_name}')
        segmentation_instance = [i for i in slice_id.iterdir() if i.is_dir()]
        cluster_instance = [i for i in segmentation_instance[0].iterdir() if i.is_dir()]
        total_segmentation = [i for i in cluster_instance[0].glob('total_colored_segmentation_*.jpg')]
        all_segmentation_coordinate_files = [i for i in natsorted(cluster_instance[0].glob('segmented_component_*.txt'))]
        slice_image = Path(cluster_instance[0] / 'original_image.jpg')

        # Let's copy the images and segmentations
        shutil.copy2(total_segmentation[0], Path(slice_segmentation_dir / f'{z_name}.jpg'))
        shutil.copy2(slice_image, Path(slice_img_dir / f'{z_name}.jpg'))

        for seg_idx, coord_file in enumerate(all_segmentation_coordinate_files):
            all_segments.append(read_segmentation_file(coord_file=coord_file, seg_idx=seg_idx, z_value=z_value, sampling_rate=sampling_rate))
        single_slice = Slice(z_value=z_value, all_segments=all_segments, idx=slice_idx)
        all_slices.append(single_slice)
    
    full_volume = Volume(all_slices=all_slices)

    return full_volume


def repair_and_close_holes(input_mesh_path, output_mesh_path=None, max_hole_size=100):
    ms = pymeshlab.MeshSet()

    str_input_path = str(input_mesh_path)
    
    ms.load_new_mesh(str_input_path)
    
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    
    ms.meshing_close_holes(maxholesize=max_hole_size)
    
    if output_mesh_path is None:
        str_output_path = str_input_path
    else:
        str_output_path = str(output_mesh_path)
        
    ms.save_current_mesh(str(output_mesh_path), 
                            save_vertex_color=False,
                            save_vertex_coord=True,      
                            save_vertex_normal=False,
                            save_face_color=False,
                            save_wedge_texcoord=False,
                            save_wedge_normal=False,
                            save_polygonal=False)
    
    print(f"Cleaned mesh saved to: {output_mesh_path}")

    # 2. Define a temporary file path for the streaming process
    temp_path = output_mesh_path.with_suffix('.tmp')

    # 3. Stream line-by-line to fix the 1.0000000 precision overhead
    with output_mesh_path.open("r") as infile, temp_path.open("w") as outfile:
        for line in infile:
            if line.startswith("v "):  # Match only vertex coordinate lines
                parts = line.split()
                # Truncate each coordinate to 3 decimal places
                # Turns "v 1.2345678 2.0000000" into "v 1.235 2.000"
                outfile.write(f"v {float(parts[1]):.3f} {float(parts[2]):.3f} {float(parts[3]):.3f}\n")
            else:
                # Pass faces (f) and headers through unmodified
                outfile.write(line)

    # 4. Atomic replace to overwrite the bloated file with the optimized one
    temp_path.replace(output_mesh_path)
    print(f"Mesh processed and saved to: {str_output_path}")

def repair_close_and_clean(input_mesh_path, output_mesh_path=None, vertical_displacenment_ratio:float = 0.0, max_hole_size=100):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_mesh_path))
    

    ms.meshing_repair_non_manifold_vertices(vertdispratio=vertical_displacenment_ratio)
    
    ms.meshing_repair_non_manifold_edges()
    
    ms.meshing_close_holes(maxholesize=max_hole_size)
    
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.01))
    
    ms.meshing_remove_null_faces()
    
    ms.meshing_remove_unreferenced_vertices()
    
    if output_mesh_path is None:
        output_mesh_path = input_mesh_path
    
    ms.save_current_mesh(str(output_mesh_path), 
                            save_vertex_color=False,
                            save_vertex_coord=True,      
                            save_vertex_normal=False,
                            save_face_color=False,
                            save_wedge_texcoord=False,
                            save_wedge_normal=False,
                            save_polygonal=False)
    
    print(f"Cleaned mesh saved to: {output_mesh_path}")

    # 2. Define a temporary file path for the streaming process
    temp_path = output_mesh_path.with_suffix('.tmp')

    # 3. Stream line-by-line to fix the 1.0000000 precision overhead
    with output_mesh_path.open("r") as infile, temp_path.open("w") as outfile:
        for line in infile:
            if line.startswith("v "):  # Match only vertex coordinate lines
                parts = line.split()
                # Truncate each coordinate to 3 decimal places
                # Turns "v 1.2345678 2.0000000" into "v 1.235 2.000"
                outfile.write(f"v {float(parts[1]):.3f} {float(parts[2]):.3f} {float(parts[3]):.3f}\n")
            else:
                # Pass faces (f) and headers through unmodified
                outfile.write(line)

    # 4. Atomic replace to overwrite the bloated file with the optimized one
    temp_path.replace(output_mesh_path)


def parse_arguments():
    parser = argparse.ArgumentParser()

    file_group = parser.add_argument_group(title="File I/O handler")
    file_group.add_argument('-i', '--input-dir', help="Input directory that has the segmentation files", required=True, type=Path)
    file_group.add_argument('-o', '--output-dir', help="Output directory that will have all the files", required=True, type=Path)
    
    mesh_group = parser.add_argument_group(title="Mesh parameter handler")
    mesh_group.add_argument('-r','--kd-match-radius', help="Radius for KD-Tree for matching segments", required=False, type=float, default=1.0)
    mesh_group.add_argument('-m','--output-mesh-path', help="Output mesh filename", required=False, type=Path, default='mesh.obj')
    mesh_group.add_argument('--sampling-rate', help="Provide a sampling rate so that it can make a shallow yet meaningful mesh.", type=int, default=1)


    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    kd_match_radius = args.kd_match_radius
    output_mesh_path = args.output_mesh_path
    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    sampling_rate = int(args.sampling_rate)

    print(f'Using sampling rate: {sampling_rate}')


    SLICE_IMG_DIR = Path(output_dir / 'slice_images')
    SLICE_IMG_DIR.mkdir(parents=True, exist_ok=True)

    SLICE_WINDING_DIR = Path(output_dir / 'slice_winding_images')
    SLICE_WINDING_DIR.mkdir(parents=True, exist_ok=True)

    SLICE_SEGMENTATION_DIR = Path(output_dir / 'slice_segmentation_images')
    SLICE_SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)

    full_volume = preprocess(input_dir=input_dir, slice_img_dir=SLICE_IMG_DIR, slice_segmentation_dir=SLICE_SEGMENTATION_DIR,
                             sampling_rate=sampling_rate)

    det_vol = determine_winding(volume=full_volume)

    fixed_vol = fix_winding(volume=det_vol, target_winding='+')

    save_winding(volume=fixed_vol, slice_winding_dir=SLICE_WINDING_DIR, slice_img_dir=SLICE_IMG_DIR)

    alignment_tree, segmentation_tree, point_indices_list = make_alignment_tree_query(volume=fixed_vol, kd_match_radius=kd_match_radius)

    # # 1. Grab the global center of gravity vector for the volume
    # vol_centroid_np = fixed_vol.get_centroid().get_point()

    # # 2. Pass it directly into the update mesh generator
    # uuid_to_obj_idx, global_faces = make_mesh(
    #     alignment_tree=alignment_tree, 
    #     segmentation_tree=segmentation_tree, 
    #     point_indices_list=point_indices_list,
    #     volume_centroid=vol_centroid_np
    # )

    
    # uuid_to_obj_idx, global_faces = make_mesh_old(alignment_tree=alignment_tree, segmentation_tree=segmentation_tree, point_indices_list=point_indices_list)

    # sorted_uuids = sorted(uuid_to_obj_idx, key=uuid_to_obj_idx.get)

    # with open(f'{output_mesh_path}', 'w') as fmesh:
    #     for _point_uuid in sorted_uuids:
    #         _point = Point.get_by_idx(idx=_point_uuid)
    #         _point_np = _point.get_point().tolist()
    #         fmesh.write(f'v {_point_np[0]} {_point_np[1]} {_point_np[2]}\n')
        
    #     for _face in global_faces:
    #         fmesh.write(f'f {_face[0]} {_face[1]} {_face[2]}\n')
        
    #     print(f'Finished writing mesh.obj file!')
    # Point._registry.clear()

    # with output_mesh_path.open('w', newline='\n') as fmesh:
    #     for _point_uuid in sorted_uuids:
    #         _point = Point.get_by_idx(idx=_point_uuid)
    #         _point_np = _point.get_point().tolist()
            
    #         # Clamp precision to 3 decimal places here using :.3f
    #         # This converts [1.0000000, 2.5000000, 3.1415926] directly to "1.000 2.500 3.142"
    #         fmesh.write(f'v {_point_np[0]:.1f} {_point_np[1]:.1f} {_point_np[2]:.1f}\n')
        
    #     for _face in global_faces:
    #         # Keep faces strictly as integers, no normals attached
    #         fmesh.write(f'f {_face[0]} {_face[1]} {_face[2]}\n')
        
    #     print(f'Finished writing mesh.obj file!')
    # Point._registry.clear()

    uuid_to_obj_idx, component_faces = make_connected_mesh(
            alignment_tree=alignment_tree, 
            segmentation_tree=segmentation_tree, 
            point_indices_list=point_indices_list
        )
    
    # 2. Create a fast reverse lookup: OBJ_Idx -> Point Object
    # This allows us to instantly check what slice and segment any face vertex came from
    obj_idx_to_point = {}
    for uuid_str, obj_idx in uuid_to_obj_idx.items():
        obj_idx_to_point[obj_idx] = Point.get_by_idx(uuid_str)
    
    sorted_uuids = sorted(uuid_to_obj_idx, key=uuid_to_obj_idx.get)


    # 3. Process and save each component independently
    for comp_id, faces in enumerate(component_faces):
        if len(faces) < 10:
            continue
            
        # --- Save the OBJ File ---
        cc_obj_name = f"{output_mesh_path.stem}_cc{comp_id}.obj"
        cc_mesh_path = output_mesh_path.parent / cc_obj_name

        with cc_mesh_path.open('w', newline='\n') as fmesh:
            for _point_uuid in sorted_uuids:
                _point = Point.get_by_idx(idx=_point_uuid)
                _point_np = _point.get_point().tolist()
                fmesh.write(f'v {_point_np[0]:.1f} {_point_np[1]:.1f} {_point_np[2]:.1f}\n')
            for _face in faces:
                fmesh.write(f'f {_face[0]} {_face[1]} {_face[2]}\n')
        
        print(f"Finished writing geometry: {cc_obj_name}")

        # --- Build the Component-Specific Subtree ---
        # We look at the triangles in this component, find their original tracking segments,
        # and pull their connectivity straight out of the global segmentation_tree.
        cc_subtree = {}
        
        for face in faces:
            for vertex_idx in face:
                point_obj = obj_idx_to_point.get(vertex_idx)
                if point_obj is None:
                    continue
                
                # Retrieve the historical slice and segment placement for this vertex
                # Note: Assuming your polar_segmentation tracking script attaches these attributes to Point or its metadata links
                # If they aren't explicitly attributes, you can look them up via the alignment tree links.
                # Here, we grab the slice/segment tracks active in this face block:
                curr_slice = point_obj._z  # Or your internal slice index tracking attribute
                
                # Safely pull the segment's path directly out of the global tree
                if curr_slice in segmentation_tree:
                    for src_seg_idx, pointing_to in segmentation_tree[curr_slice].items():
                        # Link this segment topology to our isolated component pickle
                        if curr_slice not in cc_subtree:
                            cc_subtree[curr_slice] = {}
                        cc_subtree[curr_slice][src_seg_idx] = pointing_to

        # --- Save the Component Tree Pickle ---
        cc_pkl_name = f"{output_mesh_path.stem}_cc{comp_id}_tree.pkl"
        cc_pkl_path = output_mesh_path.parent / cc_pkl_name
        
        with open(cc_pkl_path, 'wb') as f_pkl:
            pickle.dump(cc_subtree, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"Finished pickling topology: {cc_pkl_name}")

    # Track how many valid component files we actually write out
    written_components_count = 0

    # Iterate through each component's faces independently
    for comp_id, faces in enumerate(component_faces):
        # If a component is empty or just noise (e.g., less than 10 faces), skip it
        if len(faces) < 10:
            continue
            
        # Dynamically construct the component file name: mesh_cc0.obj, mesh_cc1.obj...
        cc_filename = f"{output_mesh_path.stem}_cc{comp_id}{output_mesh_path.suffix}"
        cc_mesh_path = output_mesh_path.parent / cc_filename

        with cc_mesh_path.open('w', newline='\n') as fmesh:
            # 1. Write the vertex block for this file
            for _point_uuid in sorted_uuids:
                _point = Point.get_by_idx(idx=_point_uuid)
                _point_np = _point.get_point().tolist()
                fmesh.write(f'v {_point_np[0]:.1f} {_point_np[1]:.1f} {_point_np[2]:.1f}\n')
            
            # 2. Write only the faces belonging strictly to this isolated component
            for _face in faces:
                fmesh.write(f'f {_face[0]} {_face[1]} {_face[2]}\n')
            
        print(f'Finished writing isolated component: {cc_mesh_path.name} ({len(faces)} faces)')
        written_components_count += 1
        
    Point._registry.clear()
    print(f'Successfully exported {written_components_count} independent component meshes!')

    

if __name__ == "__main__":
    main()