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


class Point:
    def __init__(self, x:float, y:float, z:int):
        self._x = x
        self._y = y
        self._z = z

    def get_point(self) -> np.array:
        return np.array([self._x, self._y, self._z])
    
    def get_polar(self):
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
        self._all_points = np.array(all_points)
        self._idx = idx
        self._z_value = z_value
        self._winding = None
    
    def get_winding(self):
        return self._winding
    
    def set_winding(self, winding:str):
        print(f'Setting winding to {winding}')
        self._winding = winding
    
    def get_seg_idx(self)->int:
        return self._idx
    
    def get_z_value(self)->int:
        return self._z_value
    
    def get_point_at(self, idx)->Point:
        return self._all_points[idx].get_point()
    
    def get_len(self)->int:
        return self._all_points.shape[0]
    
    def get_segment(self)->list[Point]:
        return self._all_points.astype(list)
    
    def reverse(self)->None:
        print(f'Reversing Segment!')
        self._all_points = np.flip(self._all_points, axis=0)

    def get_start_point(self)->Point:
        return self._all_points[0]
    
    def get_end_point(self)->Point:
        return self._all_points[-1]
    
    def get_centroid(self)->Point:
        return np.mean(self._all_points, axis=0)


class Slice:
    def __init__(self, z_value:int, idx:int, all_segments:list[Segment]):
        self._z_value = z_value
        self._all_segments = all_segments
        self._kd_tree = None
        self._kd_metadata = []
        self._idx = idx

    def _build_kd_tree(self):
        # Building the data and the metadata
        _kd_data = []
        self._kd_metadata = []
        for _segment in self._all_segments:
            _seg_idx = _segment.get_seg_idx()
            for _point_idx, _point in enumerate(_segment.get_segment()):
                _point_np = _point.get_point()
                _kd_data.append(_point_np)
                self._kd_metadata.append({
                    'slice':self, 
                    'seg_idx':_seg_idx, 
                    'point_idx':_point_idx, 
                    'point':_point
                })
        self._kd_tree = KDTree(data=_kd_data)
    
    def get_kd_tree(self, force_build:bool=False):
        if not self._kd_tree or force_build:
            self._build_kd_tree()
        return self._kd_tree, self._kd_metadata

    def get_total_segments(self) -> int:
        return len(self._all_segments)
    
    def get_segment_at(self, idx) -> Segment:
        return self._all_segments[idx]
    
    def update_segment_at(self, idx, new_segment):
        self._all_segments[idx] = new_segment
    
    def get_z_value(self)->int:
        return self._z_value
    
    def get_idx(self):
        return self._idx
    
    def get_all_segments(self)->list[Segment]:
        return self._all_segments
    
    def get_centroid(self)->np.array:
        _centroid = [seg.get_centroid() for seg in self._all_segments]
        return np.mean(np.array(_centroid), axis=0)



class Volume:
    def __init__(self, all_slices: list[Slice]):
        self._all_slices = all_slices
    
    def get_slice_at(self, idx)->Slice:
        return self._all_slices[idx]
    
    def update_slice_at(self, idx, new_slice:Slice):
        self._all_slices[idx] = new_slice
    
    def get_all_slices(self)->list[Slice]:
        return self._all_slices
    
    def get_centroid(self)->np.array:
        _centroid = [slc.get_centroid() for slc in self._all_slices]
        return np.mean(np.array(_centroid), axis=0)

def align_segments(volume: Volume):
    _mesh_map = {}
    all_slices = volume.get_all_slices()
    slc_idx_list = [slc.get_idx() for slc in all_slices]

    curr_slc_idx = slc_idx_list[0]

    for next_slc_idx in slc_idx_list[1:]:
        _mesh_map[curr_slc_idx] = {}
        print(
            f"Current slice idx: {curr_slc_idx}, next slice idx: {next_slc_idx}"
        )

        curr_slc = volume.get_slice_at(curr_slc_idx)
        next_slc = volume.get_slice_at(next_slc_idx)
        
        curr_slc_kd_tree, curr_slc_metadata = curr_slc.get_kd_tree()
        next_slc_kd_tree, next_slc_metadata = next_slc.get_kd_tree()

        for curr_seg in curr_slc.get_all_segments():
            _mesh_map[curr_slc_idx][curr_seg.get_seg_idx()] = []
        results = curr_slc_kd_tree.query_ball_tree(next_slc_kd_tree, r=1)

        for curr_tree_idx, next_tree_indices in enumerate(results):
            if not next_tree_indices:
                continue
            curr_meta = curr_slc_metadata[curr_tree_idx]
            curr_seg_idx = curr_meta["seg_idx"]
            curr_point_obj = curr_meta["point"]
            left_side = [curr_slc_idx, curr_point_obj, curr_seg_idx]

            for next_tree_idx in next_tree_indices:
                next_meta = next_slc_metadata[next_tree_idx]

                right_side = [
                    next_slc_idx,
                    next_meta["point"],
                    next_meta["seg_idx"]
                ]
                point_pair = [left_side, right_side]
                _mesh_map[curr_slc_idx][curr_seg_idx].append(point_pair)

        curr_slc_idx = next_slc_idx
    return _mesh_map



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


def export_mesh_map_to_obj(mesh_map, output_filepath):
    global_vertex_counter = 1  # OBJ files use 1-based indexing for faces

    with open(output_filepath, "w") as obj_file:
        # Write a simple header
        obj_file.write("# Generated Alignment Mesh\n")
        obj_file.write(f"# Target: {output_filepath}\n\n")

        # Wrap the slice loop in tqdm to monitor the progress across all layers
        for curr_slc_idx, segments in tqdm(
            mesh_map.items(), desc="Exporting OBJ Mesh", unit="slice"
        ):

            for seg_idx, point_pairs in segments.items():
                if not point_pairs:
                    continue

                obj_file.write(
                    f"g slice_{curr_slc_idx}_segment_{seg_idx}\n"
                )

                points_2d = []
                mapping_3d = []
                seen_points = {}

                # 1. Build a quick lookup map to get a point's index within its segment.
                first_pair = point_pairs[0]
                left_sample_point = first_pair[0][1]
                right_sample_point = first_pair[1][1]

                l_slice_obj = point_pairs[0][0][1]

                left_points_seen = {}
                right_points_seen = {}

                for left_side, right_side in point_pairs:
                    l_slc, l_point_obj, l_seg_idx = left_side
                    r_slc, r_point_obj, r_seg_idx = right_side

                    l_id = id(l_point_obj)
                    if l_id not in left_points_seen:
                        left_points_seen[l_id] = len(left_points_seen)

                    r_id = id(r_point_obj)
                    if r_id not in right_points_seen:
                        right_points_seen[r_id] = len(right_points_seen)

                    l_idx = left_points_seen[l_id]
                    r_idx = right_points_seen[r_id]

                    # --- Left side (Slice A, Y=0) ---
                    l_key = (l_slc, l_idx)
                    if l_key not in seen_points:
                        seen_points[l_key] = len(points_2d)
                        points_2d.append([l_idx, 0])
                        mapping_3d.append(l_point_obj.get_point())

                    # --- Right side (Slice B, Y=1) ---
                    r_key = (r_slc, r_idx)
                    if r_key not in seen_points:
                        seen_points[r_key] = len(points_2d)
                        points_2d.append([r_idx, 1])
                        mapping_3d.append(r_point_obj.get_point())

                if len(points_2d) < 3:
                    continue

                points_2d_np = np.array(points_2d)

                # 2. Write the 3D Vertices to the file
                for coords in mapping_3d:
                    obj_file.write(
                        f"v {coords[0]:.4f} {coords[1]:.4f} {coords[2]:.4f}\n"
                    )

                # 3. Compute Delaunay Triangulation in 2D
                tri = Delaunay(points_2d_np)

                # 4. Write the Faces referencing global indices
                for simplex in tri.simplices:
                    global_v1 = simplex[0] + global_vertex_counter
                    global_v2 = simplex[1] + global_vertex_counter
                    global_v3 = simplex[2] + global_vertex_counter

                    obj_file.write(f"f {global_v1} {global_v2} {global_v3}\n")

                # 5. Update global counter
                global_vertex_counter += len(mapping_3d)
                obj_file.write("\n")

    print(
        f"\nSuccessfully saved consolidated mesh to {output_filepath} with {global_vertex_counter - 1} vertices!"
    )


def read_segmentation_file(coord_file:Path, seg_idx:int, z_value: int):
    coordinates = []
    with open(coord_file, 'r') as f:
        next(f)
        for line in f:
            xy = list(map(float, line.strip().split(',')))
            coordinates.append(Point(x=xy[0], y=xy[1], z=z_value))
    return Segment(all_points=coordinates, idx=seg_idx, z_value=z_value)

def preprocess(input_dir:Path, slice_img_dir:Path, slice_segmentation_dir:Path) -> Volume:
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
            all_segments.append(read_segmentation_file(coord_file=coord_file, seg_idx=seg_idx, z_value=z_value))
        single_slice = Slice(z_value=z_value, all_segments=all_segments, idx=slice_idx)
        all_slices.append(single_slice)
    
    full_volume = Volume(all_slices=all_slices)

    return full_volume


def parse_arguments():
    parser = argparse.ArgumentParser()

    file_group = parser.add_argument_group(title="File I/O handler")
    file_group.add_argument('--input-dir', help="Input directory that has the segmentation files", required=True, type=Path)
    file_group.add_argument('--output-dir', help="Output directory that will have all the files", required=True, type=Path)

    args = parser.parse_args()

    return args

def make_json_serializable(m_map):
    serializable_map = {}

    for slc_idx, segments in m_map.items():
        # JSON keys MUST be strings
        str_slc_idx = str(slc_idx)
        serializable_map[str_slc_idx] = {}

        for seg_idx, pairs in segments.items():
            str_seg_idx = str(seg_idx)
            serializable_map[str_slc_idx][str_seg_idx] = []

            for left, right in pairs:
                # Unpack the tuples
                l_slc, l_point_obj, l_seg = left
                r_slc, r_point_obj, r_seg = right

                # Extract raw coordinates from Point objects and convert to lists
                l_coords = l_point_obj.get_point().tolist()
                r_coords = r_point_obj.get_point().tolist()

                # Rebuild as standard JSON lists
                serializable_pair = [
                    [int(l_slc), l_coords, int(l_seg)],
                    [int(r_slc), r_coords, int(r_seg)],
                ]

                serializable_map[str_slc_idx][str_seg_idx].append(
                    serializable_pair
                )

    return serializable_map

def main():
    args = parse_arguments()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)


    SLICE_IMG_DIR = Path(output_dir / 'slice_images')
    SLICE_IMG_DIR.mkdir(parents=True, exist_ok=True)

    SLICE_WINDING_DIR = Path(output_dir / 'slice_winding_images')
    SLICE_WINDING_DIR.mkdir(parents=True, exist_ok=True)

    SLICE_SEGMENTATION_DIR = Path(output_dir / 'slice_segmentation_images')
    SLICE_SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)

    full_volume = preprocess(input_dir=input_dir, slice_img_dir=SLICE_IMG_DIR, slice_segmentation_dir=SLICE_SEGMENTATION_DIR)

    det_vol = determine_winding(volume=full_volume)

    fixed_vol = fix_winding(volume=det_vol, target_winding='+')

    save_winding(volume=fixed_vol, slice_winding_dir=SLICE_WINDING_DIR, slice_img_dir=SLICE_IMG_DIR)

    mesh_map = align_segments(volume=fixed_vol)

    with open(f'{output_dir}/mesh_map.pkl', 'wb') as fid:
        pickle.dump(mesh_map, fid)

    json_ready_map = make_json_serializable(mesh_map)

    with open(f"{output_dir}/mesh_map.json", "w") as fid:
        json.dump(json_ready_map, fid, indent=4)
    
    export_mesh_map_to_obj(mesh_map, f'{output_dir}/aligned_volume.obj')




if __name__ == "__main__":
    main()