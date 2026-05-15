from pathlib import Path
import imageio.v3 as iio
import argparse
from natsort import natsorted
import shutil
import numpy as np
import cv2

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
        self.in_idx = None
        self.out_idx = None
        self._avg_rad = None
        self._start_rad = None
        self._end_rad = None
        self._combined_score = None
    
    def get_combined_score(self)->float:
        return self._combined_score
    
    def set_combined_score(self, combined_score:float):
        self._combined_score = combined_score

    def get_avg_rad(self)->float:
        return self._avg_rad
    
    def set_avg_rad(self, avg_rad:float):
        self._avg_rad = avg_rad

    def get_start_rad(self)->float:
        return self._start_rad
    
    def set_start_rad(self, start_rad:float):
        self._start_rad = start_rad

    def get_end_rad(self)->float:
        return self._end_rad
    
    def set_end_rad(self, end_rad:float):
        self._end_rad = end_rad

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
    def __init__(self, z_value:int, all_segments:list[Segment]):
        self._z_value = z_value
        self._all_segments = all_segments
    
    def get_total_segments(self) -> int:
        return len(self._all_segments)
    
    def get_segment_at(self, idx) -> Segment:
        return self._all_segments[idx]
    
    def update_segment_at(self, idx, new_segment):
        self._all_segments[idx] = new_segment
    
    def get_z_value(self)->int:
        return self._z_value
    
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


def radial_sorting(volume:Volume)->Volume:
    print(f'Radial sorting')
    all_slices = volume.get_all_slices()
    for slc in all_slices:
        all_segs = slc.get_all_segments()
        k=0.8
        for seg in all_segs:
            pts = seg.get_segment()
            rad = np.array([p.get_polar()[0] for p in pts])
            theta = np.array([p.get_polar()[1] for p in pts])
            avg_theta = np.mean(theta)
            avg_rad = np.mean(rad)
            seg.set_combined_score(combined_score=seg.get_start_point().get_polar()[1] + seg.get_end_point().get_polar()[0])
            seg.set_avg_rad(avg_rad=avg_rad)
            seg.set_start_rad(start_rad=seg.get_start_point().get_polar()[0])
            seg.set_end_rad(end_rad=seg.get_end_point().get_polar()[0])
            print(f'Seg#{seg.get_seg_idx()}, avg-rad: {seg.get_avg_rad()}, start-rad: {seg.get_start_rad()}, end-rad: {seg.get_end_rad()}, combined-score: {seg.get_combined_score}')
    return volume

def save_radial(volume: Volume, slice_rad_dir: Path, slice_img_dir: Path):
    all_slices = volume.get_all_slices()
    vol_cg = volume.get_centroid().get_point()
    for slc in all_slices:
        dest_fname = f'{slice_rad_dir}/{slc.get_z_value():04d}.jpg'
        original_fname = f'{slice_img_dir}/{slc.get_z_value():04d}.jpg'
        img = iio.imread(original_fname)
        rad_map = {}
        for seg in slc.get_all_segments():
            rad_map[seg.get_seg_idx()] = seg.get_combined_score()
        print(f'avg_rad_map: {rad_map}')
        sorted_rad_idx = sorted(rad_map, key=rad_map.get)
        print(f'sorted_rad_idx: {sorted_rad_idx}')
        n = len(sorted_rad_idx)
        for i, seg_idx in enumerate(sorted_rad_idx):
            seg = slc.get_segment_at(idx=seg_idx)
            color = (0, 255 * (1 - (i/(n-1))), 255 * (i/(n-1)))
            seg_pts = seg.get_segment()
            for pt in seg_pts:
                p = pt.get_point()
                cv2.circle(img, (int(p[0]),int(p[1])), 2, color, 2)
            cv2.imwrite(f'{slice_rad_dir}/{slc.get_z_value():04d}_serial_{i}.jpg', img)
        cv2.imwrite(dest_fname, img)



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
    for slice_id in natsorted(input_dir.iterdir()):
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
        single_slice = Slice(z_value=z_value, all_segments=all_segments)
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

    SLICE_RADIAL_DIR = Path(output_dir / 'slice_radial_images')
    SLICE_RADIAL_DIR.mkdir(parents=True, exist_ok=True)

    full_volume = preprocess(input_dir=input_dir, slice_img_dir=SLICE_IMG_DIR, slice_segmentation_dir=SLICE_SEGMENTATION_DIR)

    det_vol = determine_winding(volume=full_volume)

    fixed_vol = fix_winding(volume=det_vol, target_winding='+')

    save_winding(volume=fixed_vol, slice_winding_dir=SLICE_WINDING_DIR, slice_img_dir=SLICE_IMG_DIR)

    rad_vol = radial_sorting(volume=fixed_vol)

    save_radial(volume=rad_vol, slice_rad_dir=SLICE_RADIAL_DIR, slice_img_dir=SLICE_IMG_DIR)


if __name__ == "__main__":
    main()