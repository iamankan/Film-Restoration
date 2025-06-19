import numpy as np
from pathlib import Path
import argparse
import json
import datetime as dt
from natsort import natsorted
import imageio.v3 as iio
import cv2
import uuid

# From quicksegment https://github.com/educelab/quick-segment/blob/develop/qs/data/vcps.py
def get_date():
    tz = dt.timezone.utc
    return f'{dt.datetime.now(tz).strftime("%Y%m%d%H%M%S")}'


def write_ordered_vcps(path, pointset):
    # Open output file and write ASCII header
    file_path = Path(path) / "pointset.vcps"    
    with file_path.open('wt') as file:
        file.writelines([
            f'width: {pointset.shape[1]}\n', # number of points
            f'height: {pointset.shape[0]}\n', # total number of slices
            f'dim: {pointset.shape[2]}\n', # number of coordinates
            'ordered: true\n',
            'type: double\n',
            'version: 1\n',
            '<>\n'
        ])

    # Reopen in binary append mode
    with file_path.open('ab') as file:
        # Write as doubles
        pointset.tofile(file)


def write_metadata(path, vol, uuid_val):
    data = {
        "name": str(uuid_val),
        "type": "seg",
        "uuid": str(uuid_val),
        "vcps": "pointset.vcps",
        "volume": vol
    }

    with open(path / "meta.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2))


def load_vcps(segment_dir):
    with open(segment_dir / "pointset.vcps", 'rb') as file:
        points = file.read().split(b'<>\n')[1]
        return cloud_to_dict(np.frombuffer(points, dtype='float64'))


def cloud_to_dict(cloud):
    cloud = np.reshape(cloud, (-1, 3))
    lines = dict()

    for point in cloud:
        slice = int(point[2])
        point = [point[0], point[1], slice]
        lines.setdefault(int(slice), []).append(point)

    return lines


def traverse(volpkg_path, volume_path, segment_path, volume_id, segment_id, output_dir, kernel_size, direction, number_of_slices, curr_slice):
    vcps_dict = load_vcps(segment_dir=segment_path)
    slice_keys = list(vcps_dict.keys())
    if curr_slice == None:
        if direction == -1:
            curr_slice_index = slice_keys[0]
        else:
            curr_slice_index = slice_keys[-1]
    else:
        curr_slice_index = curr_slice
    
    print(f'curr-slice-index: {curr_slice_index}')
    all_files = []
    for i in volume_path.iterdir():
        if i.suffix == '.tif' and not i.name.startswith('.'):
            all_files.append(i)
    all_files = natsorted(all_files)
    all_indices = []
    for i in range(curr_slice_index,curr_slice_index+(direction*(number_of_slices+1)), direction):
        all_indices.append(i)
    print(all_indices)
    k_delta = (kernel_size-1)//2
    kernel_ones = np.ones((kernel_size, kernel_size), dtype=np.float64)
    central_idx = np.array([k_delta,k_delta])

    writing_path = Path(output_dir) / str(uuid.uuid4())
    writing_path.mkdir(exist_ok=True, parents=True)
    print(f'Everything will be written to {writing_path}')

    for slice_idx in all_indices[:-2]:
        curr_slice_idx = slice_idx
        next_slice_idx = slice_idx+1

        curr_slice_img = cv2.imread(all_files[curr_slice_idx])
        # curr_slice_img[:,:,1:] = np.zeros_like(curr_slice_img[:,:,1:])

        next_slice_img = cv2.imread(all_files[next_slice_idx])
        # next_slice_img[:,:,1:] = np.zeros_like(next_slice_img[:,:,1:])
        
        next_slice_seg_coords = []

        seg_coords = vcps_dict[curr_slice_idx]
        for seg_order_idx, seg_coord in enumerate(seg_coords):
            curr_slice_x = int(seg_coord[0])
            curr_slice_y = int(seg_coord[1])
            curr_intensity = curr_slice_img[curr_slice_y, curr_slice_x, 0]
            curr_ones = curr_intensity * kernel_ones

            next_kernel = next_slice_img[curr_slice_y-k_delta:curr_slice_y+k_delta+1, curr_slice_y-k_delta:curr_slice_y+k_delta+1, 0]
            next_kernel = next_kernel.astype(np.float64)

            diff_square = abs(np.square(curr_ones) - np.square(next_kernel))
            # diff_square = abs(curr_ones - next_kernel)

            diff_min = diff_square.min()
            min_idx = np.column_stack(np.where(diff_square == diff_min))
            
            if min_idx.shape[0] > 1:
                selected_coord_kernel = central_idx
            else:
                # print(min_idx.shape)
                central_vector = central_idx * np.ones_like(min_idx)
                r = (min_idx - central_vector)
                dist_from_center = np.sqrt(np.square(r[:,0])+np.square(r[:,1]))
                min_dis_from_center_idx = np.argmin(dist_from_center)
                selected_coord_kernel = min_idx[min_dis_from_center_idx]
            next_slice_y, next_slice_x = int(curr_slice_y + selected_coord_kernel[0] - k_delta), int(curr_slice_x + selected_coord_kernel[1] - k_delta)


            next_slice_seg_coords.append([next_slice_x, next_slice_y, next_slice_idx])
            # next_slice_img[next_slice_y, next_slice_x, :] = [next_slice_img[next_slice_y, next_slice_x, 0], seg_order_idx*255//len(seg_coords),255]
            cv2.circle(next_slice_img, (next_slice_x, next_slice_y), radius=1, color=(0, seg_order_idx*255//len(seg_coords),255), thickness=2)
        vcps_dict[next_slice_idx] = next_slice_seg_coords
        cv2.imwrite(f'{writing_path}/slice_{next_slice_idx}.jpg', next_slice_img)
        print(f'Slice-{next_slice_idx} done!')
        # next_slice_img = cv2.cvtColor(next_slice_img, cv2.COLOR_BGR2RGB)
    #     cv2.imshow(f"Slice: {next_slice_idx}", next_slice_img)
    #     cv2.waitKey(1)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volpkg','-v', help='Volpkg path.', type=str)
    parser.add_argument('--volume', help='Volume ID. In volpkg_path/volumes', type=str)
    parser.add_argument('--segment','-s', help='Segment ID. In volpkg_path/paths', type=str)

    parser.add_argument('--current-slice', '-c', help='Current slice index.')
    parser.add_argument('--kernel-size','-k', help='Kernel to search for the intensity.', type=int)
    parser.add_argument('--direction','-d', help='The transition direction. +1/-1. +1: go in positive direction. -1: go in previous direction. DEFAULT: 1', 
                        type=int, choices=[1,-1], default=1)
    parser.add_argument('--number-of-slices','-n', help='Number of slices to traverse.', type=int)
    parser.add_argument('--output-folder','-o', help='Output folder.', type=str)
    args = parser.parse_args()
    
    volpkg_path = Path(args.volpkg)
    volume_id = args.volume
    segment_id = args.segment

    volume_path = volpkg_path / f'volumes/{volume_id}'
    segment_path = volpkg_path / f'paths/{segment_id}'

    curr_slice = args.current_slice
    kernel_size = args.kernel_size
    direction = args.direction
    number_of_slices = args.number_of_slices
    output_folder = Path(args.output_folder)

    print(f'Curr-slice: {curr_slice}')

    traverse(volpkg_path=volpkg_path, volume_path=volume_path, segment_path=segment_path, volume_id=volume_id, segment_id=segment_id, 
             output_dir=output_folder, kernel_size=kernel_size, direction=direction, number_of_slices=number_of_slices, curr_slice=curr_slice)



if __name__ == "__main__":
    main()