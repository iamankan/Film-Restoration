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


def traverse(volpkg_path, volume_path, segment_path, volume_id, segment_id, output_dir, kernel_size, direction, number_of_slices, curr_slice,
             similarity_kernel_size):
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
    curr_slice_kernel_delta = (kernel_size-1)//2
    next_slice_kernel_delta = (similarity_kernel_size-1)//2

    curr_kernel_ones = np.ones((kernel_size, kernel_size), dtype=np.float64)
    next_kernel_ones = np.ones((similarity_kernel_size, similarity_kernel_size), dtype=np.float64)

    curr_central = np.array([curr_slice_kernel_delta,curr_slice_kernel_delta])
    next_central = np.array([next_slice_kernel_delta,next_slice_kernel_delta])


    writing_path = Path(output_dir) / f'{str(uuid.uuid4())}_seg_{segment_id}'
    writing_path.mkdir(exist_ok=True, parents=True)
    print(f'Everything will be written to {writing_path}')

    for curr_slice_idx in all_indices[:-2]:
        curr_img_path = all_files[curr_slice_idx]
        curr_slice_img = cv2.imread(curr_img_path)
        curr_slice_img_norm = curr_slice_img[:,:,0].copy()
        curr_slice_img_norm = curr_slice_img_norm/curr_slice_img.max()


        next_slice_idx = curr_slice_idx+1
        next_img_path = all_files[next_slice_idx]
        next_slice_img = cv2.imread(next_img_path)
        next_slice_img_norm = next_slice_img[:,:,0].copy()
        next_slice_img_norm = next_slice_img_norm/next_slice_img.max()

        curr_coords = vcps_dict[curr_slice_idx]
        next_seg_coords = []
        for curr_coord_idx, curr_coord in enumerate(curr_coords):
            curr_x = int(curr_coord[0])
            curr_y = int(curr_coord[1])
            curr_kernel = curr_slice_img_norm[curr_y-curr_slice_kernel_delta:curr_y+curr_slice_kernel_delta+1, curr_x-curr_slice_kernel_delta:curr_x+curr_slice_kernel_delta+1]
            
            next_kernel = next_slice_img_norm[curr_y-next_slice_kernel_delta:curr_y+next_slice_kernel_delta+1, curr_x-next_slice_kernel_delta:curr_x+next_slice_kernel_delta+1]
            
            next_h, next_w = next_kernel.shape

            sim_result = -1*np.ones((next_h, next_w))
            # print(sim_result.shape)

            for h in range(next_h):
                for w in range(next_w):
                    ih = int(curr_y+h-1)
                    iw = int(curr_x+w-1)
                    next_box = next_slice_img_norm[ih-curr_slice_kernel_delta:ih+curr_slice_kernel_delta+1, iw-curr_slice_kernel_delta:iw+curr_slice_kernel_delta+1]
                    next_flatten = next_box.flatten()
                    curr_flatten = curr_kernel.flatten()
                    if np.linalg.norm(next_flatten) == 0 or np.linalg.norm(curr_flatten) == 0:
                        # print(f'next_flatten is {next_flatten} and curr_flatten is {curr_flatten}')
                        cosine_similarity = -1.0
                    else:
                        cosine_similarity=(next_flatten@curr_flatten)/(np.linalg.norm(next_flatten)*np.linalg.norm(curr_flatten))
                    sim_result[h,w] = cosine_similarity
            min_idx = np.column_stack(np.where(sim_result == sim_result.max()))
            # if np.isnan(sim_result.any()):
            #     print(f'shape of min-idx: {min_idx.shape} and the min is {sim_result.min()}\n{sim_result}')
            if min_idx.shape[0] > 1:
                next_central_vector = next_central * np.ones_like(min_idx)
                r = next_central_vector - min_idx
                dist = np.sqrt(np.square(r[:,0])+np.square(r[:,1]))
                next_kernel_y, next_kernel_x = min_idx[np.argmin(dist)]
            else:
                next_kernel_y, next_kernel_x = min_idx[0]
            next_y, next_x = int(curr_y+next_kernel_y-next_slice_kernel_delta), int(curr_x+next_kernel_x-next_slice_kernel_delta)
            next_seg_coords.append([next_x, next_y, next_slice_idx])


            cv2.circle(next_slice_img, (next_x, next_y), 2, (0,curr_coord_idx*255//len(curr_coords), 255), 2)


            # break
        vcps_dict[next_slice_idx]=next_seg_coords
        print(f'done... Slice-{next_slice_idx}')
        cv2.imwrite(f'{writing_path}/Slice-{next_slice_idx}.jpg', next_slice_img)
        # cv2.imshow(f'Slice-{curr_slice_idx}', next_slice_img)
        # cv2.waitKey(1)
        # break
    # cv2.waitKey(0)
    cv2.destroyAllWindows()





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volpkg','-v', help='Volpkg path.', type=str)
    parser.add_argument('--volume', help='Volume ID. In volpkg_path/volumes', type=str)
    parser.add_argument('--segment','-s', help='Segment ID. In volpkg_path/paths', type=str)

    parser.add_argument('--current-slice', '-c', help='Current slice index.', type=int)
    parser.add_argument('--kernel-size','-k1', help='Kernel to search for the intensity.', type=int)
    parser.add_argument('--similarity-kernel-size','-k2', help='Kernel to search for the intensity similarity.', type=int)
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
    similarity_kernel_size = args.similarity_kernel_size
    direction = args.direction
    number_of_slices = args.number_of_slices
    output_folder = Path(args.output_folder)

    print(f'Curr-slice: {curr_slice}')

    traverse(volpkg_path=volpkg_path, volume_path=volume_path, segment_path=segment_path, volume_id=volume_id, segment_id=segment_id, 
             output_dir=output_folder, kernel_size=kernel_size, direction=direction, number_of_slices=number_of_slices, curr_slice=curr_slice,
             similarity_kernel_size=similarity_kernel_size)



if __name__ == "__main__":
    main()


'''
Commands to run:
python3 segmentation/scripts/kmeans_vcps.py --volpkg /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26861.volpkg --volume 20250214141357 --slice-name 1000.tif -t 3 -k 3 -n 5000 -s 1000 -o /localdisk0/moma-test-W26861
python3 segmentation/scripts/traverse_cos_sim.py --volpkg /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26861.volpkg --volume 20250214141357 -c 1000 --segment 20250619022247_kmeans_thin -k1 3 -k2 3 --direction +1 --number-of-slices 5 --output-folder /localdisk0/moma_vcps_traverse_test_W26861/

python3 segmentation/scripts/traverse_cos_sim.py --volpkg /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26868.volpkg --volume 20250214142323 -c 1500 --segment 20250619052644_kmeans_thin -k1 3 -k2 3 --direction +1 --number-of-slices 2500 --output-folder /localdisk0/moma_vcps_traverse_test_W26868/
python3 segmentation/scripts/traverse_cos_sim.py --volpkg /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26868.volpkg --volume 20250214142323 -c 1500 --segment 20250619052630_kmeans_thin -k1 3 -k2 3 --direction +1 --number-of-slices 2500 --output-folder /localdisk0/moma_vcps_traverse_test_W26868/


python3 segmentation/scripts/kmeans_vcps.py --volpkg /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26858.volpkg --volume 20250213145507 --slice-name 1000.tif -t 3 -k 3 -n 5000 -s 1000 -o /localdisk0/moma-segment-W26861

python3 segmentation/scripts/traverse_cos_sim.py --volpkg /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26861.volpkg --volume 20250214141357 -c 1000 --segment 20250619221456_kmeans_thin -k1 3 -k2 3 --direction +1 --number-of-slices 2000 --output-folder /localdisk0/moma_vcps_traverse_test_W26861/

'''