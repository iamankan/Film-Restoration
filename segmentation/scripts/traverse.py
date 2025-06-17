import numpy as np
from pathlib import Path
import argparse
import json
import datetime as dt
from natsort import natsorted

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


def traverse(volpkg_path, volume_path, segment_path, volume_id, segment_id, output_dir, kernel_size, direction, number_of_slices):
    vcps = load_vcps(segment_dir=segment_path)
    file_name = natsorted([x for x in volume_path.iterdir()])
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volpkg','-v', help='Volpkg path.', type=str)
    parser.add_argument('--volume', help='Volume ID. In volpkg_path/volumes', type=str)
    parser.add_argument('--segment','-s', help='Segment ID. In volpkg_path/paths', type=str)

    parser.add_argument('--kernel-size','-k', help='Kernel to search for the intensity.', type=int)
    parser.add_argument('--direction','-d', help='The transition direction. +1/-1. +1: go in positive direction. -1: go in previous direction. DEFAULT: 1', 
                        choices=[1,-1], default=1)
    parser.add_argument('--number-of-slices','-n', help='Number of slices to traverse.', type=int)
    parser.add_argument('--output-folder','-o', help='Output folder.', type=str)
    args = parser.parse_args()
    
    volpkg_path = Path(args.volpkg)
    volume_id = args.volume
    segment_id = args.segment

    volume_path = volpkg_path / f'volumes/{volume_id}'
    segment_path = volpkg_path / f'paths/{segment_id}'

    kernel_size = args.kernel_size
    direction = args.direction
    number_of_slices = args.number_of_slices
    output_folder = Path(args.output_folder)



