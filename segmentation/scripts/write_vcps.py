import numpy as np
from pathlib import Path
import json

# From quicksegment https://github.com/educelab/quick-segment/blob/develop/qs/data/vcps.py 
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


def write_seg_json(path, pointset):
    # Open output file and write ASCII header
    with open(path + "/pointset.json", 'w', encoding='utf-8') as file:
        json.dump(pointset, file)

def write_metadata(path, vol, uuid):

    data = {
        "name": uuid,
        "type": "seg",
        "uuid": uuid,
        "vcps": "pointset.vcps",
        "volume": vol
    }

    with open(path + "/meta.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2))

if __name__ == "__main__":
    vcps_dir = Path("/localdisk0/test_vcps")
    vcps_dir.mkdir(exist_ok=True, parents=True)
    pointset = [[]]
    for i in range(5):
        for j in range(3):
            pointset[0].append([i, j, 0.0])
    
    for i in range(5):
        for j in range(3):
            pointset[0].append([i, j, 0.0])
    
    for i in range(5):
        for j in range(2):
            pointset[0].append([i, j, 0.0])
            
    pointset = np.array(pointset)
    print(pointset.shape)
    write_ordered_vcps(path=vcps_dir, pointset=pointset)