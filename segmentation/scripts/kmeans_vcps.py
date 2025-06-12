import argparse
import cv2
import numpy as np
from pathlib import Path
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from scipy.ndimage import convolve
import networkx as ntx
from scipy.spatial.distance import euclidean
matplotlib.use('TkAgg')
from collections import deque
import heapq
from math import sqrt 
import uuid
from itertools import combinations
from sklearn.cluster import KMeans
import json
import datetime as dt


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

def write_vcps(filename, points):
    
    with open(filename, 'wb') as f:
        # Write ASCII header
        f.write(f"width: {points.shape[0]}\n".encode('ascii'))
        f.write(f"height: {1}\n".encode('ascii'))
        f.write(f"dim: {points.shape[1]}\n".encode('ascii'))
        f.write(f"ordered: true\n".encode('ascii'))
        f.write(f"type: double\n".encode('ascii'))
        f.write(f"version: 1\n".encode('ascii'))
        f.write(b"<>\n")
        
        f.write(points.astype(np.double).tobytes())


def euclidean(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def dijkstra_cheapest_path(skeleton, start, end, center_point):
    h, w = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    dist = np.full(skeleton.shape, np.inf)
    parent = dict()

    dist[start] = euclidean(start, center_point)
    heap = [(dist[start], start)]

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    while heap:
        cost, (y, x) = heapq.heappop(heap)
        if visited[y, x]:
            continue
        visited[y, x] = True

        if (y, x) == end:
            # Reconstruct path
            path = []
            while (y, x) != start:
                path.append((y, x))
                y, x = parent[(y, x)]
            path.append(start)
            path.reverse()
            return path, dist[end]

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0:
                new_cost = cost + euclidean((ny, nx), center_point)
                if new_cost < dist[ny, nx]:
                    dist[ny, nx] = new_cost
                    parent[(ny, nx)] = (y, x)
                    heapq.heappush(heap, (new_cost, (ny, nx)))

    return None, np.inf

def shortest_path_skeleton(skeleton, start, end):
    h, w = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    parent = dict()

    queue = deque([start])
    visited[start] = True

    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    while queue:
        y, x = queue.popleft()
        if (y, x) == end:
            # Reconstruct path from end to start
            path = []
            while (y, x) != start:
                path.append((y, x))
                y, x = parent[(y, x)]
            path.append(start)
            path.reverse()
            return path

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if skeleton[ny, nx] > 0 and not visited[ny, nx]:
                    visited[ny, nx] = True
                    parent[(ny, nx)] = (y, x)
                    queue.append((ny, nx))

    return None  # no path found

def find_endpoints(component_mask):
    endpoints = []
    padded = np.pad(component_mask, 1, mode='constant')  # pad to handle edges
    h, w = component_mask.shape

    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if padded[y, x] == 1:
                # Count 8 neighbors
                neighbors = [
                    padded[y-1, x-1], padded[y-1, x], padded[y-1, x+1],
                    padded[y, x-1],             
                    padded[y, x+1],
                    padded[y+1, x-1], padded[y+1, x], padded[y+1, x+1]
                ]
                if sum(neighbors) == 1:
                    # Exactly one neighbor → endpoint
                    endpoints.append((y-1, x-1))  # remove padding offset
    return endpoints

def shortest_path_length(skeleton, start, end):
    h, w = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    dist = np.full_like(skeleton, -1, dtype=int)  # distance array

    queue = deque([start])
    visited[start] = True
    dist[start] = 0

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    while queue:
        y, x = queue.popleft()
        if (y, x) == end:
            return dist[end]  # shortest path length

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if skeleton[ny, nx] > 0 and not visited[ny, nx]:
                    visited[ny, nx] = True
                    dist[ny, nx] = dist[y, x] + 1
                    queue.append((ny, nx))
    return -1  # no path found

def is_path_between_points(skeleton, start, end):
    h, w = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    queue = deque([start])
    visited[start] = True
    
    # 8-connected neighbors relative positions
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    while queue:
        y, x = queue.popleft()
        if (y, x) == end:
            return True
        
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if skeleton[ny, nx] > 0 and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return False


def is_y_branch_point(x, y, img):
    # 8-connected neighbors in clockwise order (P2 to P9)
    neighbors = [
        img[x-1, y],   # P2
        img[x-1, y+1], # P3
        img[x, y+1],   # P4
        img[x+1, y+1], # P5
        img[x+1, y],   # P6
        img[x+1, y-1], # P7
        img[x, y-1],   # P8
        img[x-1, y-1]  # P9
    ]
    neighbors = [int(n > 0) for n in neighbors]

    # Count 0→1 transitions in circular neighborhood
    transitions = sum((neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1) for i in range(8))

    return transitions >= 3

'''
python3 segmentation/scripts/kmeans.py -s /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26855.volpkg/volumes/20250214115505/1000.tif -k 3 -n 100
'''

def thin(volpkg_dir: Path, volume: str, film_slice: str, original_image: np.array, clustered: np.array, save_at: str, cluster_id: int, cluster_mask: np.array, threshold_factor: float = 2.0, cv_show: bool=True, cv_wait_key_val: int=0, num_seg_points: int = 1000):
    
    print(f'Shape of the film slice is: {original_image.shape}')
    img_min = original_image[:, :, 0].min()
    img_max = original_image[:, :, 0].max()
    print(f'min: {img_min}, max: {img_max}')

    cy, cx, _ = original_image.shape

    center_point = (cy//2, cx//2)
    
    _, binary = cv2.threshold(clustered, (img_max - img_min) // threshold_factor, img_max, cv2.THRESH_BINARY)
    
    skeleton = cv2.ximgproc.thinning(binary)

    num_labels, labels = cv2.connectedComponents(skeleton, connectivity=8)
    print(f"Total components: {num_labels} (including background).")
    if save_at:
        save_at_cluster = Path(save_at) / f'cluster_{cluster_id}'
        save_at_cluster.mkdir(parents=True, exist_ok=True)
        colored_path = original_image.copy()
        cv2.imwrite(f'{save_at_cluster}/original_image.jpg', original_image)
        cv2.imwrite(f'{save_at_cluster}/cluster_mask.jpg', cluster_mask)
        cv2.imwrite(f'{save_at_cluster}/cluster.jpg', clustered)
    

    for idx in range(1, num_labels):  # skip background
        component_mask = (labels == idx).astype(np.uint8)

        endpoints = find_endpoints(component_mask=component_mask)
        
        num_points = cv2.countNonZero(component_mask)

        if num_points < num_seg_points:
            # print(f"Number of points less than {num_seg_points}. Skipping component.")
            continue

        print(f"\nComponent {idx} (Cluster: {cluster_id}): {num_points} points")


        y_branch_points = []
        padded = np.pad(component_mask, 1, mode='constant')
        h, w = component_mask.shape

        for x in range(1, h + 1):
            for y in range(1, w + 1):
                if padded[x, y] == 1 and is_y_branch_point(x, y, padded):
                    y_branch_points.append((x - 1, y - 1))

        if len(y_branch_points) > 0:
            print(f"Component {idx} (Cluster: {cluster_id}) contains Y-branching structure ({len(y_branch_points)} Y-points).")
            if save_at:
                with open(f'{save_at}/details.txt', 'a') as fid:
                    fid.write(f'Component {idx} (Cluster: {cluster_id}) contains Y-branching structure ({len(y_branch_points)} Y-points).\n')
        else:
            print(f"Component {idx} (Cluster: {cluster_id}) has no Y-junctions.")
            if save_at:
                with open(f'{save_at}/details.txt', 'a') as fid:
                    fid.write(f"Component {idx} (Cluster: {cluster_id}) has no Y-junctions.")

        # Draw pruned skeleton on original image as green dots
        color_overlay = cv2.cvtColor(original_image.copy(), cv2.COLOR_BGR2RGB)

        ys, xs = np.where(component_mask == 1)
        for (y, x) in zip(ys, xs):
            color_overlay[y, x] = [0, 255, 255]
        
        for yb in y_branch_points:
            cv2.circle(color_overlay, (yb[1], yb[0]), 1, (255,0,0), 1) # green 
        
        pathlen = 0
        start = (0,0)
        end = (0,0)
        for p1, p2 in combinations(endpoints, 2):
            q = shortest_path_length(binary, p1, p2)
            if q >= pathlen:
                pathlen = q
                start, end = p1, p2
        
        print(f'Start: {start}, End: {end}')
        cv2.circle(color_overlay, (start[1], start[0]), 10, (0,255,0),2) # start - Green
        cv2.circle(color_overlay, (end[1], end[0]), 10, (0,0,255),2) # end - Red
        
        shortestpath, shortestdist = dijkstra_cheapest_path(skeleton=binary, start=start, end=end, center_point=center_point)
        print(f'Length of the shortest path between start and end is: {len(shortestpath)} pixels, and cost is {shortestdist}.')
        pointset = [[]]
        slice_no = int(film_slice.stem)
        if save_at:
            with open(f'{save_at_cluster}/segmented_component_{idx}_cluster{cluster_id}.txt', 'w') as f:
                f.write(f'x,y\n')
                for spidx, sp in enumerate(shortestpath):
                    cv2.circle(color_overlay, (sp[1], sp[0]), 1, (0,int(255*(1-(spidx/len(shortestpath)))),int(255*spidx/len(shortestpath))),1)
                    f.write(f'{sp[1]},{sp[0]}\n')
                    pointset[0].append([float(sp[1]), float(sp[0]), float(slice_no)])
                f.write(f'Cost: {shortestdist}')
        else:
            for spidx, sp in enumerate(shortestpath):
                cv2.circle(color_overlay, (sp[1], sp[0]), 1, (0,int(255*(1-(spidx/len(shortestpath)))),int(255*spidx/len(shortestpath))),1)
                pointset[0].append([float(sp[1]), float(sp[0]), float(slice_no)])
        print(f'Length of pointset[0]: {len(pointset[0])}')
        
        if len(pointset[0])>0:
            pointset = np.array(pointset)
            seg_id = f'{get_date()}_kmeans_thin'
            seg_path = volpkg_dir / f'paths/{seg_id}'
            seg_path.mkdir(exist_ok=True, parents=True)
            print(f'Generated segmentation id: {seg_id} and the path is {seg_path}')
            write_ordered_vcps(path=seg_path, pointset=pointset)
            write_metadata(path=seg_path, uuid_val=seg_id, vol=volume)
            print('Finished writing meta.json and pointset.vcps')
        

        colored_path = cv2.bitwise_or(colored_path, color_overlay)
        

        cv2.line(color_overlay, (start[1], start[0]), (end[1], end[0]), (0,0,255), 5)

        if cv_show:
            cv2.imshow(f"Cluster {cluster_id} | Component {idx} - Skeleton Overlay", color_overlay)
            cv2.waitKey(cv_wait_key_val)

        if save_at:
            cv2.imwrite(f'{save_at_cluster}/segmented_component_{idx}_cluster{cluster_id}.jpg', img=color_overlay)
    
    if save_at:
        cv2.imwrite(f'{save_at_cluster}/binary_cluster{cluster_id}.jpg', img=binary)
        cv2.imwrite(f'{save_at_cluster}/total_colored_segmentation_cluster{cluster_id}.jpg', img=colored_path)

    
    

def kmeans(volpkg_dir: Path, film_slice: str, output_folder: str, volume: str, threshold_factor:float=2.0, cv_show: bool=True, cv_wait_key: bool=False, number_of_clusters: int = 3, num_seg_points: int = 1000):
    
    if output_folder:
        uuid_id = str(uuid.uuid4())
        save_at = Path(output_folder) / uuid_id
        print(f"Saving things at {save_at}")
        save_at.mkdir(parents=True, exist_ok=True)
        with open(f'{save_at}/details.txt', 'w') as fid:
            fid.write(f'KMEANS->Thinning\nFilm slice: {film_slice}\nthreshold factor: {threshold_factor}\nnumber of clusters: {number_of_clusters}\nmax_num_of_seg_points: {num_seg_points}\n')
    else:
        print(f"Nothing is being saved. So, you will see the outputs. And press a key after every output to see the next.")
        cv_show = True
        cv_wait_key = True
        save_at = None

    if cv_wait_key:
        cv_wait_key_val = 0 # Wait if true
    else:
        cv_wait_key_val = 1 # Don't wait if false
    


    original_image = cv2.imread(film_slice)
    if original_image is None:
        print("Failed to load original image.")
        return

    img = cv2.imread(film_slice, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image.")
        return

    h, w = img.shape
    flat_img = img.reshape(-1, 1)

    # Apply KMeans
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0, n_init="auto")
    kmeans.fit(flat_img)
    labels = kmeans.labels_.reshape(h, w)

    # Display each cluster separately
    for cluster_id in range(number_of_clusters):
        mask = (labels == cluster_id).astype(np.uint8) * 255  # Binary mask
        cluster_img = cv2.bitwise_and(img, img, mask=mask)
        print(f'cluster_img shape: {cluster_img.shape}')

    for cluster_id in range(number_of_clusters):
        mask = (labels == cluster_id).astype(np.uint8) * 255  # Binary mask
        cluster_img = cv2.bitwise_and(img, img, mask=mask)
        print(f"Starting thinning for cluster {cluster_id}")
        thin(volpkg_dir=volpkg_dir, original_image=original_image, clustered=cluster_img, save_at=save_at, cluster_id=cluster_id, cv_show=cv_show,
             cv_wait_key_val=cv_wait_key_val, num_seg_points=num_seg_points, threshold_factor=threshold_factor, cluster_mask=mask, volume=volume,
             film_slice=film_slice)
    
    print("Press any key to exit the program!")
    cv2.waitKey(cv_wait_key_val)
    cv2.destroyAllWindows()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volpkg', help="Path to the volpkg.", type=str)
    parser.add_argument('--volume', help="Volume ID inside the volpkg.", type=str)
    parser.add_argument('--slice-name', help="Name of the slice to segment. DEFAULT: 0000.tif", type=str, default='0000.tif')

    parser.add_argument('--threshold-factor', '-t', help="Factor by which the threshold is divided.", type=float, default=2.0)
    parser.add_argument('--number-of-clusters', '-k', help="Number of clusters you are expecting. DEFAULT=3", type=int, default=3)
    parser.add_argument('--num-seg-points', '-n', help="Minimum number of segmentation points.", type=int, default=1000)
    parser.add_argument('--output-folder','-o', help="Output folder where the images will be saved. This should be outside volpkg. Default: Nothing will be saved", type=str)
    parser.add_argument('--cv-wait-key',help="Enter to wait.", action='store_true')
    parser.add_argument('--cv-show',help="Enter to wait.", action='store_true')
    args = parser.parse_args()

    volpkg = Path(args.volpkg)
    volume = args.volume
    film_slice = volpkg / f'volumes/{volume}/{args.slice_name}'
    if not film_slice.exists():
        print(f'{film_slice} does not exist!')
        return


    threshold_factor = args.threshold_factor
    number_of_clusters = args.number_of_clusters
    num_seg_points = args.num_seg_points
    output_folder = args.output_folder
    cv_wait_key = args.cv_wait_key
    cv_show = args.cv_show
    print(f'CV_WAIT_KEY: {cv_wait_key}, CV_SHOW: {cv_show}')

    if number_of_clusters==0:
        print("Number of clusters cannot be zero.")
        return
    

    kmeans(volpkg_dir=volpkg, film_slice=film_slice, number_of_clusters=number_of_clusters, num_seg_points=num_seg_points, output_folder=output_folder, 
           cv_wait_key=cv_wait_key, cv_show=cv_show, threshold_factor=threshold_factor, volume=volume)


if __name__ == "__main__":
    main()