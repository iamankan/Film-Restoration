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
from sklearn.cluster import DBSCAN


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


def dbscan(film_slice: str, output_folder: str, cv_show: bool=True, cv_wait_key: bool=False, threshold_factor: float = 2.0, num_seg_points: int = 1000):
    
    if output_folder:
        uuid_id = str(uuid.uuid4())
        save_at = Path(output_folder) / uuid_id
        print(f"Saving things at {save_at}")
        save_at.mkdir(parents=True, exist_ok=True)
        with open(f'{save_at}/details.txt', 'w') as fid:
            fid.write(f'Film slice: {film_slice}\nthreshold-factor: {threshold_factor}\nmax_num_of_seg_points: {num_seg_points}\n')
    else:
        print(f"Nothing is being saved. So, you will see the outputs. And press a key after every output to see the next.")
        cv_show = True
        cv_wait_key = True

    if cv_wait_key:
        cv_wait_key_val = 0 # Wait if true
    else:
        cv_wait_key_val = 1 # Don't wait if false

    img = cv2.imread(film_slice)
    if img is None:
        print(f"Error: Could not load image {film_slice}")
        return

    print(f'Shape of the film slice is: {img.shape}')
    img_min = img[:, :, 0].min()
    img_max = img[:, :, 0].max()
    print(f'min: {img_min}, max: {img_max}')

    cy, cx, _ = img.shape

    center_point = (cy//2, cx//2)

    _, binary = cv2.threshold(img[:, :, 0], (img_max - img_min) // threshold_factor, img_max, cv2.THRESH_BINARY)
    skeleton = cv2.ximgproc.thinning(binary)

    # Get coordinates of skeleton pixels
    coords = np.column_stack(np.where(skeleton > 0))  # shape: (N, 2), format: (row, col)

    if len(coords) == 0:
        print("No skeleton pixels found.")
        return

    # Apply DBSCAN
    db = DBSCAN(eps=5, min_samples=5).fit(coords)
    labels = db.labels_
    unique_labels = set(labels)

    # Create overlay image
    color_overlay = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    for cluster_id in unique_labels:
        rgba = colors(cluster_id % 20) if cluster_id != -1 else (0.5, 0.5, 0.5, 1.0)  # gray for noise
        cluster_color = tuple(int(c * 255) for c in rgba[:3])

        cluster_points = coords[labels == cluster_id]
        for x, y in cluster_points:
            color_overlay[x, y] = cluster_color

    # Display result
    cv2.imshow("DBSCAN Clusters on Skeleton", color_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-slice', '-s', help="Path to the first slice.", type=str)
    parser.add_argument('--threshold-factor', '-t', help="Factor by which the threshold is divided.", type=float, default=2.0)
    parser.add_argument('--num-seg-points', '-n', help="Minimum number of segmentation points.", type=int, default=1000)
    parser.add_argument('--output-folder','-o', help="Output folder where the images will be saved. Default: Nothing will be saved", type=str)
    parser.add_argument('--cv-wait-key',help="Enter to wait.", action='store_true')
    parser.add_argument('--cv-show',help="Enter to wait.", action='store_true')
    args = parser.parse_args()
    film_slice = args.input_slice
    threshold_factor = args.threshold_factor
    num_seg_points = args.num_seg_points
    output_folder = args.output_folder
    cv_wait_key = args.cv_wait_key
    cv_show = args.cv_show
    print(f'CV_WAIT_KEY: {cv_wait_key}, CV_SHOW: {cv_show}')

    if threshold_factor==0:
        print("Threshold cannot be zero.")
        return
    dbscan(film_slice=film_slice, threshold_factor=threshold_factor, num_seg_points=num_seg_points, output_folder=output_folder, cv_wait_key=cv_wait_key,
         cv_show=cv_show)


if __name__ == "__main__":
    main()