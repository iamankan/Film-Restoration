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


def thin(film_slice: str, threshold_factor: float = 2.0, num_seg_points: int = 1000):
    img = cv2.imread(film_slice)
    if img is None:
        print(f"Error: Could not load image {film_slice}")
        return

    print(f'Shape of the film slice is: {img.shape}')
    img_min = img[:, :, 0].min()
    img_max = img[:, :, 0].max()
    print(f'min: {img_min}, max: {img_max}')

    cy, cx, _ = img.shape

    center_point = (cx//2, cy//2)

    _, binary = cv2.threshold(img[:, :, 0], (img_max - img_min) // threshold_factor, img_max, cv2.THRESH_BINARY)
    skeleton = cv2.ximgproc.thinning(binary)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

    for i in range(1, num_labels):  # skip background
        component_mask = (labels == i).astype(np.uint8)
        end_points = find_endpoints(component_mask=component_mask)
        print(f'Endpoints: {end_points}')
        num_points = cv2.countNonZero(component_mask)
        print(f"\nComponent {i}: {num_points} points")

        if num_points < num_seg_points:
            print(f"Number of points less than {num_seg_points}. Skipping component.")
            continue

        y_branch_points = []
        padded = np.pad(component_mask, 1, mode='constant')
        h, w = component_mask.shape

        for x in range(1, h + 1):
            for y in range(1, w + 1):
                if padded[x, y] == 1 and is_y_branch_point(x, y, padded):
                    y_branch_points.append((x - 1, y - 1))

        if len(y_branch_points) > 0:
            print(f"Component {i} contains Y-branching structure ({len(y_branch_points)} Y-points).")
        else:
            print(f"Component {i} has no Y-junctions.")

        # Draw pruned skeleton on original image as green dots
        color_overlay = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

        ys, xs = np.where(component_mask == 1)
        for (y, x) in zip(ys, xs):
            color_overlay[y, x] = [0, 255, 255]
        
        for yidx, yb in enumerate(y_branch_points):
            cv2.circle(color_overlay, (yb[1], yb[0]), 1, (255,0,0), 1) # green 
        
        for epidx, ep in enumerate(end_points):
            cv2.circle(color_overlay, (ep[1], ep[0]), 2+epidx, (0,0,255), 5) # red 
                




        cv2.imshow(f"Component {i} - Pruned Skeleton Overlay", color_overlay)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-slice', '-s', help="Path to the first slice.", type=str)
    parser.add_argument('--threshold-factor', '-t', help="Factor by which the threshold is divided.", type=float, default=2.0)
    parser.add_argument('--num-seg-points', '-n', help="Minimum number of segmentation points.", type=int, default=1000)
    args = parser.parse_args()
    film_slice = args.input_slice
    threshold_factor = args.threshold_factor
    num_seg_points = args.num_seg_points
    if threshold_factor==0:
        print("Threshold cannot be zero.")
        return
    thin(film_slice=film_slice, threshold_factor=threshold_factor, num_seg_points=num_seg_points)


if __name__ == "__main__":
    main()