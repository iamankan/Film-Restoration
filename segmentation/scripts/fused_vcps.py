import argparse
from pathlib import Path
import uuid
import cv2
import numpy as np
import skan as csr
import networkx as nx
from skimage.morphology import skeletonize
from skimage.util import invert
from itertools import combinations
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.cluster import DBSCAN
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from scipy.spatial.distance import cdist

"""
This does remove and try to seperate fused regions of the films.

"""


def find_nearest_pairs(points):
    points = points.copy()
    pairs = []

    while len(points) > 1:
        # Compute pairwise distances
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)  # prevent self-match

        # Find the closest pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        pt1, pt2 = points[i], points[j]
        pairs.append((pt1, pt2))

        # Remove the matched points
        points = np.delete(points, [i, j], axis=0)

    return pairs


def get_date():
    tz = dt.timezone.utc
    return f'{dt.datetime.now(tz).strftime("%Y%m%d%H%M%S")}'


def get_longest_path(G):
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    max_path = []
    max_len = 0

    for u, v in combinations(endpoints, 2):
        try:
            path = nx.shortest_path(G, source=u, target=v)
            if len(path) > max_len:
                max_path = path
                max_len = len(path)
        except nx.NetworkXNoPath:
            continue

    return max_path


def skeleton_to_graph(skeleton):
    G = nx.Graph()
    rows, cols = skeleton.shape
    for y in range(rows):
        for x in range(cols):
            if skeleton[y, x] == 0:
                continue
            # Add a node for every white pixel
            G.add_node((y, x))
            # 8-connectivity neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx_ < cols and skeleton[ny, nx_] > 0:
                        G.add_edge((y, x), (ny, nx_))
    return G

def get_endpoints(G):
    return [n for n in G.nodes if G.degree[n] == 1]

def get_branchpoints(G):
    return [n for n in G.nodes if G.degree[n] > 2]

def prune_short_branches(G, min_length=20):
    endpoints = get_endpoints(G) # (y,x),(y,x),...
    print(f'Total number of endpoints: {len(endpoints)}')
    for ep in endpoints:
        # Perform DFS or BFS to find nearest junction or endpoint
        path = [ep]
        visited = set(path)
        current = ep

        while True:
            neighbors = [n for n in G.neighbors(current) if n not in visited]
            if not neighbors:
                break
            next_node = neighbors[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            if G.degree[current] != 3:
                break  # stop at junction or another endpoint

        if len(path) < min_length:
            print("Removed path")
            G.remove_nodes_from(path[:-1])


def remove_cycles(G):
    cycles = nx.cycle_basis(G.copy())
    for cycle in cycles:
        print(f'Found cycle! Removing.')
        G.remove_nodes_from(cycle)

def graph_to_skeleton(G, shape):
    img = np.zeros(shape, dtype=np.uint8)
    for y, x in G.nodes:
        img[y, x] = 255
    return img

def prune_skeleton_with_graph(binary_skeleton, min_branch_length=20, min_cycle_length=20):
    # Assume binary_skeleton is a 0/255 skeletonized image
    skeleton = (binary_skeleton > 0).astype(np.uint8)
    G = skeleton_to_graph(skeleton)
    # draw_graph_with_endpoints(G, '/localdisk0/images_saved/original.png')
    prune_short_branches(G, min_length=min_branch_length)
    # draw_graph_with_endpoints(G, '/localdisk0/images_saved/after_shortest batch removal.png')
    remove_cycles(G)
    # draw_graph_with_endpoints(G, '/localdisk0/images_saved/after_cycle_removal.png')
    final_skeleton = graph_to_skeleton(G, skeleton.shape)
    connected_components = list(nx.connected_components(G))
    print(f"Final Number of connected components: {len(connected_components)}")
    final_endpoints = get_endpoints(G)
    paths = {}
    for u,v in combinations(final_endpoints, 2): # NC2
        path = nx.shortest_path(G, u, v)
        paths[len(path)]=path
    dist = paths.keys()
    shortest_path = paths[min(dist)]
    return final_skeleton, shortest_path

def draw_graph_with_endpoints(G, output_path=None):
    pos = {n: (n[1], -n[0]) for n in G.nodes}
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    junctions = [n for n in G.nodes if G.degree[n] >= 3]
    regular = [n for n in G.nodes if G.degree[n] == 2]

    plt.figure(figsize=(100, 100))
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=0.3)
    nx.draw_networkx_nodes(G, pos, nodelist=regular, node_color='blue', node_size=2, label='Regular')
    nx.draw_networkx_nodes(G, pos, nodelist=endpoints, node_color='green', node_size=10, label='Endpoints')
    nx.draw_networkx_nodes(G, pos, nodelist=junctions, node_color='red', node_size=10, label='Junctions')

    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def angle_between_vectors_in_degrees(v1, v2):
    # Normalize the vectors
    v1 = v1 / (np.linalg.norm(v1))
    v2 = v2 / (np.linalg.norm(v2))
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def find_index_in_contour(p0, contour):
    distances = np.linalg.norm(contour - p0, axis=1)
    return np.argmin(distances)


def segment(volpkg_path: Path, volume_id: str, output_dir: Path, slice_name: str, threshold: int, min_connected_points: int, connectivity:int, 
            gaussian_kernel: int,
            thinning_algorithm: int, num_connected_components: int,
            junction_angle_threshold: float, junction_window_size: int, junction_window_min: int):
    print(f'Volpkg path: {volpkg_path}, Volume ID: {volume_id}, output-dir: {output_dir}, slice-name: {slice_name}, threshold: {threshold}')
    slice_image_path = Path(f'{volpkg_path}/volumes/{volume_id}/{slice_name}')
    slice_image = cv2.imread(slice_image_path)
    assert slice_image.any(), f'{slice_image_path} not present'
    print(f'Shape of the slice image is {slice_image.shape}')
    dummy_image = slice_image.copy()
    window_size = junction_window_size
    angle_threshold = junction_angle_threshold
    length = 15
    junction_window_max = window_size

    if output_dir:
        print(f'Saving auxiliary output to {output_dir}')
        cv2.imwrite(f'{output_dir}/dummy_image.jpg', dummy_image)
    working_image = dummy_image[:,:,0]
    working_image = cv2.GaussianBlur(working_image, (gaussian_kernel, gaussian_kernel), 0)
    working_image_norm = (working_image - working_image.min())/(working_image.max() - working_image.min())
    print(working_image_norm.min(), working_image_norm.max())
    _, working_image_threshold = cv2.threshold(src=working_image_norm, thresh=working_image_norm.max()/threshold, maxval=working_image_norm.max(), 
                                               type=cv2.THRESH_BINARY)
    working_image_threshold = working_image_threshold.astype(np.uint8)
    print(working_image_threshold.min(), working_image_threshold.max(), working_image_threshold.dtype)
    if output_dir:
        cv2.imwrite(f'{output_dir}/threshold_image.jpg', working_image_threshold*255)

    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(image=working_image_threshold, connectivity=connectivity)
    mask = {}
    print(f"Number of connected components: {num_labels}")
    for i in range(1, num_labels): # 0 is the background, always. So, starting from 1
        componentMask = (labels_im == i).astype("uint8")
        if len(np.where(componentMask==1)[0]) >= min_connected_points:
            mask[i] = componentMask
    
    components = mask.keys()
    print(f"Number of acceptable connected components: {len(components)}")
    junctions = []
    if len(components) > 0:
        for component_id in components:
            print(f'Working on component-{component_id}')
            componentMask = mask[component_id]
            if output_dir:
                component_dir = Path(output_dir) / f'Component_{component_id}'
                component_dir.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(f'{component_dir}/componentMask_{i}.jpg', componentMask*255)
            tmp_img = dummy_image.copy()
            tmp_img[:,:,0] = tmp_img[:,:,0]*componentMask
            tmp_img[:,:,1] = tmp_img[:,:,1]*componentMask
            tmp_img[:,:,2] = tmp_img[:,:,2]*componentMask
            if output_dir:
                cv2.imwrite(f'{component_dir}/maskedcomponent_{component_id}.jpg', tmp_img)
            print(f'Componentmask max, min: {componentMask.max(), componentMask.min()}')
            thinned_mask = cv2.ximgproc.thinning(componentMask*255, thinningType=thinning_algorithm)
            if output_dir:
                cv2.imwrite(f'{component_dir}/thinned_mask_component_{component_id}.jpg', thinned_mask)
            """
            Paper reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8219709
            """
            # Try to detect fused regions
            print(f'Trying to detect fused regions')

            binary_image = componentMask.copy()

            contours, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            print(len(contours))

            composite_fused_canvas = np.zeros_like(dummy_image)
            composite_fused_canvas[:,:,0] = componentMask*255
            composite_fused_canvas[:,:,1] = componentMask*255
            composite_fused_canvas[:,:,2] = componentMask*255
            
            # Let's detect the junction now
            for c_idx, contour in enumerate(contours):
                contour_image = np.zeros_like(binary_image)
                cv2.drawContours(contour_image, contour, contourIdx=-1, color=1, thickness=1)
                if output_dir:
                    cv2.imwrite(f'{component_dir}/edges_{component_id}_contour_{c_idx}.jpg', contour_image*255)

                fused_canvas = np.zeros_like(dummy_image)
                fused_canvas[:,:,0] = componentMask*255
                fused_canvas[:,:,1] = componentMask*255
                fused_canvas[:,:,2] = componentMask*255
                # junction detection
                edge_image = np.zeros_like(fused_canvas)
                edge_image[:,:,0] = contour_image*255
                edge_image[:,:,1] = contour_image*255
                edge_image[:,:,2] = contour_image*255

                contour_ordered_coordinates = np.squeeze(contour)
                n = len(contour_ordered_coordinates)
                color_canvas = np.zeros_like(dummy_image)
                for i in range(n):
                    frac = i/n
                    cv2.circle(color_canvas, (contour_ordered_coordinates[i][0], contour_ordered_coordinates[i][1]),
                               1, (0, int((1-frac)*255), int(frac*255)), 1)
                if output_dir:
                    cv2.imwrite(f'{component_dir}/ordered_edges_{component_id}_contour_{c_idx}.jpg', color_canvas)
                for i in range(0, n, window_size):
                    p0 = contour_ordered_coordinates[i]
                    angles = []
                    for offset in range(junction_window_min, window_size + 1):
                        pl = contour_ordered_coordinates[(i - offset) % n]
                        pr = contour_ordered_coordinates[(i + offset) % n]

                        p0pl = p0 - pl
                        p0pr = p0 - pr

                        angle = angle_between_vectors_in_degrees(p0pl, p0pr)
                        angles.append(angle)

                    avg_angle = np.mean(angles)
                    pl = contour_ordered_coordinates[(i - window_size) % n]
                    pr = contour_ordered_coordinates[(i + window_size) % n]

                    if avg_angle < angle_threshold:
                        pts = np.array(np.array([[p0[0], p0[1]], [pl[0], pl[1]], [pr[0], pr[1]]]), dtype=np.int32)
                        mask = np.zeros_like(binary_image)
                        cv2.fillPoly(mask, [pts], 255)

                        values = binary_image[mask==255]
                        print(type(values))
                        one_cnt = np.sum(values==1)
                        zero_cnt = np.sum(values==0)
                        print(f'one_count: {one_cnt}, zero_count: {zero_cnt}')
                        if one_cnt < zero_cnt:
                            print("Allowed!")

                            tmp_triangle = np.zeros_like(dummy_image)
                            tmp_triangle[:,:,0] = componentMask*255
                            tmp_triangle[:,:,1] = componentMask*255
                            tmp_triangle[:,:,2] = componentMask*255
                            cv2.circle(tmp_triangle, center=(p0[0], p0[1]), radius=2, color=(0,255,0), thickness=1)
                            cv2.circle(tmp_triangle, center=(pl[0], pl[1]), radius=2, color=(0,0,255), thickness=1) # left red
                            cv2.circle(tmp_triangle, center=(pr[0], pr[1]), radius=2, color=(0,0,255), thickness=1) # right blue
                            cv2.fillPoly(tmp_triangle, pts=[pts], color=(255,255,0))
                            if output_dir:
                                cv2.imwrite(f'{component_dir}/triangle_{component_id}_contour_{c_idx}_pt{i}_angle_{avg_angle}.jpg', tmp_triangle)
                            
                            junctions.append(p0)
                            cv2.circle(fused_canvas, (p0[0], p0[1]), 2, (0,0,255), 2)
                            cv2.circle(edge_image, (p0[0], p0[1]), 2, (0,0,255), 2)
                            cv2.circle(composite_fused_canvas, (p0[0], p0[1]), 2, (0,0,255), 2)
            
            
                if output_dir:
                    cv2.imwrite(f'{component_dir}/fused_{component_id}_contour_{c_idx}.jpg', fused_canvas)
                    cv2.imwrite(f'{component_dir}/fused_edges_{component_id}_contour_{c_idx}.jpg', edge_image)
            

            ordered_junction_canvas = np.zeros_like(dummy_image)
            ordered_junction_canvas[:,:,0] = componentMask*255
            ordered_junction_canvas[:,:,1] = componentMask*255
            ordered_junction_canvas[:,:,2] = componentMask*255
            for jni, junction_coords in enumerate(junctions):
                cv2.circle(ordered_junction_canvas, (junction_coords[0], junction_coords[1]), 2, (0, int((1-(jni/len(junctions))*255)), int(((jni/len(junctions))*255))),
                           2)
            if output_dir:
                cv2.imwrite(f'{component_dir}/composite_fused_edges_{component_id}_contour.jpg', composite_fused_canvas)

                cv2.imwrite(f'{component_dir}/ordered_junction_points_{component_id}_contour.jpg', ordered_junction_canvas)

            print(f'Total number of junctions detected: {len(junctions)}')
            
            print(f'Junctions are:\n{junctions}')

            # now let's connect the junctions
            pairs = find_nearest_pairs(junctions)
            print("Pairs")
            print(pairs)
            pair_canvas = binary_image.copy()
            for p in pairs:
                cv2.line(pair_canvas, (p[0][0], p[0][1]), (p[1][0],p[1][1]), 0, 2)
            if output_dir:
                cv2.imwrite(f'{component_dir}/fuse_lines_{component_id}_contour.jpg', pair_canvas*255)


            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volpkg','-v', help="Path to the volpkg", type=str, required=True)
    parser.add_argument('--volume', help="Volume ID", type=str, required=True)
    parser.add_argument('--output-dir', '-o', help="Output directory", type=str)
    parser.add_argument('--slice-name', help="File name of the slice with extension from the volume folder. Example: 0001.tif.", required=True, type=str)
    parser.add_argument('--threshold', '-t', help="A threshold value to threshold the slice image.", type=float, default=2)
    parser.add_argument('--min-connected-points', help="Minimum points needed to qualify as a component.", type=int, required=True)
    parser.add_argument('--connectivity', help="Connectivity for connected components.", type=int, choices=[4,8], default=4)
    parser.add_argument('--gaussian-kernel', help="Kernel size for gaussian blur. For example, if provided 5, the kernel will be (5,5).", type=int, default=9)
    parser.add_argument('--thinning-algorithm', help="Provide the thinning algorithm to be use. 0: Zhang-Suen;1: Guo Hall", type=int, default=0, choices=[0,1])
    parser.add_argument('--num-connected-components', help="Enter the number of points to qualify as a connected component.", type=int, required=True)
    parser.add_argument('--junction-angle', help="Angle less than which will represent a junction.", type=float, required=True)
    parser.add_argument('--junction-window', help="Window size which will be used to detect junction", type=int, required=True)
    parser.add_argument('--junction-window-min', help="minimum offset from which the junction calulation will start. It should be less than window size.", 
                        type=int, default=1)

    args= parser.parse_args()

    volpkg = Path(args.volpkg)
    volume_id = args.volume
    output_dir = args.output_dir
    slice_name = args.slice_name
    threshold = args.threshold
    min_connected_points = args.min_connected_points
    connectivity = args.connectivity
    gaussian_kernel = args.gaussian_kernel
    thinning_algorithm = args.thinning_algorithm
    num_connected_components = args.num_connected_components
    junction_angle = args.junction_angle
    junction_window = args.junction_window
    junction_window_min = args.junction_window_min

    assert junction_window_min < junction_window, "Min junction window should be less than junction window size."


    thinning_algorithm_list = {
        cv2.ximgproc.THINNING_ZHANGSUEN: 'Zhang-Suen',
        cv2.ximgproc.THINNING_GUOHALL: 'Guo-Hall'
    }

    print(f'Thinning algorithm used: {thinning_algorithm_list[thinning_algorithm]}')

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        aux_id = get_date()
        aux_id = f'{aux_id}_{volpkg.stem}_vol_{volume_id}'
        print(f'Auxiliary id is {aux_id}')
        aux_path = output_dir / aux_id
        aux_path.mkdir(exist_ok=True, parents=True)
        print(f'Auxiliary outputs will be saved to {aux_path}')
    else:
        print(f'No auxiliary outputs will be saved')
    segment(volpkg_path=volpkg, volume_id=volume_id, output_dir=aux_path, slice_name=slice_name, threshold=threshold, 
            min_connected_points=min_connected_points, connectivity=connectivity, gaussian_kernel=gaussian_kernel, thinning_algorithm=thinning_algorithm,
            num_connected_components=num_connected_components, junction_angle_threshold=junction_angle, junction_window_size=junction_window,
            junction_window_min=junction_window_min)

if __name__ == "__main__":
    main()