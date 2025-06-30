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

"""
This does remove and try to seperate fused regions of the films.

"""
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
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def segment(volpkg_path: Path, volume_id: str, output_dir: Path, slice_name: str, threshold: int, min_connected_points: int, connectivity:int, gaussian_kernel: int,
            thinning_algorithm: int, num_connected_components: int):
    print(f'Volpkg path: {volpkg_path}, Volume ID: {volume_id}, output-dir: {output_dir}, slice-name: {slice_name}, threshold: {threshold}')
    slice_image_path = Path(f'{volpkg_path}/volumes/{volume_id}/{slice_name}')
    slice_image = cv2.imread(slice_image_path)
    assert slice_image.any(), f'{slice_image_path} not present'
    print(f'Shape of the slice image is {slice_image.shape}')
    dummy_image = slice_image.copy()
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
            fused_canvas = np.zeros_like(dummy_image)
            fused_canvas[:,:,0] = componentMask*255
            fused_canvas[:,:,1] = componentMask*255
            fused_canvas[:,:,2] = componentMask*255
            # junction detection
            binary_image = componentMask.copy()
            # edge_image = np.zeros_like(binary_image)
            # # detect edges (Go with rasterization) XOR operation
            # h, w = binary_image.shape
            # for i in range(h):
            #     for j in range(1, w):
            #         prev = binary_image[i][j-1]
            #         curr = binary_image[i][j]
            #         edge_image[i][j] = prev ^ curr
            # print(f'Completed calculating edges')
            contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour_coordinates = contours[0] # ordered pointset
            boundary_image = np.zeros_like(binary_image)
            cv2.drawContours(boundary_image, contours, contourIdx=-1, color=1, thickness=1)
            if output_dir:
                cv2.imwrite(f'{component_dir}/edges_{component_id}.jpg', boundary_image*255)
            # Let's detect the junction now
            boundary = np.squeeze(contour_coordinates)
            boundary_len = len(boundary)
            window_size = 9
            print(f'Boundary length is: {boundary_len}')
            for i in range(boundary_len):
                left_indices = [(i - j) % boundary_len for j in range(s, 0, -1)]
                left_neighbors = boundary[left_indices]

                right_indices = [(i + j) % boundary_len for j in range(1, s+1)]
                right_neighbors = boundary[right_indices]



            



            

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
            num_connected_components=num_connected_components)

if __name__ == "__main__":
    main()