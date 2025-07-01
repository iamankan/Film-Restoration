import argparse
from pathlib import Path
import uuid
import cv2
import numpy as np
import skan as csr
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.cluster import DBSCAN
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
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def find_index_in_contour(p0, contour):
    distances = np.linalg.norm(contour - p0, axis=1)
    return np.argmin(distances)

def get_junction_sections_from_triplets(junction_triplets, eps=10, min_samples=1):
    if not junction_triplets:
        return []

    p0_points = np.array([triplet[0] for triplet in junction_triplets])

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(p0_points)
    labels = db.labels_
    unique_labels = set(labels)

    clustered_triplets = []

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_triplets = [junction_triplets[i] for i in cluster_indices]
        clustered_triplets.append(cluster_triplets)

    return clustered_triplets

def draw_cut_from_single_point(p0, contour, mask, cut_length=5, cut_thickness=1):
    contour = np.squeeze(contour)
    contour = contour.astype(np.float32)
    p0 = np.asarray(p0, dtype=np.float32)
    n = len(contour)
    
    matches = np.where(np.all(np.isclose(contour, p0, atol=1), axis=1))[0]
    if len(matches) == 0:
        return f"p0 {p0} not found in contour."
    i = matches[0]

    # Estimate tangent
    p_prev = contour[(i - 1) % n]
    p_next = contour[(i + 1) % n]
    tangent = p_next - p_prev
    tangent = tangent / (np.linalg.norm(tangent) + 1e-12)

    if tangent.shape != (2,):
        raise ValueError(f"Tangent is malformed: {tangent}")

    normal = np.array([-tangent[1], tangent[0]])

    p1 = np.round(p0 + cut_length * normal).astype(int) # p=p+nt
    p2 = np.round(p0 - cut_length * normal).astype(int) # p=p+nt
    
    cv2.line(mask, tuple(p1), tuple(p2), color=0, thickness=cut_thickness)

def normal_vector(v):
    v = v / (np.linalg.norm(v) + 1e-12)
    return np.array([-v[1], v[0]])



def extend_until_background_from_point(p0, direction, binary_mask, max_len=15):
    h, w = binary_mask.shape
    for i in range(1, max_len + 1):
        step = np.round(p0 + i * direction).astype(int)
        x, y = step
        if 0 <= x < w and 0 <= y < h:
            if binary_mask[y, x] == 0:
                return tuple(step)
        else:
            break
    return tuple(step)

def get_tangent(contour, i):
    prev = contour[(i - 1) % len(contour)]
    next = contour[(i + 1) % len(contour)]
    tangent = next - prev
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
    return tangent

def get_smoothed_tangent(contour, i, window=3):
    n = len(contour)
    prev = contour[(i - window) % n]
    nxt = contour[(i + window) % n]
    tangent = nxt - prev
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
    return tangent

def get_pca_normal(contour, i, window=5):
    n = len(contour)
    idxs = [(i + offset) % n for offset in range(-window, window + 1)]
    pts = np.array([contour[idx] for idx in idxs])

    # Center the points
    pts_mean = pts.mean(axis=0)
    pts_centered = pts - pts_mean

    # PCA
    U, S, Vt = np.linalg.svd(pts_centered)
    tangent = Vt[0]
    normal = Vt[1]  # perpendicular to tangent
    return normal / (np.linalg.norm(normal) + 1e-8)

def get_pca_normal_from_contour(contour, p0, window=5):
    
    contour = np.asarray(contour)
    i = np.where((contour == p0).all(axis=1))[0]
    if len(i) == 0:
        return None  # p0 not found in contour
    i = i[0]

    n = len(contour)
    idxs = [(i + offset) % n for offset in range(-window, window + 1)]
    pts = np.array([contour[idx] for idx in idxs])

    pts_mean = pts.mean(axis=0)
    pts_centered = pts - pts_mean

    _, _, Vt = np.linalg.svd(pts_centered)
    normal = Vt[1]  # normal is perpendicular to tangent (which is Vt[0])
    return normal / (np.linalg.norm(normal) + 1e-8)

def extend_until_black(p0, normal, binary_img, init_length=5, max_length=100):
    h, w = binary_img.shape
    p0 = np.array(p0, dtype=np.float32)

    def trace(direction):
        for d in range(init_length, max_length):
            candidate = p0 + direction * d
            x, y = int(round(candidate[0])), int(round(candidate[1]))
            if not (0 <= x < w and 0 <= y < h):
                break
            if binary_img[y, x] == 0:  # hit black
                return (int(round(candidate[0])), int(round(candidate[1])))
        return (int(round(candidate[0])), int(round(candidate[1])))

    forward = trace(normal)
    backward = trace(-normal)
    return backward, forward

def sort_junction_triplets_by_contour(junction_triplets, contour):
    contour = np.squeeze(contour)

    ordered = []
    for triple in junction_triplets:
        p0 = tuple(triple[0])
        # Find p0 in contour
        indices = np.where(np.all(contour == p0, axis=1))[0]
        if len(indices) > 0:
            idx = indices[0]
            ordered.append((idx, triple))
        else:
            print(f"Warning: p0 {p0} not found in contour")

    # Sort by p0's index in the contour
    ordered.sort(key=lambda x: x[0])
    return [triple for _, triple in ordered]

def segment(volpkg_path: Path, volume_id: str, output_dir: Path, slice_name: str, threshold: int, min_connected_points: int, connectivity:int, 
            gaussian_kernel: int,
            thinning_algorithm: int, num_connected_components: int,
            junction_angle_threshold: float, junction_window_size: int, junction_window_min: int,
            skip: int, fuse_sensitivity: float):
    print(f'Volpkg path: {volpkg_path}, Volume ID: {volume_id}, output-dir: {output_dir}, slice-name: {slice_name}, threshold: {threshold}')
    slice_image_path = Path(f'{volpkg_path}/volumes/{volume_id}/{slice_name}')
    slice_image = cv2.imread(slice_image_path)
    assert slice_image.any(), f'{slice_image_path} not present'
    print(f'Shape of the slice image is {slice_image.shape}')
    dummy_image = slice_image.copy()
    window_size = junction_window_size
    angle_threshold = junction_angle_threshold

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
    junction_triplets = []
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

            all_junction_canvas = tmp_img.copy()

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
                tmp_jtrip = []
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
                for i in range(0, n, skip):
                    p0 = contour_ordered_coordinates[i]
                    angles = []
                    for offset in range(junction_window_min, window_size + 1):
                        pl = contour_ordered_coordinates[(i - offset) % n]
                        pr = contour_ordered_coordinates[(i + offset) % n]

                        # p0pl = (pl - p0).astype(float)
                        # p0pr = (pr - p0).astype(float)
                        p0pl = pl - p0
                        p0pr = pr - p0

                        angle = angle_between_vectors_in_degrees(p0pl, p0pr)
                        angles.append(angle)

                    avg_angle = np.mean(angles)
                    pl = contour_ordered_coordinates[(i - window_size) % n]
                    pr = contour_ordered_coordinates[(i + window_size) % n]

                    pts = np.array(np.array([[p0[0], p0[1]], [pl[0], pl[1]], [pr[0], pr[1]]]), dtype=np.int32)
                    mask = np.zeros_like(binary_image)
                    cv2.fillPoly(mask, [pts], 255)

                    values = binary_image[mask==255]

                    # tmp_triangle = np.zeros_like(dummy_image)
                    # tmp_triangle[:,:,0] = componentMask*255
                    # tmp_triangle[:,:,1] = componentMask*255
                    # tmp_triangle[:,:,2] = componentMask*255
                    # cv2.circle(tmp_triangle, center=(p0[0], p0[1]), radius=2, color=(0,255,0), thickness=1)
                    # cv2.circle(tmp_triangle, center=(pl[0], pl[1]), radius=2, color=(0,0,255), thickness=1) # left red
                    # cv2.circle(tmp_triangle, center=(pr[0], pr[1]), radius=2, color=(0,0,255), thickness=1) # right blue
                    # cv2.fillPoly(tmp_triangle, pts=[pts], color=(255,255,0))
                    # if output_dir:
                    #     triangle_path = component_dir / "traingle"
                    #     triangle_path.mkdir(exist_ok=True, parents=True)
                    #     cv2.imwrite(f'{triangle_path}/triangle_{component_id}_contour_{c_idx}_pt{i}_angle_{avg_angle}.jpg', tmp_triangle)
                    

                    if avg_angle < angle_threshold:
                        one_cnt = np.sum(values==1)
                        zero_cnt = np.sum(values==0)
                        if one_cnt < zero_cnt:
                            print(f"Allowed! Angle: {avg_angle} degrees")
                            tmp_jtrip.append([p0, pl, pr])
                            cv2.circle(fused_canvas, (p0[0], p0[1]), 2, (0,0,255), 2)
                            cv2.circle(edge_image, (p0[0], p0[1]), 2, (0,0,255), 2)
                            cv2.circle(composite_fused_canvas, (p0[0], p0[1]), 1, (0,0,255), 1)
                tmp_jtrip = sort_junction_triplets_by_contour(junction_triplets=tmp_jtrip, contour=contour)
                junction_triplets = junction_triplets + tmp_jtrip
                
                if output_dir:
                    cv2.imwrite(f'{component_dir}/fused_{component_id}_contour_{c_idx}.jpg', fused_canvas)
                    cv2.imwrite(f'{component_dir}/fused_edges_{component_id}_contour_{c_idx}.jpg', edge_image)
                    cv2.imwrite(f'{component_dir}/composite_fused_edges_{component_id}_contour_{c_idx}.jpg', composite_fused_canvas)
            

            print(f'Total number of junctions detected: {len(junction_triplets)}')
            
            print(f'Junctions are:\n{junction_triplets}')

            # now let's connect the junctions
            # pairs = find_nearest_pairs(junctions)
            # print("Pairs")
            # print(pairs)
            # pair_canvas = binary_image.copy()
            # for p in pairs:
            #     cv2.line(pair_canvas, (p[0][0], p[0][1]), (p[1][0],p[1][1]), 0, 2)
            # if output_dir:
            #     cv2.imwrite(f'{component_dir}/fuse_lines_{component_id}_contour.jpg', pair_canvas*255)

            jtrip_sections = get_junction_sections_from_triplets(junction_triplets=junction_triplets)
            print("J-sections:")

            junction_triplets_middle = []

            for section in jtrip_sections:
                p0s = [np.array(triplet[0]) for triplet in section]
                print(f'p0s-len: {len(p0s)}')
                # center = np.mean(np.array(p0s), axis=0)

                # dists = [np.linalg.norm(p0 - center) for p0 in p0s]
                # min_idx = np.argmin(dists)
                # representative_triplet = section[min_idx]  # (p0, pl, pr)
                # representative_triplet = section[len(p0s)//2]
                representative_triplet = section[-1]

                junction_triplets_middle.append(representative_triplet)
            
            print(f'Middle junction taken after DBSCAN.')
            
            binary_mask = binary_image.copy()
            partitioning_binary = binary_image.copy()
            plpr_bin = binary_image.copy()
            
            triplet_canvas = np.zeros_like(dummy_image)
            triplet_canvas[:,:,0] = componentMask*255
            triplet_canvas[:,:,1] = componentMask*255
            triplet_canvas[:,:,2] = componentMask*255
            pre_breakup_canvas = triplet_canvas.copy()

            for jtrip in junction_triplets_middle:
                print(jtrip)
                p0 = np.array(jtrip[0])
                pl = np.array(jtrip[1])
                pr = np.array(jtrip[2])
                pl = tuple(map(int, pl))
                pr = tuple(map(int, pr))
                cv2.circle(pre_breakup_canvas, p0, 2, (0,0,255), 2)
                cv2.line(plpr_bin, pl, pr, 0, 2)
                for contour in contours:
                    if np.any(np.all(contour.squeeze() == p0, axis=1)):
                        normal = get_pca_normal_from_contour(contour.squeeze(), p0, window=5)
                        if normal is not None:
                            start, end = extend_until_black(p0, normal, binary_mask, init_length=5, max_length=100)
                            cv2.line(triplet_canvas, start, end, (0, 0, 255), 1)
                            cv2.line(partitioning_binary, start, end, 0, 2)

            
            if output_dir:
                cv2.imwrite(f'{component_dir}/triplet_junction_{component_id}.jpg', triplet_canvas)
                cv2.imwrite(f'{component_dir}/partitioned_binary_{component_id}.jpg', partitioning_binary*255)
                cv2.imwrite(f'{component_dir}/junctions_{component_id}.jpg', pre_breakup_canvas)
                cv2.imwrite(f'{component_dir}/plpr_{component_id}.jpg', plpr_bin*255)
            

            
            fuse_component_path = component_dir / 'fused_components'
            fuse_component_path.mkdir(exist_ok=True, parents=True)
            fuse_num_labels, fuse_labels_im, fuse_stats, fuse_centroids = cv2.connectedComponentsWithStats(image=partitioning_binary, 
                                                                                                       connectivity=connectivity)
            # print(f'fuse_labels_im: {fuse_labels_im}')
            all_junction_points = [(int(p0[0]), int(p0[1])) for p0, _, _ in junction_triplets]
            
            new_mask = []
            
            for i in range(1, fuse_num_labels): # 0 is the background, always. So, starting from 1
                
                new_componentMask = (fuse_labels_im == i).astype("uint8")
                new_mask.append(new_componentMask)
                nx, ny = np.where(new_componentMask==1)
                # match_list = [(int(f[0]), int(f[1])) in all_junction_points for f in zip(nx, ny)]
                match_list = [(y, x) in all_junction_points for y, x in zip(ny, nx)]
                number_of_trues = sum(match_list)
                frac = number_of_trues/len(match_list)
                if frac > fuse_sensitivity:
                    print(f'Trues:: {sum(match_list)}, total: {len(match_list)}, frac(/1): {sum(match_list)/len(match_list)}')
                    print(f'Fuse component detected!')
                    fused_mask=new_componentMask
                    if output_dir:
                        cv2.imwrite(f'{fuse_component_path}/fuse_component_{i}_main_component_{component_id}.jpg', fused_mask*255)

                if output_dir:
                    cv2.imwrite(f'{fuse_component_path}/new_component{i}_main_component_{component_id}.jpg', new_componentMask*255)
            
                    
                

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
    parser.add_argument('--skip',help="Number of pixels to skip while traversing the contour.", type=int, default=1)
    parser.add_argument('--fuse-sensitivity', help="Enter a number above which the fuse is detectable.", type=float, default=0.001)

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
    skip = args.skip
    fuse_sensitivity = args.fuse_sensitivity

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
            junction_window_min=junction_window_min, skip=skip, fuse_sensitivity=fuse_sensitivity)

if __name__ == "__main__":
    main()