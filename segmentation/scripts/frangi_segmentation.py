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
from collections import deque
import heapq
from math import sqrt 
import uuid
from itertools import combinations
from sklearn.cluster import KMeans
import json
import datetime as dt
import networkx as nx
from skimage.filters import frangi
import itk

def get_date():
    tz = dt.timezone.utc
    return f'{dt.datetime.now(tz).strftime("%Y%m%d%H%M%S")}'

def write_ordered_vcps(path, pointset):
    file_path = Path(path) / "pointset.vcps"    
    with file_path.open('wt') as file:
        file.writelines([
            f'width: {pointset.shape[1]}\n',
            f'height: {pointset.shape[0]}\n',
            f'dim: {pointset.shape[2]}\n',
            'ordered: true\n',
            'type: double\n',
            'version: 1\n',
            '<>\n'
        ])
    with file_path.open('ab') as file:
        file.write(pointset.tobytes())

def write_metadata(path, vol, uuid_val):
    data = {
        "name": str(uuid_val),
        "type": "seg",
        "uuid": str(uuid_val),
        "vcps": "pointset.vcps",
        "volume": vol
    }
    with open(path / "meta.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def write_vcps(filename, points):
    with open(filename, 'wb') as f:
        f.write(f"width: {points.shape[0]}\n".encode('ascii'))
        f.write(f"height: {1}\n".encode('ascii'))
        f.write(f"dim: {points.shape[1]}\n".encode('ascii'))
        f.write(f"ordered: true\n".encode('ascii'))
        f.write(f"type: double\n".encode('ascii'))
        f.write(f"version: 1\n".encode('ascii'))
        f.write(b"<>\n")
        f.write(points.astype(np.double).tobytes())

def select_n_points(pointset, required_number, trim_val=0):
    if len(pointset) < 2:
        return pointset
    step = (len(pointset) - 1) / (required_number - 1)
    if trim_val and trim_val < int(0.1 * required_number):
        return [pointset[int(round(i * step))] for i in range(required_number)][trim_val:-trim_val]
    else:
        return [pointset[int(round(i * step))] for i in range(required_number)]

# =========================================================================
# HIGHLY OPTIMIZED GRAPH FUNCTIONS (Replaces loops with NumPy Vectorization)
# =========================================================================

def skeleton_to_graph(skeleton):
    """Vectorized graph construction from a binary skeleton."""
    G = nx.Graph()
    ys, xs = np.where(skeleton > 0)
    nodes = list(zip(ys, xs))
    G.add_nodes_from(nodes)
    
    if len(nodes) == 0:
        return G

    # Create a fast lookup map for pixel coordinates
    h, w = skeleton.shape
    pixel_idx = -np.ones((h, w), dtype=np.int32)
    pixel_idx[ys, xs] = np.arange(len(nodes))

    # Shift offsets for 8-connectivity (omitting center 0,0)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    edges = []

    for dy, dx in offsets:
        ny, nx_ = ys + dy, xs + dx
        # Mask valid bounds
        valid = (ny >= 0) & (ny < h) & (nx_ >= 0) & (nx_ < w)
        if not np.any(valid):
            continue
        
        # Check if the shifted pixel exists in the skeleton
        neighbor_indices = pixel_idx[ny[valid], nx_[valid]]
        valid_neighbors = neighbor_indices >= 0
        
        if np.any(valid_neighbors):
            u_indices = np.where(valid)[0][valid_neighbors]
            v_indices = neighbor_indices[valid_neighbors]
            
            # Prevent double adding symmetrical graph edges
            mask = u_indices < v_indices
            for u_idx, v_idx in zip(u_indices[mask], v_indices[mask]):
                edges.append((nodes[u_idx], nodes[v_idx]))

    G.add_edges_from(edges)
    return G

def find_graph_endpoints(G):
    return [n for n, d in G.degree() if d == 1]

def find_graph_junctions(G):
    return [n for n, d in G.degree() if d >= 3]

def skeleton_to_weighted_graph(skeleton, center_point, image, alpha=1.0):
    """Vectorized calculation of structural costs and edge weights."""
    G = nx.Graph()
    ys, xs = np.where(skeleton > 0)
    nodes = list(zip(ys, xs))
    G.add_nodes_from(nodes)
    
    if len(nodes) == 0:
        return G

    h, w = skeleton.shape
    image_norm = (image.astype(np.float32) - image.min()) / (image.max() - image.min() + 1e-8)

    # Precalculate node attributes to save repeated computation
    cy, cx = center_point
    dists = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    penalties = 1.0 - image_norm[ys, xs]

    pixel_idx = -np.ones((h, w), dtype=np.int32)
    pixel_idx[ys, xs] = np.arange(len(nodes))

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    weighted_edges = []

    for dy, dx in offsets:
        ny, nx_ = ys + dy, xs + dx
        valid = (ny >= 0) & (ny < h) & (nx_ >= 0) & (nx_ < w)
        if not np.any(valid):
            continue
        
        neighbor_indices = pixel_idx[ny[valid], nx_[valid]]
        valid_neighbors = neighbor_indices >= 0
        
        if np.any(valid_neighbors):
            u_indices = np.where(valid)[0][valid_neighbors]
            v_indices = neighbor_indices[valid_neighbors]
            
            mask = u_indices < v_indices
            for u_idx, v_idx in zip(u_indices[mask], v_indices[mask]):
                # Compute costs natively via vectors
                avg_dist = (dists[u_idx] + dists[v_idx]) / 2.0
                avg_brightness_penalty = (penalties[u_idx] + penalties[v_idx]) / 2.0
                cost = avg_dist + alpha * avg_brightness_penalty
                weighted_edges.append((nodes[u_idx], nodes[v_idx], cost))

    G.add_weighted_edges_from(weighted_edges)
    return G

def dijkstra_cheapest_path_nx(G, start, end):
    try:
        path = nx.dijkstra_path(G, source=start, target=end, weight='weight')
        cost = nx.dijkstra_path_length(G, source=start, target=end, weight='weight')
        return path, cost
    except nx.NetworkXNoPath:
        return None, np.inf

def remove_skeleton_bridges(G, max_bridge_length=15):
    junctions = {node for node, degree in G.degree() if degree >= 3}
    if not junctions:
        return G
    G_broken = G.copy()
    G_broken.remove_nodes_from(junctions)
    return G_broken

# =========================================================================
# CORE EXECUTION PROCESSING
# =========================================================================

def thin(volpkg_dir: Path, volume: str, film_slice: str, original_image: np.array, clustered: np.array, save_at: str, cluster_id: int, 
         cluster_mask: np.array, total_seg_points: int, threshold_factor: float = 2.0, cv_show: bool=True, cv_wait_key_val: int=0, 
         num_seg_points: int = 1000, gaussian_kernel: int=5, intensity_alpha: float=1, mask_thickness: int=2,
         txt_coord: bool=False, write_vcps: bool=False,
         frangi_sigma_min:int=6, frangi_sigma_max:int=12,
         frangi_sigma_step:int=1, frangi_black_ridges:bool=False, 
         frangi_alpha:float=0.5, frangi_beta:float=0.5, frangi_gamma:float=15, trim_val: int=0, frangi_weight:float = 1.0,
         use_frangi:bool = True):
    
    cy, cx, _ = original_image.shape
    center_point = (cy // 2, cx // 2)

    if gaussian_kernel>0:
        clustered_gaussian = cv2.GaussianBlur(clustered, (gaussian_kernel, gaussian_kernel), 0)
    else:
        clustered_gaussian = clustered
    if use_frangi:
        clustered_frangi = frangi(clustered_gaussian, 
                                sigmas=range(frangi_sigma_min, frangi_sigma_max, frangi_sigma_step),
                                alpha=frangi_alpha, beta=frangi_beta, gamma=frangi_gamma,
                                black_ridges=frangi_black_ridges)
        
    else:
        clustered_frangi = clustered_gaussian


    clustered_frangi = np.clip(clustered_frangi * frangi_weight, 0, 1)
    clustered_frangi = (clustered_frangi * 255).astype(np.uint8)
    frangi_min, frangi_max = clustered_frangi.min(), clustered_frangi.max()

    _, binary = cv2.threshold(clustered_frangi, int((frangi_max - frangi_min) // threshold_factor), int(frangi_max), cv2.THRESH_BINARY)
    # skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    # skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skeleton = cv2.ximgproc.thinning(binary)

    num_labels, labels = cv2.connectedComponents(skeleton, connectivity=8)
    clean_skeleton = np.zeros_like(skeleton)

    for idx in range(1, num_labels):
        component_mask = (labels == idx).astype(np.uint8)
        graph = skeleton_to_graph(skeleton=component_mask)
        graph_broken = remove_skeleton_bridges(graph, max_bridge_length=15)
        
        # Faster canvas assignment via zip unpack
        if graph_broken.number_of_nodes() > 0:
            nodes_array = np.array(graph_broken.nodes())
            clean_skeleton[nodes_array[:, 0], nodes_array[:, 1]] = 255

    if save_at:
        save_at_cluster = Path(save_at) / f'cluster_{cluster_id}'
        save_at_cluster.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f'{save_at_cluster}/skeleton_{gaussian_kernel}.jpg', clean_skeleton)
        cv2.imwrite(f'{save_at_cluster}/original_image.jpg', original_image)
        cv2.imwrite(f'{save_at_cluster}/cluster_mask.jpg', cluster_mask)
        cv2.imwrite(f'{save_at_cluster}/cluster.jpg', clustered)
        cv2.imwrite(f'{save_at_cluster}/binary_{gaussian_kernel}.jpg', binary)
        cv2.imwrite(f'{save_at_cluster}/gaussian_{gaussian_kernel}.jpg', clustered_gaussian)
        cv2.imwrite(f'{save_at_cluster}/frangi.jpg', clustered_frangi)

    binary_segmentation_mask = np.zeros_like(original_image[:, :, 0])
    colored_path = original_image.copy()

    num_clean_labels, clean_labels = cv2.connectedComponents(clean_skeleton, connectivity=8)
    
    for idx in range(1, num_clean_labels):
        component_mask_clean = (clean_labels == idx).astype(np.uint8)
        num_points = cv2.countNonZero(component_mask_clean)
        if num_points < num_seg_points:
            continue

        graph = skeleton_to_graph(skeleton=component_mask_clean)
        deg_1 = find_graph_endpoints(G=graph)

        if len(deg_1) == 0:
            junc = find_graph_junctions(G=graph)
            if len(junc) < 2: continue
            endpoints = junc
        elif len(deg_1) < 2:
            junc = find_graph_junctions(G=graph)
            if not junc: continue
            endpoints = junc + deg_1
        else:
            endpoints = deg_1

        color_overlay = cv2.cvtColor(original_image.copy(), cv2.COLOR_BGR2RGB)
        ys, xs = np.where(component_mask_clean == 1)
        color_overlay[ys, xs] = [0, 255, 255]

        weighted_graph = skeleton_to_weighted_graph(skeleton=component_mask_clean, center_point=center_point, image=clustered, alpha=intensity_alpha)
        
        paths, costs, uvs = [], [], []
        
        # Pathfinding loop
        for u, v in combinations(endpoints, 2):
            pth, cst = dijkstra_cheapest_path_nx(G=weighted_graph, start=u, end=v)
            if pth is not None:
                paths.append(pth)
                costs.append(cst)
                uvs.append((u, v))
            
        if not costs:
            continue

        expensive_idx = np.argmax(np.array(costs))
        shortestpath = paths[expensive_idx]
        start, end = uvs[expensive_idx][0], uvs[expensive_idx][1]

        cv2.circle(color_overlay, (start[1], start[0]), 10, (0, 255, 0), 2)
        cv2.circle(color_overlay, (end[1], end[0]), 10, (0, 0, 255), 2)
        cv2.circle(colored_path, (start[1], start[0]), 10, (0, 255, 0), 2)  
        cv2.circle(colored_path, (end[1], end[0]), 10, (0, 0, 255), 2)  

        # Dynamic mapping to drawing mask
        sp_arr = np.array(shortestpath)
        for pt in shortestpath:
            cv2.circle(binary_segmentation_mask, (pt[1], pt[0]), mask_thickness, 255, mask_thickness)
            
        if total_seg_points > 0:
            shortestpath = select_n_points(shortestpath, total_seg_points, trim_val)
            
        pointset = [[]]
        slice_name = film_slice.stem
        slice_no = int(film_slice.stem) if film_slice.stem.isdigit() else 0
        
        if save_at and txt_coord:
            with open(f'{save_at_cluster}/segmented_component_{idx}_cluster{cluster_id}.txt', 'w') as f:
                f.write(f'x,y\n')
                for spidx, sp in enumerate(shortestpath):
                    g_val = int(255 * (1 - (spidx / len(shortestpath))))
                    r_val = int(255 * spidx / len(shortestpath))
                    cv2.circle(color_overlay, (sp[1], sp[0]), 2, (0, g_val, r_val), 2)
                    cv2.circle(colored_path, (sp[1], sp[0]), 2, (0, g_val, r_val), 2)
                    f.write(f'{sp[1]},{sp[0]}\n')
                    pointset[0].append([float(sp[1]), float(sp[0]), float(slice_no)])
        else:
            for spidx, sp in enumerate(shortestpath):
                g_val = int(255 * (1 - (spidx / len(shortestpath))))
                r_val = int(255 * spidx / len(shortestpath))
                cv2.circle(color_overlay, (sp[1], sp[0]), 2, (0, g_val, r_val), 2)
                cv2.circle(colored_path, (sp[1], sp[0]), 2, (0, g_val, r_val), 2)
                pointset[0].append([float(sp[1]), float(sp[0]), float(slice_no)])

        ps_len = len(pointset[0])
        if ps_len > 0:
            pointset = np.array(pointset)
            seg_id = f'{get_date()}_kmeans_thin_frangi_dijkstra_P{ps_len}_S{slice_name}_a_{frangi_alpha}_b_{frangi_beta}_g_{frangi_gamma}'
            if write_vcps:
                seg_path = volpkg_dir / f'paths/{seg_id}'
                seg_path.mkdir(exist_ok=True, parents=True)
                write_ordered_vcps(path=seg_path, pointset=pointset)
                write_metadata(path=seg_path, uuid_val=seg_id, vol=volume)
        
        cv2.line(color_overlay, (start[1], start[0]), (end[1], end[0]), (125, 255, 255), 5)

        if cv_show:
            cv2.imshow(f"Cluster {cluster_id} | Component {idx} - Skeleton Overlay", color_overlay)
            cv2.waitKey(cv_wait_key_val)

        if save_at and ps_len > 0:
            cv2.imwrite(f'{save_at_cluster}/segmented_{seg_id}_component_{idx}_cluster{cluster_id}.jpg', img=color_overlay)
    
    if save_at:
        cv2.imwrite(f'{save_at_cluster}/total_colored_segmentation_K_{gaussian_kernel}_cluster{cluster_id}.jpg', img=colored_path)
        cv2.imwrite(f'{save_at_cluster}/total_segmentation_mask_K_{gaussian_kernel}cluster{cluster_id}.jpg', img=binary_segmentation_mask)

def kmeans(volpkg_dir: Path, film_slice: str, output_folder: str, volume: str, total_seg_points: int, threshold_factor:float=2.0, cv_show: bool=True, 
           cv_wait_key: bool=False, number_of_clusters: int = 3, num_seg_points: int = 1000, gaussian_kernel: int=5,
           intensity_alpha: float=1, mask_thickness: int=2, txt_coord: bool=False, write_vcps: bool=False,
           frangi_sigma_min:int=6, frangi_sigma_max:int=12,
           frangi_sigma_step:int=1, frangi_black_ridges:bool=False, 
           frangi_alpha:float=0.5, frangi_beta:float=0.5, frangi_gamma:float=15, trim_val: int=0, frangi_weight: float=1.0,
           use_frangi:bool = True):
    
    if output_folder:
        uuid_id = str(uuid.uuid4())
        save_at = Path(output_folder) / uuid_id
        save_at.mkdir(parents=True, exist_ok=True)
        with open(f'{save_at}/details.txt', 'w') as fid:
            fid.write(f'KMEANS->Thinning\nFilm slice: {film_slice}\nthreshold factor: {threshold_factor}\nnumber of clusters: {number_of_clusters}\nmax_num_of_seg_points: {num_seg_points}\n')
    else:
        cv_show = True
        cv_wait_key = True
        save_at = None

    cv_wait_key_val = 0 if cv_wait_key else 1

    original_image = cv2.imread(str(film_slice))
    if original_image is None:
        print("Failed to load original image.")
        return

    img = cv2.imread(str(film_slice), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image.")
        return

    h, w = img.shape
    flat_img = img.reshape(-1, 1)

    kmeans_obj = KMeans(n_clusters=number_of_clusters, random_state=0, n_init="auto")
    kmeans_obj.fit(flat_img)
    labels = kmeans_obj.labels_.reshape(h, w)

    for cluster_id in range(number_of_clusters):
        mask = (labels == cluster_id).astype(np.uint8) * 255
        cluster_img = cv2.bitwise_and(img, img, mask=mask)
        thin(volpkg_dir=volpkg_dir, original_image=original_image, clustered=cluster_img, save_at=save_at, cluster_id=cluster_id, cv_show=cv_show,
             cv_wait_key_val=cv_wait_key_val, num_seg_points=num_seg_points, threshold_factor=threshold_factor, cluster_mask=mask, volume=volume,
             film_slice=film_slice, total_seg_points=total_seg_points, gaussian_kernel=gaussian_kernel, 
             intensity_alpha=intensity_alpha, mask_thickness=mask_thickness, txt_coord=txt_coord, write_vcps=write_vcps,
             frangi_sigma_min=frangi_sigma_min, frangi_sigma_max=frangi_sigma_max,
             frangi_sigma_step=frangi_sigma_step, frangi_black_ridges=frangi_black_ridges, 
             frangi_alpha=frangi_alpha, frangi_beta=frangi_beta, frangi_gamma=frangi_gamma,
             trim_val=trim_val, frangi_weight=frangi_weight, use_frangi=use_frangi)
    
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volpkg', help="Path to the volpkg.", type=str)
    parser.add_argument('--volume', help="Volume ID inside the volpkg.", type=str)
    parser.add_argument('--slice-name', help="Name of the slice to segment. DEFAULT: 0000.tif", type=str, default='0000.tif')
    parser.add_argument('--threshold-factor', '-t', type=float, default=2.0)
    parser.add_argument('--number-of-clusters', '-k', type=int, default=3)
    parser.add_argument('--num-seg-points', '-n', type=int, default=1000)
    parser.add_argument('--total-seg-points', '-s', type=int)
    parser.add_argument('--output-folder','-o', type=str)
    parser.add_argument('--gaussian-kernel', type=int, default=5)
    parser.add_argument('--intensity-alpha', type=float, default=1.0)
    parser.add_argument('--txt-coord', action='store_true')
    parser.add_argument('--mask-thickness', type=int, default=2)
    parser.add_argument('--frangi-sigma-min', type=int, default=6)
    parser.add_argument('--frangi-sigma-max', type=int, default=12)
    parser.add_argument('--frangi-sigma-step', type=int, default=4)
    parser.add_argument('--frangi-black-ridges', action='store_true')
    parser.add_argument('--frangi-alpha', type=float, default=0.5)
    parser.add_argument('--frangi-beta', type=float, default=0.5)
    parser.add_argument('--frangi-gamma', type=float, default=15)
    parser.add_argument('--frangi-weight', type=float, default=1.0)
    parser.add_argument('--use-frangi', action='store_true')
    parser.add_argument('--trim-val', type=int, default=0)
    parser.add_argument('--write-vcps', action='store_true')
    parser.add_argument('--cv-wait-key', action='store_true')
    parser.add_argument('--cv-show', action='store_true')
    args = parser.parse_args()

    volpkg = Path(args.volpkg)
    volume = args.volume
    film_slice = volpkg / f'volumes/{volume}/{args.slice_name}'
    if not film_slice.exists():
        print(f'{film_slice} does not exist!')
        return

    if args.number_of_clusters == 0:
        print("Number of clusters cannot be zero.")
        return
    
    print(f'USE FRANGI: {args.use_frangi}')

    kmeans(volpkg_dir=volpkg, film_slice=film_slice, number_of_clusters=args.number_of_clusters, num_seg_points=args.num_seg_points, 
           output_folder=args.output_folder, cv_wait_key=args.cv_wait_key, cv_show=args.cv_show, threshold_factor=args.threshold_factor, 
           volume=volume, total_seg_points=args.total_seg_points, intensity_alpha=args.intensity_alpha, gaussian_kernel=args.gaussian_kernel, 
           mask_thickness=args.mask_thickness, txt_coord=args.txt_coord, write_vcps=args.write_vcps, frangi_sigma_min=args.frangi_sigma_min, 
           frangi_sigma_max=args.frangi_sigma_max, frangi_sigma_step=args.frangi_sigma_step, frangi_black_ridges=args.frangi_black_ridges, 
           frangi_alpha=args.frangi_alpha, frangi_beta=args.frangi_beta, frangi_gamma=args.frangi_gamma, trim_val=args.trim_val, 
           frangi_weight=args.frangi_weight, use_frangi=args.use_frangi)

if __name__ == "__main__":
    main()