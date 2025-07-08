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
# matplotlib.use('TkAgg')
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


################USING GRAPH Networkx######################
def skeleton_to_graph(skeleton):
    G = nx.Graph()
    h, w = skeleton.shape
    for y in range(h):
        for x in range(w):
            if skeleton[y, x]:
                # Add current pixel as a node
                G.add_node((y, x))

                # Check 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx_ = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_]:
                            G.add_edge((y, x), (ny, nx_))
    return G

def find_graph_endpoints(G):
    return [n for n in G.nodes if G.degree[n] == 1] # list of (y,x)

def find_graph_junctions(G):
    return [n for n in G.nodes if G.degree[n] >= 3] # list of (y,x)

def find_graph_Tpoints(G):
    return [n for n in G.nodes if G.degree[n] == 3] # list of (y,x)

def skeleton_to_weighted_graph(skeleton, center_point, image, alpha=1.0):
    G = nx.Graph()
    h, w = skeleton.shape
    image = image.astype(np.float32)
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    print(f'Image normalized: {image_norm.min(), image_norm.max()}')

    for y in range(h):
        for x in range(w):
            if skeleton[y, x]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx_ = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_]:
                            # Distance to center
                            dist1 = np.linalg.norm(np.array([y, x]) - center_point)
                            dist2 = np.linalg.norm(np.array([ny, nx_]) - center_point)
                            avg_dist = (dist1 + dist2) / 2

                            # Brightness penalty
                            b1 = image_norm[y, x]
                            b2 = image_norm[ny, nx_]
                            avg_brightness = (b1 + b2) / 2
                            brightness_penalty = 1.0 - avg_brightness

                            # Total cost
                            cost = avg_dist + alpha * brightness_penalty
                            G.add_edge((y, x), (ny, nx_), weight=cost)
    return G

def dijkstra_cheapest_path_nx(G, start, end):
    try:
        path = nx.dijkstra_path(G, source=start, target=end, weight='weight')
        cost = nx.dijkstra_path_length(G, source=start, target=end, weight='weight')
        return path, cost
    except nx.NetworkXNoPath:
        return None, np.inf


'''
python3 segmentation/scripts/kmeans.py -s /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26855.volpkg/volumes/20250214115505/1000.tif -k 3 -n 100
'''

def select_n_points(pointset, required_number, trim_val=0):
    step = (len(pointset) - 1) / (required_number - 1)
    if trim_val and trim_val<int(0.1*required_number): # number of trim points have to be atmost 10% of the required points
        print(f'Trimming {trim_val} points from the ends')
        return [pointset[int(round(i * step))] for i in range(required_number)][trim_val:-trim_val]
    else:
        return [pointset[int(round(i * step))] for i in range(required_number)]


def thin(volpkg_dir: Path, volume: str, film_slice: str, original_image: np.array, clustered: np.array, save_at: str, cluster_id: int, 
         cluster_mask: np.array, total_seg_points: int, threshold_factor: float = 2.0, cv_show: bool=True, cv_wait_key_val: int=0, 
         num_seg_points: int = 1000, gaussian_kernel: int=5, intensity_alpha: float=1, mask_thickness: int=2,
         txt_coord: bool=False, write_vcps: bool=False,
         frangi_sigma_min:int=6, frangi_sigma_max:int=12,
           frangi_sigma_step:int=1, frangi_black_ridges:bool=False, 
           frangi_alpha:float=0.5, frangi_beta:float=0.5, frangi_gamma:float=15,
           trim_val:int =0):
    
    print(f'Shape of the film slice is: {original_image.shape}')
    img_min = original_image[:, :, 0].min()
    img_max = original_image[:, :, 0].max()
    print(f'min: {img_min}, max: {img_max}')
    print(f'Clustered_min: {clustered.min()}, clustered_max: {clustered.max()}')

    



    cy, cx, _ = original_image.shape

    center_point = (cy//2, cx//2)

    clustered_gaussian = cv2.GaussianBlur(clustered, (gaussian_kernel, gaussian_kernel), 0)

    clustered_frangi = frangi(clustered_gaussian, 
                              sigmas=range(frangi_sigma_min,frangi_sigma_max,frangi_sigma_step),
                              alpha=frangi_alpha,
                              beta=frangi_beta, # should help ignore the mounts
                              gamma=frangi_gamma, # ignore low contrast as much as possible
                              black_ridges=frangi_black_ridges # films are bright
                              ) # gets 0-1
    

    print(f'First: Frangi: min: {clustered_frangi.min()}, max: {clustered_frangi.max()}')
    clustered_frangi = np.clip(clustered_frangi, 0, 1) # Just to ensure. IT returns 0-1, but for precaution.
    clustered_frangi = clustered_frangi*255 # This is float
    clustered_frangi = clustered_frangi.astype(np.uint8) # Convert type to int.

    # print(f'Type gaussian: {clustered_gaussian.dtype}. Type frangi: {clustered_frangi.dtype}')

    print(f'Frangi: min: {clustered_frangi.min()}, max: {clustered_frangi.max()}')

    print("Frangi filter parameters (inside thin):")
    print(f"  sigma_min      : {frangi_sigma_min}")
    print(f"  sigma_max      : {frangi_sigma_max}")
    print(f"  sigma_step     : {frangi_sigma_step}")
    print(f"  black_ridges   : {frangi_black_ridges}")
    print(f"  alpha          : {frangi_alpha}")
    print(f"  beta           : {frangi_beta}")
    print(f"  gamma          : {frangi_gamma}")

    # print(f"Performing thresholding on blurred clustered masked image. GaussianBlur is used with kernel ({gaussian_kernel}, {gaussian_kernel})")
    
    # _, binary = cv2.threshold(clustered_gaussian, (img_max - img_min) // threshold_factor, img_max, cv2.THRESH_BINARY)
    _, binary = cv2.threshold(clustered_frangi, (img_max - img_min) // threshold_factor, img_max, cv2.THRESH_BINARY)
    
    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)

    binary_segmentation_mask = np.zeros_like(original_image[:, :, 0])
        

    num_labels, labels = cv2.connectedComponents(skeleton, connectivity=8)
    # print(f"Total components: {num_labels} (including background).")
    if save_at:
        save_at_cluster = Path(save_at) / f'cluster_{cluster_id}'
        save_at_cluster.mkdir(parents=True, exist_ok=True)
        colored_path = original_image.copy()
        cv2.imwrite(f'{save_at_cluster}/original_image.jpg', original_image)
        cv2.imwrite(f'{save_at_cluster}/cluster_mask.jpg', cluster_mask)
        cv2.imwrite(f'{save_at_cluster}/cluster.jpg', clustered)
        cv2.imwrite(f'{save_at_cluster}/skeleton_{gaussian_kernel}.jpg', skeleton)
        cv2.imwrite(f'{save_at_cluster}/binary_{gaussian_kernel}.jpg', binary)
        cv2.imwrite(f'{save_at_cluster}/gaussian_{gaussian_kernel}.jpg', clustered_gaussian)
        cv2.imwrite(f'{save_at_cluster}/frangi_on_gaussian.jpg', clustered_frangi)
    

    for idx in range(1, num_labels):  # skip background
        component_mask = (labels == idx).astype(np.uint8)

        graph = skeleton_to_graph(skeleton=component_mask)

        
        num_points = cv2.countNonZero(component_mask)

        if num_points < num_seg_points:
            continue


        deg_1 = find_graph_endpoints(G=graph)

        if len(deg_1) == 0:
            junc = find_graph_junctions(G=graph)
            if len(junc) == 0 or len(junc) < 2:
                return
            endpoints = junc
        elif len(deg_1) < 2:
            junc = find_graph_junctions(G=graph)
            if len(junc) ==0:
                return
            endpoints = junc+deg_1
        else:
            endpoints=deg_1

        print(f'Numbers of endpoints (deg=1) = {len(deg_1)}')

        print(f'Total source+destinations to check: {len(endpoints)}')

        color_overlay = cv2.cvtColor(original_image.copy(), cv2.COLOR_BGR2RGB)

        ys, xs = np.where(component_mask == 1)
        for (y, x) in zip(ys, xs):
            color_overlay[y, x] = [0, 255, 255]

        weighted_graph = skeleton_to_weighted_graph(skeleton=component_mask, center_point=center_point, image=clustered, 
                                                    alpha=intensity_alpha)
        paths=[]
        costs=[]
        uvs = []
        for u, v in combinations(endpoints, 2):
            pth, cst = dijkstra_cheapest_path_nx(G=weighted_graph, start=u, end=v)
            paths.append(pth)
            costs.append(cst)
            uvs.append((u,v))
        expensive_idx = np.argmax(np.array([costs]))
        shortestpath = paths[expensive_idx]
        shortestdist = costs[expensive_idx]
        start, end = uvs[expensive_idx][0], uvs[expensive_idx][1]

        cv2.circle(color_overlay, (start[1], start[0]), 10, (0,255,0),2) # start - Green
        cv2.circle(color_overlay, (end[1], end[0]), 10, (0,0,255),2) # end - Red

        cv2.circle(colored_path, (start[1], start[0]), 10, (0,255,0),2) # start - Green
        cv2.circle(colored_path, (end[1], end[0]), 10, (0,0,255),2) # end - Red

        # print(f'Length of the shortest path between start and end is: {len(shortestpath)} pixels, and cost is {shortestdist}.')
        if total_seg_points:
            # print(f'Saving the segmentation mask binary image.')
            # binary_segmentation_mask
            for sp in shortestpath:
                cv2.circle(binary_segmentation_mask, (sp[1], sp[0]), mask_thickness, 255,
                           mask_thickness)
            # print(f'Making the total-seg-points from {len(shortestpath)} to {total_seg_points}')
            shortestpath = select_n_points(shortestpath, total_seg_points,trim_val)
            # print(f'Now the total points are {len(shortestpath)}')
        


        pointset = [[]]
        slice_name = film_slice.stem
        slice_no = int(film_slice.stem)
        if save_at and txt_coord:
            with open(f'{save_at_cluster}/segmented_component_{idx}_cluster{cluster_id}.txt', 'w') as f:
                f.write(f'x,y\n')
                for spidx, sp in enumerate(shortestpath):
                    cv2.circle(color_overlay, (sp[1], sp[0]), 2, (0,int(255*(1-(spidx/len(shortestpath)))),int(255*spidx/len(shortestpath))),2)
                    cv2.circle(colored_path, (sp[1], sp[0]), 2, (0,int(255*(1-(spidx/len(shortestpath)))),int(255*spidx/len(shortestpath))),2)
                    f.write(f'{sp[1]},{sp[0]}\n')
                    pointset[0].append([float(sp[1]), float(sp[0]), float(slice_no)])
                # f.write(f'Cost: {shortestdist}')
        else:
            for spidx, sp in enumerate(shortestpath):
                cv2.circle(color_overlay, (sp[1], sp[0]), 2, (0,int(255*(1-(spidx/len(shortestpath)))),int(255*spidx/len(shortestpath))),2)
                cv2.circle(colored_path, (sp[1], sp[0]), 2, (0,int(255*(1-(spidx/len(shortestpath)))),int(255*spidx/len(shortestpath))),2)
                pointset[0].append([float(sp[1]), float(sp[0]), float(slice_no)])
        # print(f'Length of pointset[0]: {len(pointset[0])}')
        ps_len = len(pointset[0])
        
        if len(pointset[0])>0:
            pointset = np.array(pointset)
            seg_id = f'{get_date()}_kmeans_thin_frangi_gaussian_dijkstraP{ps_len}_S{slice_name}_k_{gaussian_kernel}_a_{frangi_alpha}_b_{frangi_beta}_g_{frangi_gamma}'
            if write_vcps:
                seg_path = volpkg_dir / f'paths/{seg_id}'
                seg_path.mkdir(exist_ok=True, parents=True)
                print(f'Generated segmentation id: {seg_id} and the path is {seg_path}')
                write_ordered_vcps(path=seg_path, pointset=pointset)
                write_metadata(path=seg_path, uuid_val=seg_id, vol=volume)
                print('Finished writing meta.json and pointset.vcps')
        

        # colored_path = cv2.bitwise_or(colored_path, color_overlay)
        

        cv2.line(color_overlay, (start[1], start[0]), (end[1], end[0]), (125,255,255), 5)

        if cv_show:
            cv2.imshow(f"Cluster {cluster_id} | Component {idx} - Skeleton Overlay", color_overlay)
            cv2.waitKey(cv_wait_key_val)

        if save_at:
            if len(pointset[0])>0:
                cv2.imwrite(f'{save_at_cluster}/segmented_{seg_id}_component_{idx}_cluster{cluster_id}.jpg', img=color_overlay)
    
    if save_at:
        print(f"Writing binary and total segmentation colored files for cluster {cluster_id}.")
        # cv2.imwrite(f'{save_at_cluster}/binary_cluster{cluster_id}.jpg', img=binary)
        cv2.imwrite(f'{save_at_cluster}/total_colored_segmentation_K_{gaussian_kernel}_cluster{cluster_id}.jpg', img=colored_path)
        cv2.imwrite(f'{save_at_cluster}/total_segmentation_mask_K_{gaussian_kernel}cluster{cluster_id}.jpg', img=binary_segmentation_mask)


    
    

def kmeans(volpkg_dir: Path, film_slice: str, output_folder: str, volume: str, total_seg_points: int, threshold_factor:float=2.0, cv_show: bool=True, 
           cv_wait_key: bool=False, number_of_clusters: int = 3, num_seg_points: int = 1000, gaussian_kernel: int=5,
           intensity_alpha: float=1, mask_thickness: int=2, txt_coord: bool=False, write_vcps: bool=False,
           frangi_sigma_min:int=6, frangi_sigma_max:int=12,
           frangi_sigma_step:int=1, frangi_black_ridges:bool=False, 
           frangi_alpha:float=0.5, frangi_beta:float=0.5, frangi_gamma:float=15, trim_val:int=0):
    
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
    for cluster_id in range(0, number_of_clusters):
        mask = (labels == cluster_id).astype(np.uint8) * 255  # Binary mask
        cluster_img = cv2.bitwise_and(img, img, mask=mask)
        print(f'cluster_img shape: {cluster_img.shape}')

    for cluster_id in range(0, number_of_clusters):
        mask = (labels == cluster_id).astype(np.uint8) * 255  # Binary mask
        cluster_img = cv2.bitwise_and(img, img, mask=mask)
        print(f"Starting thinning for cluster {cluster_id}")
        thin(volpkg_dir=volpkg_dir, original_image=original_image, clustered=cluster_img, save_at=save_at, cluster_id=cluster_id, cv_show=cv_show,
             cv_wait_key_val=cv_wait_key_val, num_seg_points=num_seg_points, threshold_factor=threshold_factor, cluster_mask=mask, volume=volume,
             film_slice=film_slice, total_seg_points=total_seg_points, gaussian_kernel=gaussian_kernel, 
             intensity_alpha=intensity_alpha, mask_thickness=mask_thickness, txt_coord=txt_coord, write_vcps=write_vcps,
             frangi_sigma_min=frangi_sigma_min, frangi_sigma_max=frangi_sigma_max,
           frangi_sigma_step=frangi_sigma_step, frangi_black_ridges=frangi_black_ridges, 
           frangi_alpha=frangi_alpha, frangi_beta=frangi_beta, frangi_gamma=frangi_gamma,
           trim_val=trim_val)
    
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
    parser.add_argument('--total-seg-points', '-s', help="Total number of segmentation points.", type=int)
    parser.add_argument('--output-folder','-o', help="Output folder where the images will be saved. This should be outside volpkg. Default: Nothing will be saved", type=str)
    parser.add_argument('--gaussian-kernel',help="Enter the kernel height. It will be treated as nxn.", type=int, default=5)
    parser.add_argument('--intensity-alpha',help="Enter weight for intensity importance.", type=float, default=1)
    parser.add_argument('--txt-coord', help="Use this flag to turn on writing coors to a txt file. (Turned OFF by default.)", action='store_true')
    parser.add_argument('--mask-thickness', help="Thickness of the binary mask to be drawn", type=int, default=2)

    parser.add_argument('--frangi-sigma-min', help="Enter the minimum sigma for frangi filtering.", type=int, default=6)
    parser.add_argument('--frangi-sigma-max', help="Enter the maximum sigma for frangi filtering.", type=int, default=12)
    parser.add_argument('--frangi-sigma-step', help="Enter the step for frangi filtering by which sigma should increase.", type=int, default=4)
    parser.add_argument('--frangi-black-ridges', help="Use this flag if black ridges are considered. \
                        It is only necessary if spockets are present.", action='store_true')
    parser.add_argument('--frangi-alpha', help="Enter alpha value for frangi.", type=float, default=0.5)
    parser.add_argument('--frangi-beta', help="Enter beta value for frangi.", type=float, default=0.5)
    parser.add_argument('--frangi-gamma', help="Enter gamma value for frangi.", type=float, default=15)

    parser.add_argument('--trim-val', help="Enter the number of points you want to trim from the final segmentation, \
                        such that you don't encounter any bad meshing.", type=int, default=0)

    parser.add_argument('--write-vcps',help="Enter to write vcps.", action='store_true')
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
    total_seg_points = args.total_seg_points
    intensity_alpha = args.intensity_alpha
    gaussian_kernel = args.gaussian_kernel
    mask_thickness = args.mask_thickness
    txt_coord = args.txt_coord
    write_vcps = args.write_vcps

    frangi_sigma_min = args.frangi_sigma_min
    frangi_sigma_max = args.frangi_sigma_max
    frangi_sigma_step = args.frangi_sigma_step
    frangi_black_ridges = args.frangi_black_ridges
    frangi_alpha = args.frangi_alpha
    frangi_beta = args.frangi_beta
    frangi_gamma = args.frangi_gamma

    trim_val = args.trim_val

    print("Frangi filter parameters:")
    print(f"  sigma_min      : {frangi_sigma_min}")
    print(f"  sigma_max      : {frangi_sigma_max}")
    print(f"  sigma_step     : {frangi_sigma_step}")
    print(f"  black_ridges   : {frangi_black_ridges}")
    print(f"  alpha          : {frangi_alpha}")
    print(f"  beta           : {frangi_beta}")
    print(f"  gamma          : {frangi_gamma}")


    print(f'Writing coords to txt file: {txt_coord}')
    if total_seg_points:
        print(f'Reduction in points needed to {total_seg_points}')

    output_folder = args.output_folder
    cv_wait_key = args.cv_wait_key
    cv_show = args.cv_show
    print(f'CV_WAIT_KEY: {cv_wait_key}, CV_SHOW: {cv_show}')

    if number_of_clusters==0:
        print("Number of clusters cannot be zero.")
        return
    

    kmeans(volpkg_dir=volpkg, film_slice=film_slice, number_of_clusters=number_of_clusters, num_seg_points=num_seg_points, output_folder=output_folder, 
           cv_wait_key=cv_wait_key, cv_show=cv_show, threshold_factor=threshold_factor, volume=volume, total_seg_points=total_seg_points,
           intensity_alpha=intensity_alpha, gaussian_kernel=gaussian_kernel, mask_thickness=mask_thickness, txt_coord=txt_coord,
           write_vcps=write_vcps, frangi_sigma_min=frangi_sigma_min, frangi_sigma_max=frangi_sigma_max,
           frangi_sigma_step=frangi_sigma_step, frangi_black_ridges=frangi_black_ridges, 
           frangi_alpha=frangi_alpha, frangi_beta=frangi_beta, frangi_gamma=frangi_gamma,
           trim_val=trim_val)


if __name__ == "__main__":
    main()