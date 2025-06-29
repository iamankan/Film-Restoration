import argparse
from pathlib import Path
import uuid
import cv2
import numpy as np

def segment(volpkg_path: Path, volume_id: str, output_dir: Path, slice_name: str, threshold: int, min_connected_points: int, connectivity:int, gaussian_kernel: int):
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
    _, working_image_threshold = cv2.threshold(src=working_image_norm, thresh=working_image_norm.max()/threshold, maxval=working_image_norm.max(), type=cv2.THRESH_BINARY)
    working_image_threshold = working_image_threshold.astype(np.uint8)
    print(working_image_threshold.min(), working_image_threshold.max(), working_image_threshold.dtype)
    if output_dir:
        cv2.imwrite(f'{output_dir}/threshold_image.jpg', working_image_threshold*255)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(image=working_image_threshold, connectivity=connectivity)
    mask = {}
    print(f"Number of connected components: {num_labels}")
    for i in range(1, num_labels): # 0 is the background, always. So, starting from 1
        componentMask = (labels_im == i).astype("uint8")
        if len(np.where(componentMask==1)[0]) > min_connected_points:
            mask[i] = componentMask
            if output_dir:
                cv2.imwrite(f'{output_dir}/componentMask_{i}.jpg', componentMask*255)
    components = mask.keys()
    print(f"Number of acceptable connected components: {len(components)}")
    if len(components) > 0:
        for component_id in components:
            print(f'Working on component-{component_id}')
            componentMask = mask[component_id]
            tmp_img = dummy_image.copy()
            tmp_img[:,:,0] = tmp_img[:,:,0]*componentMask
            tmp_img[:,:,1] = tmp_img[:,:,1]*componentMask
            tmp_img[:,:,2] = tmp_img[:,:,2]*componentMask
            if output_dir:
                cv2.imwrite(f'{output_dir}/maskedcomponent_{component_id}.jpg', tmp_img)





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
    args= parser.parse_args()

    volpkg = Path(args.volpkg)
    volume_id = args.volume
    output_dir = args.output_dir
    slice_name = args.slice_name
    threshold = args.threshold
    min_connected_points = args.min_connected_points
    connectivity = args.connectivity
    gaussian_kernel = args.gaussian_kernel

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        aux_id = str(uuid.uuid4())
        print(f'Auxiliary id is {aux_id}')
        aux_path = output_dir / aux_id
        aux_path.mkdir(exist_ok=True, parents=True)
        print(f'Auxiliary outputs will be saved to {aux_path}')
    else:
        print(f'No auxiliary outputs will be saved')
    segment(volpkg_path=volpkg, volume_id=volume_id, output_dir=aux_path, slice_name=slice_name, threshold=threshold, 
            min_connected_points=min_connected_points, connectivity=connectivity, gaussian_kernel=gaussian_kernel)

if __name__ == "__main__":
    main()