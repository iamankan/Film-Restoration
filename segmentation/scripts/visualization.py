import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
from natsort import natsorted
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


OUTPUT_PATH_NAME = 'visualization'
SLICE_PATH_NAME = 'slice_images'
SLICE_SEGMENTATION_PATH_NAME = 'slice_segmentation_images'
SLICE_LOCAL_WINDING_PATH_NAME = 'slice_winding_images'
SLICE_GLOBAL_WINDING_PATH_NAME = 'slice_global_winding_cleaned'

def stack_images_clean(image_list, spacing=10, alpha=0.5, target_size=(150, 150)):
    # Set background color (e.g., 'white' or 'black')
    bg_color = 'white' 
    fig = plt.figure(figsize=(10, 10), facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)

    for i, img in enumerate(image_list):
        # Resize logic
        pil_img = Image.fromarray(img.astype(np.uint8))
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        img_small = np.array(pil_img) / 255.0
        
        h, w, _ = img_small.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        z = np.full_like(x, i * spacing)

        rgba = np.zeros((h, w, 4))
        rgba[..., :3] = img_small[..., :3]
        rgba[..., 3] = alpha

        # Plot with linewidth=0 to remove the mesh grid lines
        ax.plot_surface(x, y, z, facecolors=rgba, 
                        rstride=1, cstride=1, 
                        shade=False, antialiased=True,
                        linewidth=0)

    # --- THE CLEANUP ---
    ax.set_axis_off()          # Removes axes, ticks, and labels
    ax.grid(False)             # Ensures grid is off
    ax.set_box_aspect([1, 1, 0.5]) 
    
    # Optional: adjust view for a better perspective
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    # plt.show()
    return fig


def stack_images_resized(image_list, spacing=5, alpha=0.5, target_size=(150, 150)):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, img in enumerate(image_list):
        # 1. Convert to PIL Image to resize efficiently
        # Assumes img is a numpy array. If it's already a PIL object, skip Image.fromarray
        pil_img = Image.fromarray((img).astype(np.uint8))
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        
        # 2. Convert back to numpy and normalize
        img_small = np.array(pil_img) / 255.0
        
        h, w, _ = img_small.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        z = np.full_like(x, i * spacing)

        # 3. Add Alpha Channel
        rgba = np.zeros((h, w, 4))
        rgba[..., :3] = img_small[..., :3]
        rgba[..., 3] = alpha

        # 4. Plot (Keep rstride/cstride at 1 since we already downsampled)
        ax.plot_surface(x, y, z, facecolors=rgba, 
                        rstride=1, cstride=1, 
                        shade=False, antialiased=True)

    # Improve the view and scale
    ax.view_init(elev=20, azim=45)
    ax.set_zlim(0, len(image_list) * spacing)
    
    # Force equal aspect ratio so the volume isn't stretched
    ax.set_box_aspect([1, 1, 0.5]) 
    
    plt.show()


def stack_images_3d_transparent(image_list, spacing=1.0, alpha=0.5):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, img in enumerate(image_list):
        # 1. Ensure image is float [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
            
        h, w, _ = img.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        z = np.full_like(x, i * spacing)

        # 2. Add the Alpha channel
        # Create an array of shape (h, w, 4)
        rgba = np.zeros((h, w, 4))
        rgba[..., :3] = img      # Set RGB
        rgba[..., 3] = alpha     # Set Alpha (0.0 to 1.0)

        # 3. Plot with the RGBA data
        ax.plot_surface(x, y, z, facecolors=rgba, 
                        rstride=2, cstride=2, 
                        shade=False, antialiased=True)

    ax.view_init(elev=25, azim=-45)
    
    # Optional: Set axis limits so the plot doesn't look squashed
    ax.set_zlim(0, len(image_list) * spacing)
    plt.show()


def stack_images_3d(image_list, spacing=1.0):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get dimensions from the first image
    h, w, _ = image_list[0].shape

    # Create a meshgrid for the surface
    # We use these to define the "floor" each image sits on
    x, y = np.meshgrid(np.linspace(0, w, w), np.linspace(0, h, h))

    for i, img in enumerate(image_list):
        # Normalize image to [0, 1] if it's in [0, 255]
        if img.max() > 1.0:
            img = img / 255.0
            
        # The Z-layer for this specific image
        z = np.full_like(x, i * spacing)

        # facecolors expects an array of RGBA values
        # We use the image data directly as the colors
        ax.plot_surface(x, y, z, rstride=5, cstride=5, facecolors=img, 
                        shade=False, antialiased=True)

    # Adjust the view angle
    ax.view_init(elev=20, azim=45)
    
    # Clean up the axes to make it look like a volume
    ax.set_zlabel('Image Index')
    ax.set_title('3D Stacked Volume')
    
    plt.show()

def read_slices(slice_dir: Path, target: Path, target_size=(256,256), num_slices: int=5, alpha:float = 0.8, spacing: float=1.0):
    stacked_slices = []
    for slice_idx, slice_path in enumerate(natsorted(slice_dir.iterdir())):
        if num_slices:
            if slice_idx > num_slices - 1:
                break
        slice_img = iio.imread(slice_path)
        # print(f'Read {slice_path} and the shape of the slice is {slice_img.shape}')
        stacked_slices.append(slice_img)
    print(f'Total number of slices: {len(stacked_slices)}')
    fig = stack_images_clean(image_list=stacked_slices, target_size=target_size, alpha=alpha, spacing=spacing)
    if target:
        print(f'Saving volume stack as {target}')
        fig.savefig(target, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)

def main():
    # use argparse to get the path to the images and the output directory
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, help='Path to the input images')
    args = p.parse_args()
    input_parent_dir = args.input_dir
    slice_images_dir = input_parent_dir / SLICE_PATH_NAME
    slice_segmentation_dir = input_parent_dir / SLICE_SEGMENTATION_PATH_NAME
    slice_local_winding_dir = input_parent_dir / SLICE_LOCAL_WINDING_PATH_NAME
    slice_global_winding_dir = input_parent_dir / SLICE_GLOBAL_WINDING_PATH_NAME
    output_dir = input_parent_dir / OUTPUT_PATH_NAME
    output_dir.mkdir(exist_ok=True, parents=True)
    spacing = 1.0
    alpha = 0.5
    num_slices = 5
    target_size=(256,256)
    read_slices(slice_dir=slice_images_dir, target=output_dir/"slice_volume.png", alpha=alpha, spacing=spacing, num_slices=num_slices, target_size=target_size)
    read_slices(slice_dir=slice_segmentation_dir, target=output_dir/"slice_segmentation.png", alpha=alpha, spacing=spacing, num_slices=num_slices, target_size=target_size)
    read_slices(slice_dir=slice_local_winding_dir, target=output_dir/"slice_segmentation_local_winding.png", alpha=alpha, spacing=spacing, num_slices=num_slices, target_size=target_size)
    read_slices(slice_dir=slice_global_winding_dir, target=output_dir/"slice_segmentation_global_winding.png", alpha=alpha, spacing=spacing, num_slices=num_slices, target_size=target_size)
    


if __name__ == "__main__":
    main()
    