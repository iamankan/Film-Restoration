from pathlib import Path
from skimage.filters import frangi
import argparse
import cv2
import imageio.v3 as iio
import numpy as np
import time

def frangi3d(volpkg:Path, volume_id:str, frangi_sigma_start: int, frangi_sigma_end: int, frangi_sigma_step: int, 
             frangi_alpha: float, frangi_beta: float, frangi_gamma: float):
    volume_path = volpkg / f'volumes/{volume_id}'
    assert volume_path.is_dir(), f'{volume_path} does not exist'

    start_time = time.perf_counter()
    film_volume = np.array([iio.imread(x) for x in volume_path.glob('*.tif')])    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Time taken to read the volume: {elapsed_time:.4f} seconds")

    print(f'Shape of the volume is {film_volume.shape}')

    start_time = time.perf_counter()
    frangi_volume = frangi(
        image=film_volume[:3],
        sigmas=range(frangi_sigma_start, frangi_sigma_end, frangi_sigma_step),
        alpha=frangi_alpha,
        beta=frangi_beta,
        gamma=frangi_gamma
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Time taken to do frangi over the volume: {elapsed_time:.4f} seconds")
    print(f'Frangi volume shape: {frangi_volume.shape}')





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volpkg', help="Path to the *.volpkg", type=str, required=True)
    parser.add_argument('--volume', help="The Volume ID inside the *.volpkg to work on", type=str, required=True)

    parser.add_argument('--frangi-sigma-start', help="Staring sigma value for frangi filtering", type=int, default=4)
    parser.add_argument('--frangi-sigma-end', help="Ending sigma value for frangi filtering", type=int, default=14)
    parser.add_argument('--frangi-sigma-step', help="Steps to increase sigma value for frangi filtering", type=int, default=2)
    parser.add_argument('--frangi-alpha', help="Alpha value frangi filtering (for vessel/blob)", type=float, default=0.5)
    parser.add_argument('--frangi-beta', help="Beta value frangi filtering (for 3D)", type=float, default=0.5)
    parser.add_argument('--frangi-gamma', help="Gamma value frangi filtering (for contrast)", type=float, default=1)
    args = parser.parse_args()
    
    volpkg = Path(args.volpkg)
    volume_id = args.volume
    frangi_sigma_start = args.frangi_sigma_start
    frangi_sigma_end = args.frangi_sigma_end
    frangi_sigma_step = args.frangi_sigma_step
    frangi_alpha = args.frangi_alpha
    frangi_beta = args.frangi_beta
    frangi_gamma = args.frangi_gamma

    frangi3d(volpkg=volpkg, volume_id=volume_id, frangi_sigma_start=frangi_sigma_start,
             frangi_sigma_end=frangi_sigma_end, frangi_sigma_step=frangi_sigma_step,
             frangi_alpha=frangi_alpha, frangi_beta=frangi_beta, frangi_gamma=frangi_gamma)




if __name__ == "__main__":
    main()