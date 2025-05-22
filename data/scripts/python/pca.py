import argparse
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from sklearn.decomposition import PCA
from einops import rearrange
from natsort import natsorted

import logging
logger = logging.getLogger(__name__)

def __pca__(input: Path, output: Path) -> None:
    logger.info("PCA started!")

    images = natsorted([i for i in input.iterdir()])
    logger.debug(images)

    # Step 1: Read images into numpy array
    np_images = np.array([iio.imread(image) for image in images])
    logger.debug(np_images.shape)
    

    # Step 3: Flatten each image
    n, h, w = np_images.shape
    np_images_flatten = rearrange(np_images, 'c h w -> (h w) c')  # shape: (n, h*w)

    # Step 4: PCA
    pca = PCA(n_components=n)
    pca_transform = pca.fit_transform(np_images_flatten)

    pca_transform = rearrange(pca_transform, '(h w) c -> c h w', c=n, h=h, w=w)

    pca_transform = pca_transform.astype(np.float32)

    output.mkdir(parents=True, exist_ok=True)

    [iio.imwrite(f'{output}/pca_{i:03d}.tif', x) for i,x in enumerate(pca_transform)]

    logger.info("PCA done!")
    

def main():
    """
    This function is the main function that takes in no arguments.
    """
    parser = argparse.ArgumentParser(description='This is the main function. From this function it takes some arguments to calculate and give back pca.')
    parser.add_argument('-i', '--input', help='Directory of the input images.', type=Path, required=True)
    parser.add_argument('-o', '--output', help='Destination folder where the pca files will be stored.', required=True, type=Path)
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG
    )
    # parser.add_argument('-n','--number-of-components', help='Number of principle components required. If not provided, it will take the number of images by default.', required=False, type=int)
    args = parser.parse_args()
    input = args.input
    output = args.output
    # num = args.number_of_components
    __pca__(input=input, output=output)



if __name__ == '__main__':
    main()