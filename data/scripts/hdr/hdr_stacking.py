import cv2 as cv
import argparse
from pathlib import Path
import exiftool
import numpy as np
import imageio.v3 as iio


def hdr(img_list, exposure_times):
    merge_debevec = cv.createCalibrateDebevec()
    debevec_response = merge_debevec.process(img_list, times=exposure_times.copy())
    merge_debevec = cv.createMergeDebevec()
    hdr_merged = merge_debevec.process(img_list, exposure_times.copy(), debevec_response)
    print(f'debevec done')
    return hdr_merged

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exposures-dir', help="Folder for all the exposure files to be used for HDR stacking.", required=True, type=str)
    p.add_argument('--output-dir', help="Output folder name where TIFF file HDR will be stacked.", required=True, type=str)
    p.add_argument('--dark-level', help="Enter a dark level for sensor.", type=float, default=588)
    p.add_argument('--raw-bit', help="How many bit needs to represent the true raw.", type=int, default=14)
    p.add_argument('--prefix-name', help="Prefix name of output file/image.", type=str, default='test')
    p.add_argument('--normalized', help="Use this flag if the image is already normalized between 0-1", action='store_true')
    p.add_argument('--debayered', help="Use this flag to tell whether the image is already debayered.", action='store_true')

    args = p.parse_args()
    prefix_name = args.prefix_name
    exposures_dir = Path(args.exposures_dir)
    output_dir = Path(args.output_dir)
    dark_level = args.dark_level
    raw_bit = args.raw_bit
    is_normalized = args.normalized
    is_debayered = args.debayered
    print(f'dark_level: {dark_level}, raw_bit: {raw_bit}, is_normalized: {is_normalized}, \
          is_debayered: {is_debayered}, prefix_name: {prefix_name}')
    print(exposures_dir)
    print(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    images = []
    exposure_times = []
    for exp_img_path in exposures_dir.iterdir():
        img = iio.imread(exp_img_path)
        print(f'Shape of the image that has been read: {img.shape}')
        print(img.min(), img.max(), img.dtype)
        if not is_normalized:
            # Nikon D5300 saves 14-bit NEF Raw images
            max_val = 2**raw_bit - 1
            img = (img - dark_level)/(max_val - dark_level)
        '''
        https://docs.opencv.org/3.4/d2/df0/tutorial_py_hdr.html

        The first stage is simply loading all images into a list. In addition, we will need the exposure times for the 
        regular HDR algorithms. Pay attention for the data types, as the images should be 1-channel or 3-channels 8-bit 
        (np.uint8) and the exposure times need to be float32 and in seconds.
        '''
        img = (img*(2**8-1)).astype(np.uint8)
        print(f'The image is {img.dtype}. The min val is {img.min()}, and the max val is {img.max()}')

        images.append(img)
        with exiftool.ExifToolHelper() as eth:
            metadata = eth.get_metadata(exp_img_path)[0] # type: ignore
            exposure_sec = float(metadata['EXIF:ExposureTime'])
            print(f'Exposure in seconds: {exposure_sec}')
            exposure_times.append(exposure_sec)
    exposure_times = np.array(exposure_times)
    exposure_times = exposure_times.astype(np.float32)
    print(exposure_times)

    hdr_merge = hdr(img_list=images, exposure_times=exposure_times)
    print(hdr_merge.shape, hdr_merge.dtype, hdr_merge.min(), hdr_merge.max())
    hdr_norm = ((hdr_merge - hdr_merge.min())/(hdr_merge.max() - hdr_merge.min()))
    hdr_norm_positive = 1 - hdr_norm
    
    hdr_16bit = (hdr_norm_positive*np.iinfo(np.uint16).max).astype(np.uint16)
    hdr_8bit = (hdr_norm_positive*np.iinfo(np.uint8).max).astype(np.uint8)
    iio.imwrite(f'{output_dir}/{prefix_name}_hdr_positive.tiff', hdr_norm_positive)
    iio.imwrite(f'{output_dir}/{prefix_name}_hdr_8bit.tiff', hdr_8bit)
    iio.imwrite(f'{output_dir}/{prefix_name}_hdr_16bit.tiff', hdr_16bit)

    if not is_debayered:
        # Debayering
        hdr_debayered_16 = cv.cvtColor(hdr_16bit, cv.COLOR_BayerRGGB2GRAY)
        iio.imwrite(f'{output_dir}/{prefix_name}_hdr_debayer_grey_16bit.tiff', hdr_debayered_16)

        hdr_debayered_8 = cv.cvtColor(hdr_8bit, cv.COLOR_BayerRGGB2GRAY)
        iio.imwrite(f'{output_dir}/{prefix_name}_hdr_debayer_grey_8bit.tiff', hdr_debayered_8)
    

if __name__ == "__main__":
    main()
