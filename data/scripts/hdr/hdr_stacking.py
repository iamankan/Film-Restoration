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

    args = p.parse_args()
    exposures_dir = Path(args.exposures_dir)
    output_dir = Path(args.output_dir)
    dark_level = args.dark_level
    raw_bit = args.raw_bit
    print(f'dark_level: {dark_level}, raw_bit: {raw_bit}')
    print(exposures_dir)
    print(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    images = []
    exposure_times = []
    # Nikon D5300 saves 14-bit NEF Raw images
    max_val = 2**raw_bit - 1
    for exp_img_path in exposures_dir.iterdir():
        img = iio.imread(exp_img_path)
        print(img.min(), img.max())
        img = (img - dark_level)/(max_val - dark_level)
        img = img.astype(np.float16)
        print(img.min(), img.max(), img.dtype)
        '''
        https://docs.opencv.org/3.4/d2/df0/tutorial_py_hdr.html

        The first stage is simply loading all images into a list. In addition, we will need the exposure times for the 
        regular HDR algorithms. Pay attention for the data types, as the images should be 1-channel or 3-channels 8-bit (np.uint8) 
        and the exposure times need to be float32 and in seconds.
        '''
        img = (img*(2**8-1)).astype(np.uint8)
        print(img.min(), img.max())

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
    hdr_norm = 1-((hdr_merge - hdr_merge.min())/(hdr_merge.max() - hdr_merge.min()))
    hdr_16bit = (hdr_norm*np.iinfo(np.uint16).max).astype(np.uint16)
    hdr_8bit = (hdr_norm*np.iinfo(np.uint8).max).astype(np.uint8)
    iio.imwrite(f'{output_dir}/hdr.tiff', hdr_merge)
    iio.imwrite(f'{output_dir}/hdr_8bit.tiff', hdr_8bit)
    iio.imwrite(f'{output_dir}/hdr_16bit.tiff', hdr_16bit)

    # Debayering
    hdr_debayered_16 = cv.cvtColor(hdr_16bit, cv.COLOR_BayerRGGB2GRAY)
    iio.imwrite(f'{output_dir}/hdr_debayer_grey_16bit.tiff', hdr_debayered_16)

    hdr_debayered_8 = cv.cvtColor(hdr_8bit, cv.COLOR_BayerRGGB2GRAY)
    iio.imwrite(f'{output_dir}/hdr_debayer_grey_8bit.tiff', hdr_debayered_8)
    

if __name__ == "__main__":
    main()
