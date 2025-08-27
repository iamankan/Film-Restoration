import cv2
import argparse
from pathlib import Path
import exiftool
import imageio.v3 as iio

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exposures', help="Folder for all the exposure files to be used for HDR stacking.", required=True, type=str)
    p.add_argument('--output', help="Output folder name where TIFF file HDR will be stacked.", required=True, type=str)
    args = p.parse_args()
    exposures = Path(args.exposures)
    output = Path(args.output)
    print(exposures)
    print(output)
    output.mkdir(exist_ok=True, parents=True)
    images = []
    exposure_times = []
    for exp_img_path in exposures.iterdir():
        images.append(iio.imread(exp_img_path))
        print(images[-1].min(), images[-1].max())
        with exiftool.ExifToolHelper() as eth:
            metadata = eth.get_metadata(exp_img_path)[0]
            exposure_times.append(metadata['EXIF:ExposureTime'])
    # print(images)
    print(exposure_times)

if __name__ == "__main__":
    main()
