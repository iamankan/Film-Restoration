import argparse
from pathlib import Path
import exiftool
import imageio.v3 as iio
from educelab.imgproc.correction import flatfield_correction
import subprocess
import numpy as np
import cv2


def copy_exif_data(source_image_path, destination_image_path):
    try:
        subprocess.run(
            ["exiftool", "-TagsFromFile", source_image_path, "-overwrite_original", destination_image_path],
            check=True
        )
        print(f"EXIF copied from {source_image_path} → {destination_image_path}")
    except subprocess.CalledProcessError as e:
        print("ExifTool failed:", e)



def get_exposure_sec(img_path):
    with exiftool.ExifToolHelper() as eth:
        metadata = eth.get_metadata(img_path)[0] # type: ignore
        exposure_sec = float(metadata['EXIF:ExposureTime'])
        return exposure_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image-directory', help="Directory where all the images are present", type=str)
    parser.add_argument('--input-darkfield-directory', help="Directory where there are darkfield images", type=str)
    parser.add_argument('--input-lightfield-directory', help="Directory where there are darkfield images", type=str)
    parser.add_argument('--output-flatfield-directory', help="Directory where the flatfield outputs will be stored", type=str)
    args = parser.parse_args()

    input_image_directory = Path(args.input_image_directory)
    input_darkfield_directory = Path(args.input_darkfield_directory)
    input_lightfield_directory = Path(args.input_lightfield_directory)
    output_flatfield_directory = Path(args.output_flatfield_directory)
    output_flatfield_directory.mkdir(exist_ok=True, parents=True)

    # Save the darkfields
    dfd = {} # Darkfield exposure dictionary
    for df in input_darkfield_directory.iterdir():
        dfd[get_exposure_sec(df)] = df

    # Save the lightfields
    lfd = {} # Lightfield exposure dictionary
    for lf in input_lightfield_directory.iterdir():
        lfd[get_exposure_sec(lf)] = lf
    
    print(f'Dark field: {dfd}\nLight field: {lfd}')
    
    for img in input_image_directory.iterdir():
        exposure = get_exposure_sec(img_path=img)
        print(f'Exposure (sec): {exposure}')
        print(f'DF: {dfd[exposure]}, LF: {lfd[exposure]}, img: {img}')
        my_img = iio.imread(img)
        my_img = cv2.cvtColor(my_img, cv2.COLOR_RGB2GRAY)

        my_df = iio.imread(dfd[exposure])
        my_df = cv2.cvtColor(my_df, cv2.COLOR_RGB2GRAY)

        my_lf = iio.imread(lfd[exposure])
        my_lf = cv2.cvtColor(my_lf, cv2.COLOR_RGB2GRAY)

        norm_img = my_img/(2**16 - 1)
        norm_df = my_df/(2**16 - 1)
        norm_lf = my_lf/(2**16 - 1)
        img_ffc = flatfield_correction(
            image=norm_img, 
            lf=norm_lf, 
            df=norm_df
        )
        img_ffc = img_ffc.astype(np.float32)
        img_name = img.stem
        save_as = f'{output_flatfield_directory}/{img_name}_ffc.tiff'
        iio.imwrite(save_as, img_ffc)
        print(f'Image saved as FF: {f'{output_flatfield_directory}/{img_name}_ffc.tiff'}')
        copy_exif_data(source_image_path=img, destination_image_path=save_as)
        

if __name__ == "__main__":
    main()