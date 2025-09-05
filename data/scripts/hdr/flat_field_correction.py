import argparse
from pathlib import Path
import exiftool
import imageio.v3 as iio
from educelab.imgproc.correction import flatfield_correction

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

    # print(f'Input-image-directory: {input_image_directory}')
    # print(f'Input-darkfield-directory: {input_darkfield_directory}')
    # print(f'Input-lightfield-directory: {input_lightfield_directory}')
    # print(f'Output-flatfield-directory: {output_flatfield_directory}')

    # Save the darkfields
    dfd = {} # Darkfield exposure dictionary
    for df in input_darkfield_directory.iterdir():
        # print(f'Darkfield: {df}')
        dfd[get_exposure_sec(df)] = df
    df_sorted_exp = sorted(dfd.keys())

    # Save the lightfields
    lfd = {} # Lightfield exposure dictionary
    for lf in input_lightfield_directory.iterdir():
        # print(f'Lightfield: {lf}')
        lfd[get_exposure_sec(lf)] = lf
    
    for img in input_image_directory.iterdir():
        exposure = get_exposure_sec(img_path=img)
        img_ffc = flatfield_correction(image=iio.imread(img), lf=iio.imread(lfd[exposure]), df=iio.imread(dfd[exposure]))
        img_name = img.stem
        save_as = f'{output_flatfield_directory}/{img_name}_ffc.tiff'
        iio.imwrite(save_as, img_ffc)
        print(f'Image saved as FF: {f'{output_flatfield_directory}/{img_name}_ffc.tiff'}')



    




if __name__ == "__main__":
    main()