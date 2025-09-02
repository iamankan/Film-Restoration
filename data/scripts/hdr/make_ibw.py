import argparse
from pathlib import Path
import shutil
import imageio.v3 as iio

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', help="Folder that have the data", type=str)
    p.add_argument('--target-file', help="File that needs to be used", type=str)
    p.add_argument('--output', help="Output folder path", type=str)
    args = p.parse_args()

    input_folder = Path(args.input)
    target_file = args.target_file
    output_folder = Path(args.output)

    print(f'Input folder: {input_folder}\nTarget file: {target_file}\nOutput folder: {output_folder}')

    for sample in input_folder.iterdir():
        ibw_id = sample.stem
        dest_folder = Path(output_folder / ibw_id / 'optical')
        dest_folder.mkdir(exist_ok=True, parents=True)
        dest_file = Path(dest_folder / f'{ibw_id}_{target_file}')
        src_file = sample / f'hdr/{target_file}'
        shutil.copy(src=src_file, dst=dest_file)
        img = iio.imread(src_file)
        print(f'{ibw_id}: min: {img.min()}, max: {img.max()}')


if __name__ == "__main__":
    main()
