import numpy as np
import argparse
from pathlib import Path
import imageio.v3 as iio
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def __draw_histogram__(histogram_dict: dict, output=None, bit_depths=[8, 12, 16]):
    name = histogram_dict.get('name')
    img_path = histogram_dict.get('image')
    if not name or not img_path:
        print("Missing 'name' or 'image'")
        return

    img = iio.imread(img_path)
    histograms = histogram_dict.get('histogram', {})

    n_cols = 1 + len(bit_depths)
    fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.suptitle(name, fontsize=20, y=1.02)

    # Show image
    axs[0].imshow(img, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title("Image")

    # Plot histograms
    for i, bit_depth in enumerate(bit_depths):
        ax = axs[i + 1]
        key = f"{bit_depth}-bit"
        if key in histograms:
            histogram = histograms[key]["histogram"]
            bins = histograms[key]["bin"]

            # Use bin centers if necessary
            if len(bins) == len(histogram) + 1:
                bin_centers = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))
            else:
                bin_centers = bins

            df = pd.DataFrame({"x": bin_centers, "y": histogram})
            sns.lineplot(data=df, x="x", y="y", ax=ax, color="steelblue")

            ax.set_xlim(0, 1)
            ax.set_xticks([0, 1])
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{bit_depth}-bit Histogram")
        else:
            ax.axis('off')

    plt.tight_layout()
    if output:
        output_file = f'{output}/{name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved as {output_file}")
    else:
        plt.show()


def __get_histogram__(img: np.ndarray, bit_depth: int=8):
    levels = int(math.pow(2, bit_depth))
    hist, bin_edges = np.histogram(img, bins=levels, range=(0, 1))
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2

    return {'histogram': hist, 'bin': bin_centers}

def __make_histogram__(image_path: Path, name: str, bit_depths: list=[8, 12, 16]):
    '''
    RETURN FORMAT:
        {
            'name':'optical_registered',
            'image':'/path/to/image',
            'histogram':{
                '8-bit':{'histogram': [...], 'bin': [...]},
                '12-bit':{'histogram': [...], 'bin': [...]},
                '16-bit':{'histogram': [...], 'bin': [...]}
            }
        }
    '''
    histogram_dict = {
        'name': name,
        'image': image_path,
        'histogram':{}
    }
    img = iio.imread(image_path)
    if not img.any():
        return
    dtype = img.dtype.type
    
    if np.issubdtype(dtype, np.integer):
        norm_factor = (np.iinfo(dtype).max - np.iinfo(dtype).min)
    else:
        norm_factor = (np.finfo(dtype).max - np.finfo(dtype).min)
    
    img_norm = img/norm_factor # Normalized image globally
    for bit_depth in bit_depths:
        histogram_dict['histogram'][f'{bit_depth}-bit'] = __get_histogram__(img=img_norm, bit_depth=bit_depth)
    

    return histogram_dict

def __histogram__(frame_path: Path):
    '''
    RETURN FORMAT:
        {
            'layers':{},
            'filters':{},
            'pca':{},
            'optical':{}
        }
    '''
    print('Staring generating histogram.')
    histogram_dict = {
        'layers':{},
        'filters':{},
        'pca':{},
        'optical':{}
    }
    frame_id = frame_path.stem

    # Let's start with layers
    layers_dir = frame_path / 'layers'
    for idx, layer in enumerate(layers_dir.iterdir()):
        name = f'ID:{frame_id}|Layer_{idx:03d}'
        image_path = layer
        histogram_dict['layers'][idx] = __make_histogram__(name=name, image_path=image_path)

    # Let's do filters
    filter_dir = frame_path / 'render'
    # MAX - filter
    name_max = f'ID:{frame_id}|Filter_MAX'
    image_path_max = filter_dir / f'{frame_id}_max.tif'
    histogram_dict['filters']['max'] = __make_histogram__(name=name_max, image_path=image_path_max)

    # AVG - filter
    name_avg = f'ID:{frame_id}|Filter_AVG'
    image_path_avg = filter_dir / f'{frame_id}_avg.tif'
    histogram_dict['filters']['avg'] = __make_histogram__(name=name_avg, image_path=image_path_avg)

    # MEDIAN - filter
    name_median = f'ID:{frame_id}|Filter_MEDIAN'
    image_path_median = filter_dir / f'{frame_id}_median.tif'
    histogram_dict['filters']['median'] = __make_histogram__(name=name_median, image_path=image_path_median)

    # Let's start with PCA
    pca_dir = frame_path / 'pca'
    if pca_dir.exists():
        for idx, pca in enumerate(pca_dir.iterdir()):
            name = f'ID:{frame_id}|PCA_{idx:03d}'
            image_path = pca
            histogram_dict['pca'][idx] = __make_histogram__(name=name, image_path=image_path)
    
    # Finally, LEt's do the registered optical image
    optical_dir = frame_path / 'match'
    name = f'ID:{frame_id}|OPTICAL_REGISTERED'
    image_path = optical_dir / f'{frame_id}_registered.tif'
    histogram_dict['optical']['registered'] = __make_histogram__(name=name, image_path=image_path)

    return histogram_dict

def __draw__(histogram_dictionary: dict, output_path: Path):
    '''
    INPUT
        {
            'layers':{},
            'filters':{},
            'pca':{},
            'optical':{}
        }
    '''
    if output_path:
        parent_keys = histogram_dictionary.keys()
        for parent_key in parent_keys:
            child_keys = histogram_dictionary[parent_key].keys()
            for child_key in child_keys:
                __draw_histogram__(histogram_dict=histogram_dictionary[parent_key][child_key], output=output_path)
    else:
        print(f'Provide output path.')

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i', required=True, type=str, help="Input path for the data")
    parser.add_argument('--output','-o', required=True, type=str, help="Save the histogram")
    args = parser.parse_args()
    frame_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    histogram_dict = __histogram__(frame_path=frame_path)
    __draw__(histogram_dictionary=histogram_dict, output_path=output_path)



    


if __name__ == "__main__":
    main()