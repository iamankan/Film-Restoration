import numpy as np
import argparse
from pathlib import Path
import imageio.v3 as iio
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import imageio.v3 as iio  # or import imageio
import numpy as np

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
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved to {output}")
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
    
    __draw_histogram__(histogram_dict=histogram_dict, output=Path(f'./{name}_histogram.png'))

    return histogram_dict

def __histogram__(frame_path: Path):
    print('Staring generating histogram.')
    histogram_dict = {
        'layers':{},
        'filters':{},
        'pca':{},
        'optical':{}
    }
    # Let's start with layers
    layers_dir = frame_path / 'layers'
    for idx, layer in enumerate(layers_dir.iterdir()):
        name = f'layer_{idx:03d}'
        image_path = layer
        histogram_dict['layers'][idx] = __make_histogram__(name=name, image_path=image_path)

    return histogram_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i', required=True, type=str, help="Input path for the data")
    parser.add_argument('--output','-o', required=True, type=str, help="Save the histogram")
    args = parser.parse_args()
    frame_path = Path(args.input)
    output_path = Path(args.output)
    histogram_dict = __histogram__(frame_path=frame_path)


if __name__ == "__main__":
    main()