import argparse
from pathlib import Path
import cv2
import numpy as np

import matplotlib.pyplot as plt
import json

def simulate_photographic_response(C, V, output, A=1.0, k=1.0, supertitle="Super title"):
    
    C_flat = C.flatten()
    V_flat = V.flatten()
    
    plt.figure(figsize=(40, 5)) # width, height

    plt.subplot(1, 3, 1)
    plt.imshow(C, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('CT image')

    plt.subplot(1, 3, 2)
    plt.imshow(V, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Optical image')

    plt.subplot(1, 3, 3)
    plt.scatter(V_flat, C_flat, s=1, label='Predicted C from V')  # (x, y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Visible light value (transmittance)")
    plt.ylabel("X-ray/CT recon value (attenuation)")
    plt.legend()
    plt.title("CT vs Visible Image")



    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.suptitle(supertitle, fontsize=16)

    plt.savefig(f"{output}", dpi=300, bbox_inches='tight')
    plt.close()





def make_dh(input_ct, input_optical, A, k, output, supertitle="Super title"):
    # Normalizing globally
    print(f'Making Hurter-Driffield curve')
    ct_iinfo = np.iinfo(input_ct.dtype)
    ct_min, ct_max = ct_iinfo.min, ct_iinfo.max

    optical_iinfo = np.iinfo(input_optical.dtype)
    optical_min, optical_max = optical_iinfo.min, optical_iinfo.max

    ct_norm_g = (input_ct - ct_min) / (ct_max - ct_min)
    ct_norm_l = (input_ct - input_ct.min()) / (input_ct.max() - input_ct.min())
    optical_norm_g = (input_optical - optical_min) / (optical_max - optical_min)
    optical_norm_l = (input_optical - input_optical.min()) / (input_optical.max() - input_optical.min())

    print(f'Normalised(g). CT min, max: ({ct_norm_g.min(), ct_norm_g.max()}), Optical min, max: ({optical_norm_g.min(), optical_norm_g.max()})')
    print(f'Normalised(l). CT min, max: ({ct_norm_l.min(), ct_norm_l.max()}), Optical min, max: ({optical_norm_l.min(), optical_norm_l.max()})')

    simulate_photographic_response(C=ct_norm_g, V=optical_norm_g, A=A, k=k, output=f'{output}/ct_visible_global.png', supertitle=supertitle)
    simulate_photographic_response(C=ct_norm_l, V=optical_norm_l, A=A, k=k, output=f'{output}/ct_visible_local.png', supertitle=supertitle)
    


def main():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--input-ct-image', '-c', help="Input image of a CT scan", type=str, required=True)
    parser.add_argument('--input-optical-image', '-v', help="Input image of a optical positive \
                        image formed by visible light", type=str, required=True)
    parser.add_argument('--k', help="Attenuation scaling factor", type=float, required=True)
    parser.add_argument('--A', help="Maximum optical density.", type=float, required=True)
    parser.add_argument('--output', help="Path to the output file for plot", type=str, required=True)
    parser.add_argument('--roi', help="Path to ROI file", type=str)
    parser.add_argument('--roi-key', help="Path to ROI file", type=str)
    parser.add_argument('--title', help="Title for overall plot", type=str)
    args = parser.parse_args()

    roi = args.roi
    roi_key = args.roi_key
    title = args.title

    if (bool(roi) ^ bool(roi_key)):
        return
    
    coords = {}
    if roi:
        with open(roi, 'r') as f:
            coords = json.load(f)
        print(f'ROI key is: {roi_key}')
        my_coords = coords[roi_key]
        print(my_coords)



    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    ct = Path(args.input_ct_image)
    optical = Path(args.input_optical_image)
    k = args.k
    A = args.A

    ct_image = cv2.imread(ct, 0)
    if roi:
        ct_image = ct_image[my_coords['y']:my_coords['y']+my_coords['h'], my_coords['x']:my_coords['x']+my_coords['w']]
    print(f'Shape of CT image: {ct_image.shape}, dtype: {ct_image.dtype}, iinfo: {np.iinfo(ct_image.dtype)}')

    optical_image = cv2.imread(optical, 0)
    if roi:
        optical_image = optical_image[my_coords['y']:my_coords['y']+my_coords['h'], my_coords['x']:my_coords['x']+my_coords['w']]
    
    print(f'Shape of optical image: {optical_image.shape}, dtype: {optical_image.dtype}, iinfo: {np.iinfo(optical_image.dtype)}')

    if title:
        make_dh(input_ct=ct_image, input_optical=optical_image, A=A, k=k, output=output_path, supertitle=title)
    else:
        make_dh(input_ct=ct_image, input_optical=optical_image, A=A, k=k, output=output_path)


if __name__ == "__main__":
    main()



