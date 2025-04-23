# TO-DO List
- ✅ Figure out scan protocols
- ✅ Scanning (001-015)
- ✅ Code the `packager.sh` script to generate `*.volpkg` files 
- ❌ Segmentation
- ❌ Check the orientations
- ❌ Code the `render.sh` script
- ❌ Image registration
- ❌ Rendering composites
    - ❌ Max
    - ❌ Average/Mean
    - ❌ Median
- ❌ Make the [`Dataset`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) for machine learning usage and publishing
- ❌ Calculate the following quality assesment metrics between the rendered image and the digital image:
    - ❌ Peak Signal-to-Noise Ratio (PSNR)
    - ❌ Structural Similarity Metric (SSIM)
    - ❌ Mean Absolute Error (MAE)
    - ❌ Mean Squared Error (MSE)
- ❌ Map the pixel intensity after registration between digital image and the rendered image for:
    - ❌ Max
    - ❌ Average/Mean
    - ❌ Median
- ❌ Make a shell script to produce the data in the following format:
    ```shell
    IlfordBW
    ├── ID
        ├── layers          # contains `ID_00.png`, `ID_01.png`, …, `ID_13.png`
        ├── match           # contains `ID_optical.jpg`, `ID_register.tif`
        ├── obj             # contains `ID.obj`, `ID.mtl` and `ID.tif`
        ├── ppm             # contains `ID_….tif`, `*.png`, `*.ppm`
        └── render          # contains `ID_max.tif`, `ID_mean.tif`, `ID_median.tif`
    ```


### Legend:
- ✅ **_Completed_**
- 🌝 **_In progress_**
- ❌ **_Not started_**