# Film Restoration Data collection and archiving


This sub-repository is for data creation and data handling. This sub-repository will do the following:

- Deals with scanning the Film data
- Unwrapping them films virtually and registering them with the optical data
- Making the [`Dataset`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

The folder named [media](./media) have the necessary diagrams and figure for explaining and coumenting this repository.

## About the data

The data is basically black and white 35mm film negatives from Ilford. These film negatives have base of Cellulose Acetate. We used a [Single-use Ilford 35mm point and shoot film camera](https://www.bhphotovideo.com/c/product/906263-REG/ilford_1174168_hp_5_plus_single_use.html/?ap=y&ap=y&smp=y&smp=y&store=420&lsft=BI%3A514&gad_source=1&gbraid=0AAAAAD7yMh1DVbO-pO1SMblJDb3QnAe9y&gclid=Cj0KCQjw_JzABhC2ARIsAPe3ynprAkHMB3UhmDwUgpBPw3CS5bkr8R0a7xeYWfiT26kRVtaIQQlRQyQaAt7OEALw_wcB) to take pictures of natural images of buildings, trees, and similar things. The camera has an aperture of f/9.5 and focal length of 30mm, and the film has an ISO of 400, with 27 exposures. Once we capture all the images, we send the camera to develop the film negatives to [THE DARKROOM](https://thedarkroom.com/?gad_source=1&gbraid=0AAAAAD_QVtmqLIPixAOewvZILGNeG1XDh&gclid=Cj0KCQjw_JzABhC2ARIsAPe3ynpmGXG8IHQhZgXm8cAXvLPrJYQo6qqNmeyQUNgpNU4vqNxjW1f18F0aAo9JEALw_wcB) to get the film developed and a super-resolution digital image. The super-resolution is [4492&times;6774](https://thedarkroom.com/scans/#:~:text=Super%20Scans%20are%204492×,default%20to%20the%20shortest%20dimension.).

There are about 20 frames to be scanned. After scanning, the rendered image is approximately 3732&times;1511.
Rendered from CT-Scan (3732&times;1511)            |  Digital image (4492&times;6774)
:-------------------------:|:-------------------------:
![xray](media/images/007_max.jpg)  |  ![digital](media/images/007_optical.jpg)

The above is an example of one of the images, showing a comparison of the rendered image, after segmenting, flattening and max-filtering on the left, and the developed digital image on the right.

## Scanning mount

The scanning mount is made by 3D printing using Nylon-11 using Multi-Jet Fusion (MJF) technology and the vapour polishing it. We used the [Xometry](https://www.xometry.com) 3D printing facility. 

## Scan protocol

We are using a micro-CT scanner named [SKYSCAN 1273](https://www.microphotonics.com/products/skyscan-1273/?gad_source=1&gbraid=0AAAAAD3S3B2QuPodgDBlfs7KxkPUgQmTH&gclid=Cj0KCQjw_JzABhC2ARIsAPe3ynqYqdmX3Hqahe6UjsMUkvmqDNq5EeDHVqJCxzmYp3XDas8xCLclCH4aAomsEALw_wcB). 

| Characteristic | Value |
|---------|------|
| Voltage | 60kV |
|Spot size | Medium spot |
|Frame average | 2 |
|Rotation angle | 0.150&deg; |
|360&deg; | ✅ |
|Resolution | 10&mu;m |
|Vertical position | 192mm |
|Diameter of the object | 15mm |
|Height of the object | 35mm |

## Post-processing


### Film scanning process
- Roll the film and put it into the mount and assign it an ID
- Correct the flat fielding if necessary before scanning (50% Avg with FF-off and empty FOV, 85% Max with FF-on and empty FOV, 40-60% Min with object of interest to be scanned in FOV)
- Set the protocol and scan it. For SKYSCAN it takes 1hr 20 mins to 1hr 30 mins on an average.
- After the scan is done:
- Take out the film
    - Frame it back with the corresponding ID
    - Take out a new film and redo till here from Step-1
    - The projections are saved under the folder type:
    `[Mount type]/FrameAvg/[ID]/[ID]_IBW_10um_60kV_MS`

### What to do with the films after scanning

- Import the projection into `NRecon`
- Check the misplacement and compensate for it (if needed)
- Preview few slices of reconstruction and fix the Histogram for better dynamic range
- Save the recon-files as `TIF(16 bit)` with a circular ROI, under `[Mount type]/FrameAvg/[ID]/[ID]_IBW_10um_60kV_MS/[ID]_IBW_10um_60kV_MS_Rec_ROI`

### What to do after CT-reconstruction

- Backup the projection data to the Seagate 24TB external HDD and delete it from the SKYSCAN Desktop
- Take the ROI-reconstruction data into an external SSD
- Convert that into `*.volpkg` using the following command using [`volume-cartographer](https://github.com/educelab/volume-cartographer):
```shell
vc_packager -v volpkgs/001_IBW_10um_60kV_MS.volpkg --name 001_IBW_10um_60kV_MS -m 140 -s Reconstructions/001/001_IBW_10um_60kV_MS_Rec_ROI/001_IBW_10um_60kV_MS__rec.log -n roi -u 10
```
- Open VC and do the segmentation:
- For slices 0-5, use LRPS with window size of “15”
- Then from 5-1509, use window size of “5”
- After segmentation is done, do the rendering in the following way:
    - Save the `*.obj` file (it will have a `*tif`, a `*.mtl` and a `*.obj` file)
    - Save the `*.ppm` file
    - Save the composites:
        - Max
        - Average/Mean
        - Median
    - Layers
    - Find and save the corresponding optical image
- After rendering is done, register the optical image with the max-composite render
- The final data for ML model training and data release should look like this in file structure:

    ```shell
    IlfordBW
    ├── ID
        ├── layers          # contains `ID_00.png`, `ID_01.png`, …, `ID_13.png`
        ├── match           # contains `ID_optical.jpg`, `ID_register.tif`
        ├── obj             # contains `ID.obj`, `ID.mtl` and `ID.tif`
        ├── ppm             # contains `ID_….tif`, `*.png`, `*.ppm`
        └── render          # contains `ID_max.tif`, `ID_mean.tif`, `ID_median.tif`
    ```
- A shell script should be made to create the above directory

## Preliminary analysis

As a preliminary analysis of the data, we see how a slice looks. Then label the slices to understand the film structure better and what information does the scan convey.

### Viewing a slice

The first slice for the film ID:`001` looks like this:
<div style="text-align:center"><img src="media/images/001_slice000_labelled.jpg" /></div>

The image is labelled. The white part of the film is the emulsion, which is responsible for the images formed after developing the film negative. The rest black part is air. So, in CT-scan, the darker part is air and the brighter part is the one where X-ray has been absorbed.

### Analysis of a slice

Now, as of analysis, let us measure the thickness of the film and the thickness of the emulsion.
<div style="text-align:center"><img src="media/images/001_slice000_crop_labelled.jpg" /></div>

From the image we can see that the thickness of the film is _~140&mu;m_ and the emulsion is _~40&mu;m_. So, _~28.5%_ of the thickness of the film is emulsion.

## Writing the script

Now, that we have scanned 15 frames, `ID` starting from `001` all the way upto `015`, let's start writing the shell script. We need to have two scripts, one for making the `*.volpkg` and the other for rendering. The second one should be a pipeline. So, let's start with the first script, and name it `packager.sh`. Let's name the second script as `render.sh`.

### Script 1: Making the `*.volpkg`

The name or the script should be `packager.sh`. For making the `*.volpkg`, we need to use the command `vc_packager` from [`volume-cartographer`](https://github.com/educelab/volume-cartographer). The `vc_packager` has the following arguments:
```
Usage:

Options:
  -h [ --help ]                    Show this message
  -v [ --volpkg ] arg              Path for the output volume package.

Volpkg metadata:
  --name arg                       Set a descriptive name for the VolumePkg. 
                                   Default: Filename specified by --volpkg
  -m [ --material-thickness ] arg  Estimated thickness of a material layer (in 
                                   microns). Required when making a new volume 
                                   package.

Volume:
  -s [ --slices ] arg              Path to input slice data. Ends with prefix 
                                   of slice images or log file path. Required 
                                   when making a new volume. If specified 
                                   multiple times, volume options will be 
                                   associated with the previous slices.
  -n [ --volume-name ] arg         Descriptive name for the volume. Required 
                                   when making a new volume.
  -u [ --voxel-size-um ] arg       Voxel size of the volume in microns (e.g. 
                                   13.546). Required when making a new volume.
  -f [ --flip ] arg                Flip options: Vertical flip (vf), horizontal
                                   flip (hf), both, z-flip (zf), all, [none].
  -c [ --compress ]                Compress slice images
```

Our recon folder structure is like this:
```
└── Reconstructions
    ├── 001
    │   └──001_IBW_10um_60kV_MS_Rec_ROI
    │       ├── 001_IBW_10um_60kV_MS__rec.log
    │       ├── 001_IBW_10um_60kV_MS__rec00000218.tif
    │       ├── ...
    │       └── ...
    └── ...
```

Considering this as a standard file structure for our work, we want to write a script that parses it, reads `001`, take a file-subscript `_IBW_10um_60kV_MS`, a recon-subscript `[file-subscript]_Rec_ROI` and then figure out that the `*.log` file is actually `[ID]/[ID][recon-subscript]/[file-subscript]__rec.log`. Then it should take the other arguments like for `--name` , `--material-thickness`, `--volume-name`, `--voxel-size-um`.
The shell script should have the following arguments (mandetory):

- `reconstruction-directory`
- `volpkg-directory`
- `volpkg-subscript`
- `recon-subscript`
- `name`
- `material-thickness`
- `volume-name`
- `voxel-size-um`

But, for now the easiest thing to do is:

- Take in two arguments `reconstruction-directory`, and `volpkg-directory`
- Iterate the directory and read the `ID`
- The replace the `ID` in
```shell
vc_packager -v [volpkg-directory]/[ID]_IBW_10um_60kV_MS.volpkg --name ID_IBW_10um_60kV_MS -m 140 -s [reconstruction-directory]/[ID]/[ID]_IBW_10um_60kV_MS_Rec_ROI/[ID]_IBW_10um_60kV_MS__rec.log -n roi -u 10
```
- In order to see the arguments under `packager.sh`run:
```shell
data/scripts/packager.sh --help
```
- In order to generate `*.volpkg` packages, run:
```shell
data/scripts/packager.sh --reconstruction-directory /Volumes/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/Reconstructions --volpkg-directory /Volumes/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/volpkgs
```
### Post segmentation

After the segmentations are completed, now it is time to analyse them and make the dataset. In order to analyze them, we want to put the rendered image and the optical image side-by-side for comparison, over [here](README.md).

In order to do that, we also need to reduce the size of the image, as it will be hard for GitHub to store all images, averaging to 23MB of sizes. So, we use the following command to reduce from 20MB to ~2MB:
```shell
magick mogrify -path ../optical_reduced/ -quality 25 *.jpg
```

With the initial scan, we have scanned 15 frames. The FOV of the scan consisted of the middle part of the whole film. So, compared to the digital image, the rendering is smaller. Let's also name the frames close to the content of the picture.

Also we need to register the images. For registration, we need to keep the X-ray rendered image fixed, and make the digital image moving. We use the [registration-toolkit](https://gitlab.com/educelab/registration-toolkit/-/tree/seam-flattening?ref_type=heads). Use branch [seam-flattening](https://gitlab.com/educelab/registration-toolkit/-/tree/seam-flattening?ref_type=heads). And then run these two commands:
```shell
git clone git@gitlab.com:educelab/registration-toolkit.git
cd registration-toolkit
git checkout seam-flattening
cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Release
cmake --build build/
```
Then run `rt_register` from `../registration-toolkit/build/bin/rt_register`.

For registration, the following command needs to be run in shell/terminal:
```shell
/path/to/rt_register -m /path/to/optical_image.ext -f /path/to/xray_render.ext -o /path/to/saved_registered_image.ext
```

For `.ext`, it is better to use a `.tif`, so that it is lossless value.

#### Film_001: Hardymon parking lot
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/001_render.jpg)  |  ![digital](media/images/optical_reduced/001.jpg)

Now, let's register. The command that I ran is:
```shell
/Volumes/Working_4TB/utils/registration-toolkit/build/bin/rt_register -m optical/001.jpg -f xray_render/edited/tif/001_render.tif -o registered/001_reg.tif
```

After registration, as `*.tif`, we convert that into `*.jpg` to make it lightweighted to attach the files to a document, using the following command from [imagemagick-mogrify](https://imagemagick.org/script/mogrify.php):
```shell
mogrify -path jpeg/ -quality 25 -format jpg tifs/*.tif
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/001_render.jpg)  |  ![digital](media/images/registered/jpgs/001_reg.jpg)

Now, to check how good the registration is, let's use some metric to define. One of the metrics is Error. 

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_001.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_001.jpg) 


#### Film_002: Ankan in Hardymon
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/002_render.jpg)  |  ![digital](media/images/optical_reduced/002.jpg)

Now, let's register. The command that I ran is:
```shell
/Volumes/Working_4TB/utils/registration-toolkit/build/bin/rt_register -m optical/002.jpg -f xray_render/edited/tif/002_render.tif -o registered/002_reg.tif
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/002_render.jpg)  |  ![digital](media/images/registered/jpgs/002_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_002.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_002.jpg) 

#### Film_003: Mystery hands
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/003_render.jpg)  |  ![digital](media/images/optical_reduced/003.jpg)

Now, let's register. The command that I ran is:
```shell
/Volumes/Working_4TB/utils/registration-toolkit/build/bin/rt_register -m optical/003.jpg -f xray_render/edited/tif/003_render.tif -o registered/003_reg.tif
```

It is prone to error to run these commands, by changing the ids. Let's make a shell script `register.sh`. For now let's keep everything as static and replace the `id`.Let's make the script. The shell script is ready [here](scripts/register.sh). Let's try it out. The script arguments are:
```shell
ankan@Ankans-MacBook-Air data % scripts/register.sh --help  
Unknown parameter: --help
Usage: scripts/register.sh --id <film-id>

Arguments:
  --id   Film id
Example:
  scripts/register.sh --id 001
```

Now, let's execute it with the command:
```shell
scripts/register.sh --id 003
```
And it works.

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/003_render.jpg)  |  ![digital](media/images/registered/jpgs/003_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_003.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_003.jpg) 

#### Film_004: Hardymon arch window
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/004_render.jpg)  |  ![digital](media/images/optical_reduced/004.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 004
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/004_render.jpg)  |  ![digital](media/images/registered/jpgs/004_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_004.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_004.jpg) 

#### Film_005: Prakash working
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/005_render.jpg)  |  ![digital](media/images/optical_reduced/005.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 005
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/005_render.jpg)  |  ![digital](media/images/registered/jpgs/005_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_005.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_005.jpg) 

#### Film_006: Lab of Prakash
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/006_render.jpg)  |  ![digital](media/images/optical_reduced/006.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 006
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/006_render.jpg)  |  ![digital](media/images/registered/jpgs/006_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_006.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_006.jpg) 

#### Film_007: Marksbury Ankan's office
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/007_render.jpg)  |  ![digital](media/images/optical_reduced/007.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 007
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/007_render.jpg)  |  ![digital](media/images/registered/jpgs/007_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_007.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_007.jpg) 

#### Film_008: Parking lot from lab of Prakash
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/008_render.jpg)  |  ![digital](media/images/optical_reduced/008.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 008
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/008_render.jpg)  |  ![digital](media/images/registered/jpgs/008_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_008.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_008.jpg) 

#### Film_009: Marksbury parking lot
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/009_render.jpg)  |  ![digital](media/images/optical_reduced/009.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 009
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/009_render.jpg)  |  ![digital](media/images/registered/jpgs/009_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_009.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_009.jpg) 

#### Film_010: Stephen's plants
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/010_render.jpg)  |  ![digital](media/images/optical_reduced/010.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 010
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/010_render.jpg)  |  ![digital](media/images/registered/jpgs/010_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_010.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_010.jpg) 

#### Film_011: MarksMarksbury Silvestri lab espresso machine
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/011_render.jpg)  |  ![digital](media/images/optical_reduced/011.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 011
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/011_render.jpg)  |  ![digital](media/images/registered/jpgs/011_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_011.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_011.jpg) 

#### Film_012: SKYSCAN 1273
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/012_render.jpg)  |  ![digital](media/images/optical_reduced/012.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 012
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/012_render.jpg)  |  ![digital](media/images/registered/jpgs/012_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_012.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_012.jpg) 

#### Film_013: Marksbury parking lot II
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/013_render.jpg)  |  ![digital](media/images/optical_reduced/013.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 013
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/013_render.jpg)  |  ![digital](media/images/registered/jpgs/013_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_013.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_013.jpg) 

#### Film_014: Marksbury Ankan's office II
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/014_render.jpg)  |  ![digital](media/images/optical_reduced/014.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 014
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/014_render.jpg)  |  ![digital](media/images/registered/jpgs/014_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_014.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_014.jpg) 

#### Film_015: Car in front of Marksbury
Rendered from CT-Scan            |  Digital image
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/015_render.jpg)  |  ![digital](media/images/optical_reduced/015.jpg)

Let's run the [registration script](scripts/register.sh) here:
```shell
scripts/register.sh --id 015
```

After registration, it looks like this:
Rendered from CT-Scan            |  Digital image (registered)
:-------------------------:|:-------------------------:
![xray](media/images/volpkg_renderings/jpgs/015_render.jpg)  |  ![digital](media/images/registered/jpgs/015_reg.jpg)

Let us use [this script](scripts/difference.sh) to calculate the difference.
Optical - Rendered            |  Rendered - Optical
:-------------------------:|:-------------------------:
![o_r](media/images/registration-metric/error/optical_minus_rendered_015.jpg)  |  ![r_o](media/images/registration-metric/error/rendered_minus_optical_015.jpg) 

### Analysis

In order to analyze the registration of the images, we can do a visualization and a quantitative analysis.

#### Quantitative

As of quantitative analysis, we can calculate the Peak Signal to Noise Ratio (PSNR), and Structural Similarity Metric (SSIM).

#### Visualization

As of visualization, we can calculate the difference between the two images. The way to do it is as follows:

- Normalize the X-ray renderings and the registered optical images to [0,1], based on minimum/maximum values of each image.
- Calculate the difference between the two images

The best way to do this is create a python script, that takes the following arguments:
- `--registered-path`: Path for registered images
- `--xray-path`: Path for X-ray images
- `--normalization (=0)`: =0 (Default) for min-max, =1 for  dynamic-range
- `--output`: Output path for saving the differences
- `--help`: Help options for knowing the list of arguments

_Coding standard_

- Create a folder named `metrics` under [scripts folder](scripts/)
- Under this `metrics` folder, create a `metric.py` file


Later, whene these are done, just make a package in a folder called utils. Under the utils, have a folder named `data`. Under that make a folder named `metrics`, and then use this `metric.py`

### Making the data

In order to make the dataset, we have to write a script (possibly, a `*.sh` script). The script is supposed to take in a `*.volpkg`, and generate a file structure for a single frame:

```shell
ID
├── layers/          # contains `ID_00.png`, `ID_01.png`, …, `ID_13.png`
├── match/           # contains `ID_optical.jpg`, `ID_register.tif`
├── obj/             # contains `ID.obj`, `ID.mtl` and `ID.tif`
├── ppm/             # contains `ID_….tif`, `*.png`, `*.ppm`
└── render/          # contains `ID_max.tif`, `ID_mean.tif`, `ID_median.tif`
```

To generate the `*.obj` and `*.ppm`, we use the following command:
```shell
vc_render -v /path/to/volpkg --volume < vol-id > -s < seg-id > -o /path/to/obj/< ID >.obj --output-ppm /path/to/ppm
```

Now, that we have the `*.obj`, and the `*.ppm` files, we can generate the `layers` using the following command:
```shell
vc_layers_from_ppm -v /path/to/volpkg -p /path/to/ppm -o /path/to/layers -f tif
```

Now, that we have all the `*.ppm`, and `*.obj`, and the `layers`, let's generate different texture.

**Generating the max filter texture**

This is the default filter in `volume-cartographer`.
```shell
vc_render -v /path/to/volpkg --volume < vol-id > -s < seg-id > -o /path/to/render/< ID >_max.tif -f 1
```

**Generating the median filter texture**

```shell
vc_render -v /path/to/volpkg --volume < vol-id > -s < seg-id > -o /path/to/render/< ID >_median.tif -f 2
```

**Generating the average/mean filter texture**

```shell
vc_render -v /path/to/volpkg --volume < vol-id > -s < seg-id > -o /path/to/render/< ID >_avg.tif -f 3
```

While making this `*.sh` script, it is necessary to refer the orientation and the volume mapping [here](https://docs.google.com/spreadsheets/d/17_nQrxLgzphYkBBVmUhER5XiKr3EFp-bIvN7qhNYXh4/edit?gid=0#gid=0).


So, the following arguments are needed to run the shell script:

- `--volpkg`  Path to the `*.volpkg`
- `--config`  Path to a config file that helps to run all the commands. Refer [here](https://docs.google.com/spreadsheets/d/17_nQrxLgzphYkBBVmUhER5XiKr3EFp-bIvN7qhNYXh4/edit?gid=0#gid=0)
- `--output`  Path where the dataset would eventually end up
- `--optical` Path to the optical images
- `--help`    Show help for the shell script

Register the images between optical and x-ray composite manually.

# Analysis of the data

## Principal Component Analysis (PCA)

The analysis of the data has to be done to understand what we are getting into. so, first let's do a principle component analysis (PCA) over all the layers. Then we compare it with the registered optical image and the different filters.

The script for doing the PCA in batch is [here](scripts/run_pca.sh). The way to run it is:
```shell
data/scripts/run_pca.sh -i /Volumes/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/dataset/all_renamed/L15/
```
### Film 001: PCA with max-filter
PCA_000            |  Max filter
:-------------------------:|:-------------------------:
![xray](media/images/pca/001/pca_000_enhanced.jpg)  |  ![digital](media/images/pca/001/001_max_enhanced.jpg)

The folds we see on the max-filter on the left side of the image is very less prominent in the first component of the PCA.

So, the mapping between the first principal component and the optical image is like this:
PCA_000            |  Optical image
:-------------------------:|:-------------------------:
![xray](media/images/pca/001/pca_000_enhanced.jpg)  |  ![digital](media/images/pca/001/001_registered.jpg)

So, the mapping between the max-filter composite and the optical image is like this:
Max filter            |  Optical image
:-------------------------:|:-------------------------:
![xray](media/images/pca/001/001_max_enhanced.jpg)  |  ![digital](media/images/pca/001/001_registered.jpg)

## Histogram

Let's create a histogram seperately for `PCA_000`, `[id]_max` and `[id]_registered`. Let's use [NumPy histogram](https://numpy.org/devdocs/reference/generated/numpy.histogram.html).

To create the histogram, we can make a structure first. Then we can just call that for plotting for every frame in the datset. For example, for an `id`, say `001`, we can have the following structure:
```javascript
{
  '001':{
    'layers':{
      'layer_0':{
        'name':'layer_0',
        'image':'/path/to/image',
        'histogram':{
          '8-bit':[...],
          '12-bit':[...],
          '16-bit':[...]
        }
      },
      'layer_1':{
        'name':'layer_1',
        'image':'/path/to/image',
        'histogram':{
          '8-bit':[...],
          '12-bit':[...],
          '16-bit':[...]
        }
      },...
    },
    'filters':{
      'max':{
        'name':'max_filter',
        'image':'/path/to/image',
        'histogram':{
          '8-bit':[...],
          '12-bit':[...],
          '16-bit':[...]
        }
      },
      'avg':{
        'name':'avg_filter',
        'image':'/path/to/image',
        'histogram':{
          '8-bit':[...],
          '12-bit':[...],
          '16-bit':[...]
        }
      },
      'median':{
        'name':'median_filter',
        'image':'/path/to/image',
        'histogram':{
          '8-bit':[...],
          '12-bit':[...],
          '16-bit':[...]
        }
      }
    },
    'pca':{
      'pca_000':{
        'name':'pca_000',
        'image':'/path/to/image',
        'histogram':{
          '8-bit':[...],
          '12-bit':[...],
          '16-bit':[...]
        }
      },
      'pca_001':{
        'name':'pca_001',
        'image':'/path/to/image',
        'histogram':{
          '8-bit':[...],
          '12-bit':[...],
          '16-bit':[...]
        }
      },...
    },
    'optical':{
      'registered':{
        'name':'optical_registered',
        'image':'/path/to/image',
        'histogram':{
          '8-bit':[...],
          '12-bit':[...],
          '16-bit':[...]
        }
      }
    }
  },...
}
```

So, basically the building block for the histogram to the function should be:
```javascript
{
  'name':'optical_registered',
  'image':'/path/to/image',
  'histogram':{
    '8-bit':[...],
    '12-bit':[...],
    '16-bit':[...]
  }
}
```

Once a function recieves this dictionary, it should be able to print the histogram. So, the first task will be to create a histogram from an image. Then we need to have afucntion that will generate the histogram(s), too. Once that is done, the `pathlib` can come into action and perform the magic. All these will be in [histogram.py](scripts/python/histogram.py).