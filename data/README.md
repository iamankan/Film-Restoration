# Film Restoration Data collection and archieving


This sub-repository is for data creation and data handling. This repository eill do the followings

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





