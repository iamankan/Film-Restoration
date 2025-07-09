## Automated segmentation

The code that is working is:

```shell
python3 segmentation/scripts/segment_vcps_v2.py --volpkg /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26858.volpkg --volume 20250213145507 --slice-name 1000.tif --output-dir /localdisk0/moma-seg-2 --min-connected-points 1000 --connectivity 8 --gaussian-kernel 1 --thinning-algorithm 1 -t 2 --num-connected-components 1000
```
The major problem that is happening is the fused layer. So, we make a new process that tries to remove fused regions from the componentMasks, and then run the algorithms.

Layer fusion:
```shell
python3 segmentation/scripts/fused_vcps.py --volpkg /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26858.volpkg --volume 20250213145507 --slice-name 1000.tif --output-dir /localdisk0/moma-seg-fused --min-connected-points 1000 --connectivity 4 --gaussian-kernel 5 --thinning-algorithm 1 -t 2 --num-connected-components 1000 --junction-angle 160 --junction-window 15 --junction-window-min 5
```

```shell
python3 segmentation/scripts/fused_vcps.py --volpkg /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26858.volpkg --volume 20250213145507 --slice-name 1000.tif --output-dir /localdisk0/moma-seg-fused --min-connected-points 1000 --connectivity 4 --gaussian-kernel 5 --thinning-algorithm 1 -t 2 --num-connected-components 1000 --junction-angle 160 --junction-window 15 --junction-window-min 5 --skip 2
```

```shell
python3 segmentation/scripts/frangi_gaussian_vcps_nx_dijkstra.py --volpkg /Volumes/T7_Blue/MoMA/VolPkgs/W26861.volpkg --volume 20250214141357 --slice-name 1000.tif -t 2 -k 1 -n 1000 -s 1000 -o /Volumes/Working_4TB/MoMA/nx_output_frangi_gaussian_3/W26861 --gaussian-kernel 9 --intensity-alpha 2 --mask-thickness 2 --frangi-sigma-min 2 --frangi-sigma-max 14 --frangi-sigma-step 2 --frangi-alpha 0.05 --frangi-beta 0.5 --frangi-gamma 1 --trim-val 2
```

```shell
python3 segmentation/scripts/frangi_gaussian_vcps_nx_dijkstra.py --volpkg /Volumes/T7_Blue/MoMA/VolPkgs/W26865.volpkg --volume 20250214135720 --slice-name 1000.tif -t 2 -k 1 -n 1000 -s 1000 -o /Volumes/Working_4TB/MoMA/nx_output_frangi_gaussian_3/W26865 --gaussian-kernel 11 --intensity-alpha 2 --mask-thickness 2 --frangi-sigma-min 4 --frangi-sigma-max 10 --frangi-sigma-step 1 --frangi-alpha 0.5 --frangi-beta 0.5 --frangi-gamma 1 --trim-val 2
```

```shell
python3 segmentation/scripts/frangi_gaussian_vcps_nx_dijkstra.py --volpkg /Volumes/T7_Blue/MoMA/VolPkgs/W26865.volpkg --volume 20250214135720 --slice-name 1000.tif -t 2 -k 1 -n 1000 -s 1000 -o /Volumes/Working_4TB/MoMA/nx_output_frangi_gaussian_3/W26865 --gaussian-kernel 11 --intensity-alpha 2 --mask-thickness 2 --frangi-sigma-min 4 --frangi-sigma-max 12 --frangi-sigma-step 1 --frangi-alpha 0.1 --frangi-beta 0.5 --frangi-gamma 1 --trim-val 2
```

```shell
python3 segmentation/scripts/frangi_gaussian_vcps_nx_dijkstra.py --volpkg /Volumes/T7_Blue/MoMA/VolPkgs/W26866.volpkg --volume 20250214141133 --slice-name 1000.tif -t 2 -k 1 -n 1000 -s 1000 -o /Volumes/Working_4TB/MoMA/nx_output_frangi_gaussian_3/W26866 --gaussian-kernel 1 --intensity-alpha 5 --mask-thickness 2 --frangi-sigma-min 2 --frangi-sigma-max 12 --frangi-sigma-step 2 --frangi-alpha 0.05 --frangi-beta 0.5 --frangi-gamma 2 --trim-val 2
```

```shell
python3 segmentation/scripts/frangi_gaussian_vcps_nx_dijkstra.py --volpkg /Volumes/T7_Blue/MoMA/VolPkgs/W26858.volpkg --volume 20250213145507 --slice-name 0400.tif -t 2 -k 1 -n 500 -s 500 -o /Volumes/Working_4TB/MoMA/nx_output_frangi_gaussian_3/W26858 --gaussian-kernel 3 --intensity-alpha 5 --mask-thickness 2 --frangi-sigma-min 2 --frangi-sigma-max 12 --frangi-sigma-step 2 --frangi-alpha 0.05 --frangi-beta 0.5 --frangi-gamma 2 --trim-val 2
```

```shell
python3 segmentation/scripts/frangi_gaussian_vcps_nx_dijkstra.py --volpkg /Volumes/T7_Blue/MoMA/VolPkgs/W26868.volpkg --volume 20250117152332 --slice-name 0400.tif -t 4 -k 1 -n 100 -s 100 -o /Volumes/Working_4TB/MoMA/nx_output_frangi_gaussian_3/W26868 --gaussian-kernel 1 --intensity-alpha 5 --mask-thickness 2 --frangi-sigma-min 2 --frangi-sigma-max 14 --frangi-sigma-step 2 --frangi-alpha 0.5 --frangi-beta 0.5 --frangi-gamma 0.5 --trim-val 2
```