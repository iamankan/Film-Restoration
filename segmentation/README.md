## Automated segmentation

The code that is working is:

```shell
python3 segmentation/scripts/segment_vcps_v2.py --volpkg /media/ankan/Ankan_PhD/MoMA/VolPkgs/W26858.volpkg --volume 20250213145507 --slice-name 1000.tif --output-dir /localdisk0/moma-seg-2 --min-connected-points 1000 --connectivity 8 --gaussian-kernel 1 --thinning-algorithm 1 -t 2 --num-connected-components 1000
```
The major problem that is happening is the fused layer. So, we make a new process that tries to remove fused regions from the componentMasks, and then run the algorithms.