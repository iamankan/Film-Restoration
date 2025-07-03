#!/bin/bash

for j in {1..21..2} # Iterates from 1 to 10 with a step of 2
do
  echo "Kernel: $j"

  echo "Starting ASTAR"
  python3 segmentation/scripts/kmeans_vcps_nx_astar.py \
  --volpkg "/media/ankan/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/volpkgs_test/007_IBW_10um_60kV_MS.volpkg" \
  --volume "20250416174927" --slice-name "0000.tif" --threshold-factor 2 --number-of-clusters 1 --num-seg-points 1000 --total-seg-points 1000 \
  --output-folder "/localdisk0/thesis-images/ilford-kmeans-nx-astar/IBW-007_G$j" --gaussian-kernel "$j" --intensity-alpha 2
  echo "Finished ASTAR"


  echo "Starting DIJKSTRA"
  python3 segmentation/scripts/kmeans_vcps_nx_dijkstra.py \
  --volpkg "/media/ankan/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/volpkgs_test/007_IBW_10um_60kV_MS.volpkg" \
  --volume "20250416174927" --slice-name "0000.tif" --threshold-factor 2 --number-of-clusters 1 --num-seg-points 1000 --total-seg-points 1000 \
  --output-folder "/localdisk0/thesis-images/ilford-kmeans-nx-dijkstra/IBW-007_G$j" --gaussian-kernel "$j" --intensity-alpha 2
  echo "Finished DIJKSTRA"
done