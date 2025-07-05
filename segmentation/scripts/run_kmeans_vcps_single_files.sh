#!/bin/bash

# for j in {1..15..2} # Iterates from 1 to 10 with a step of 2
# do
#   echo "Kernel: $j"

#   echo "Starting ASTAR"
#   python3 segmentation/scripts/kmeans_vcps_nx_astar.py \
#   --volpkg "/media/ankan/Ankan_PhD/MoMA/VolPkgs/W26861.volpkg" \
#   --volume "20250214141357" --slice-name "1000.tif" --threshold-factor 4 --number-of-clusters 3 --num-seg-points 1000 --total-seg-points 1000 \
#   --output-folder "/localdisk0/thesis-images/MoMA-kmeans-nx-astar/MoMA_W26861_S1000_G$j" --gaussian-kernel "$j" --intensity-alpha 1
#   echo "Finished ASTAR"


#   echo "Starting DIJKSTRA"
#   python3 segmentation/scripts/kmeans_vcps_nx_dijkstra.py \
#   --volpkg "/media/ankan/Ankan_PhD/MoMA/VolPkgs/W26861.volpkg" \
#   --volume "20250214141357" --slice-name "1000.tif" --threshold-factor 4 --number-of-clusters 3 --num-seg-points 1000 --total-seg-points 1000 \
#   --output-folder "/localdisk0/thesis-images/MoMA-kmeans-nx-dijkstra/MoMA_W26861_S1000_G$j" --gaussian-kernel "$j" --intensity-alpha 1
#   echo "Finished DIJKSTRA"
# done


# Common variables
volpkg="/media/ankan/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/volpkgs_test/007_IBW_10um_60kV_MS.volpkg"
volume="20250416174927"
slice_name="0000.tif"
slice_id="S${slice_name%.*}"  # Extracts '1000' and makes 'S1000'
threshold_factor=2
num_clusters=1
num_seg_points=1000
total_seg_points=1000
intensity_alpha=5

# Base output paths
base_out_astar="/localdisk0/thesis-images/ilford-kmeans-nx-astar-redo-1"
base_out_dijkstra="/localdisk0/thesis-images/ilford-kmeans-nx-dijkstra-redo-1"

dataset_name="IBW-007"
start_k=1
end_k=15
diff_k=2

# Kernel loop
for j in $(seq $start_k $diff_k $end_k); do
  echo "Kernel: $j"

  # Build output folder names
  out_astar="${base_out_astar}/${dataset_name}_${slice_id}_G${j}"
  out_dijkstra="${base_out_dijkstra}/${dataset_name}_${slice_id}_G${j}"

  echo "Starting ASTAR"
  python3 segmentation/scripts/kmeans_vcps_nx_astar.py \
    --volpkg "$volpkg" \
    --volume "$volume" \
    --slice-name "$slice_name" \
    --threshold-factor "$threshold_factor" \
    --number-of-clusters "$num_clusters" \
    --num-seg-points "$num_seg_points" \
    --total-seg-points "$total_seg_points" \
    --output-folder "$out_astar" \
    --gaussian-kernel "$j" \
    --intensity-alpha "$intensity_alpha"
  echo "Finished ASTAR"

  echo "Starting DIJKSTRA"
  python3 segmentation/scripts/kmeans_vcps_nx_dijkstra.py \
    --volpkg "$volpkg" \
    --volume "$volume" \
    --slice-name "$slice_name" \
    --threshold-factor "$threshold_factor" \
    --number-of-clusters "$num_clusters" \
    --num-seg-points "$num_seg_points" \
    --total-seg-points "$total_seg_points" \
    --output-folder "$out_dijkstra" \
    --gaussian-kernel "$j" \
    --intensity-alpha "$intensity_alpha"
  echo "Finished DIJKSTRA"

done