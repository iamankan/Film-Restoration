#!/bin/bash

# Common variables
volpkg="/media/ankan/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/volpkgs_test/007_IBW_10um_60kV_MS.volpkg"
volume="20250416174927"
threshold_factor=2
num_clusters=1
num_seg_points=100
total_seg_points=100
intensity_alpha=5
mask_thickness=2
gaussian_kernel=3

min_seg=100
max_seg=200
seg_interval=100

min_k=1
max_k=5
k_interval=2

# Base output paths
base_out_astar="/localdisk0/thesis-images/nx-vol/ilford-kmeans-nx-astar-all_vols"
base_out_dijkstra="/localdisk0/thesis-images/nx-vol/ilford-kmeans-nx-dijkstra-all_vols"

dataset_name="IBW-007"

# Path to slices folder
slices_dir="${volpkg}/volumes/${volume}"

# Loop over each .tif slice in the slices directory
for slice_path in "$slices_dir"/*.tif; do
  slice_name=$(basename "$slice_path")            # e.g., 0000.tif
  slice_id="S${slice_name%.*}"                     # e.g., S0000

  echo "Processing slice: $slice_name"

  for kernel in $(seq $min_k $k_interval $max_k); do
      echo "kernel: $kernel"

      for min_seg_points in $(seq $min_seg $seg_interval $max_seg); do

        echo "Kernel: $kernel, min-seg-points: $min_seg_points"

        # Build output folder names
        out_astar="${base_out_astar}/${dataset_name}_${slice_id}_G${kernel}_min${min_seg_points}"
        out_dijkstra="${base_out_dijkstra}/${dataset_name}_${slice_id}_G${kernel}_min${min_seg_points}"

        start_time=$(date +%s.%N)

        echo "Starting ASTAR"
        python3 segmentation/scripts/kmeans_vcps_nx_astar.py \
          --volpkg "$volpkg" \
          --volume "$volume" \
          --slice-name "$slice_name" \
          --threshold-factor "$threshold_factor" \
          --number-of-clusters "$num_clusters" \
          --num-seg-points "$min_seg_points" \
          --total-seg-points "$total_seg_points" \
          --output-folder "$out_astar" \
          --gaussian-kernel "$kernel" \
          --intensity-alpha "$intensity_alpha" \
          --mask-thickness "$mask_thickness"
        echo "Finished ASTAR"

        end_time=$(date +%s.%N)
        elapsed=$(echo "$end_time - $start_time" | bc)
        printf "✅ Finished processing Astar in %.3f seconds\n\n" "$elapsed"

        start_time=$(date +%s.%N)

        echo "Starting DIJKSTRA"
        python3 segmentation/scripts/kmeans_vcps_nx_dijkstra.py \
          --volpkg "$volpkg" \
          --volume "$volume" \
          --slice-name "$slice_name" \
          --threshold-factor "$threshold_factor" \
          --number-of-clusters "$num_clusters" \
          --num-seg-points "$min_seg_points" \
          --total-seg-points "$total_seg_points" \
          --output-folder "$out_dijkstra" \
          --gaussian-kernel "$kernel" \
          --intensity-alpha "$intensity_alpha" \
          --mask-thickness "$mask_thickness"
        echo "Finished DIJKSTRA"

        end_time=$(date +%s.%N)
        elapsed=$(echo "$end_time - $start_time" | bc)
        printf "✅ Finished processing Dijkstra in %.3f seconds\n\n" "$elapsed"

      done
    
  done

done