#!/bin/bash


show_help() {
    echo "Usage: $0 --volpkg PATH TO VOLPKG --config PATH TO CONFIG FILE --output OUTPUT PATH"
    echo "Description:"
    echo "  This script help make the IBW dataset from the optical image and the volpkg with its segmented outputs"
    echo
    echo "Arguments:"
    echo "  --volpkg    Path to the volpkg"
    echo "  --config    Path to the config file, that has the data-ID, the segmentatoion ID, the volume ID and the transformations needed to run volume-cartographer commands"
    echo "  --output    Output path where the dataset will be saved"
    echo "  --optical   Path to the optical/digital image"
    echo "  --help      Show help with this command"
}

# Declare variables to save the arguments
volpkg_dir=
config=
output_dir=
optical_dir=

# Parse the named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --volpkg) volpkg_dir="$2"; shift;;
        --config) config="$2"; shift;;
        --output) output_dir="$2"; shift;;
        --optical) optical_dir="$2"; shift;;
        --help) show_help; exit 0;;
        *) echo "Unknown argument $1"; show_help; exit 1;;
    esac
    shift
done

# If missing any argument, show help
if [[ -z "$volpkg_dir" || -z "$config" || -z "$output_dir" || -z "$optical_dir" ]]; then
    echo "Error: Missing required arguments."
    echo
    show_help
    exit 1
fi

# Make the directories for the data
if [ ! -d "$output_dir" ]; then
  mkdir $output_dir
fi

log_file="$output_dir"/"$(date +%s)".log

test_str="ankan bhattacharyya"

read -r -a test_args <<< "$test_str"
for arg in "${test_args[@]}"; do
  echo "$arg"
done

# Read the config file as CSV
IFS=','
tail -n +2 "$config" | while IFS=',' read -r ID Volpkg Volume Segmentation Transformation Extra; do
  echo "ID: $ID"
  echo "Volpkg: $Volpkg"
  echo "Volume: $Volume"
  echo "Segmentation: $Segmentation"
  echo "Transformation: $Transformation"

  frame_id=$ID
  volpkg_name=$Volpkg
  segmentation_id=$Segmentation
  volume_id=$Volume
  transformation_str=$Transformation

  IFS=' ' read -ra transformation_args <<< "$transformation_str"

  for arg in "${transformation_args[@]}"; do
    echo "$arg"
  done

  
  echo "Volpkg-name: $volpkg_name"

  id_dir=$output_dir/$frame_id
  layers_dir=$id_dir/layers
  match_dir=$id_dir/match
  obj_dir=$id_dir/obj
  ppm_dir=$id_dir/ppm
  render_dir=$id_dir/render

  if [ ! -d "$id_dir" ]; then
    mkdir $id_dir
    mkdir $layers_dir
    mkdir $match_dir
    mkdir $obj_dir
    mkdir $ppm_dir
    mkdir $render_dir
  fi

  # Generate the ppm and obj
  echo "Generating ppm and obj for frame-$frame_id"
  vc_render -v "$volpkg_dir"/"$volpkg_name" --volume "$volume_id" -s "$segmentation_id" -o "$obj_dir"/"$frame_id".obj --output-ppm "$ppm_dir"/"$frame_id".ppm  "${transformation_args[@]}">> "$log_file" 2>&1
  
  # Generating the layers
  echo "Generating the layers for frame-$frame_id"
  vc_layers_from_ppm -v "$volpkg_dir"/"$volpkg_name" -p "$ppm_dir"/"$frame_id".ppm -o "$layers_dir">> "$log_file" 2>&1

  # Generating max-filter texture
  echo "Generating the max-texture composite for frame-$frame_id"
  vc_render -v "$volpkg_dir"/"$volpkg_name" --volume "$volume_id" -s "$segmentation_id" -o "$render_dir"/"$frame_id"_max.tif -f 1  "${transformation_args[@]}">> "$log_file" 2>&1

  # Generating median-filter texture
  echo "Generating the median-texture composite for frame-$frame_id"
  vc_render -v "$volpkg_dir"/"$volpkg_name" --volume "$volume_id" -s "$segmentation_id" -o "$render_dir"/"$frame_id"_median.tif -f 2  "${transformation_args[@]}">> "$log_file" 2>&1

  # Generating average-filter texture
  echo "Generating the average-texture composite for frame-$frame_id"
  vc_render -v "$volpkg_dir"/"$volpkg_name" --volume "$volume_id" -s "$segmentation_id" -o "$render_dir"/"$frame_id"_avg.tif -f 3  "${transformation_args[@]}">> "$log_file" 2>&1

  # Copying the optical image
  echo "Copying the optical image"
  # cp $optical_dir/$frame_id.jpg $match_dir/"$frame_id".jpg
  cp "$optical_dir/$frame_id"* "$match_dir/"

done