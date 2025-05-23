#!/bin/bash

print_help() {
  echo ""
  echo "Usage: $0 -i <input_folder>"
  echo ""
  echo "Description:"
  echo "  This script iterates through each subdirectory (assumed to be sample IDs) inside the input folder"
  echo "  and runs histograms on the 'layers', 'filters', 'optical image' and 'PCA' (if available) directory of each ID."
  echo "  The results are saved in a 'pca' subfolder."
  echo ""
  echo "Required Arguments:"
  echo "  -i    Path to the input folder containing subfolders named by ID"
  echo ""
  echo "Optional Flags:"
  echo "  -h, --help    Show this help message and exit"
  echo ""
  echo "Example:"
  echo "  $0 -i /path/to/data"
  echo "    Runs: python3 data/scripts/python/histogram.py -i /path/to/data/<ID> -o /path/to/data/<ID>/histogram"
  echo ""
  exit 0
}

# Manually check for --help
for arg in "$@"; do
  if [[ "$arg" == "--help" ]]; then
    print_help
  fi
done

# Parse short options
while getopts "i:h" opt; do
  case ${opt} in
    i )
      INPUT_FOLDER="$OPTARG"
      ;;
    h )
      print_help
      ;;
    * )
      echo "Error: Invalid argument."
      print_help
      ;;
  esac
done



# Check if input folder is provided
if [ -z "$INPUT_FOLDER" ]; then
  echo "Error: Input folder is required."
  usage
fi

# Check if the input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
  echo "Error: Directory '$INPUT_FOLDER' does not exist."
  exit 1
fi

# Iterate through subdirectories (assumed to be IDs)
for id_dir in "$INPUT_FOLDER"/*; do
  if [ -d "$id_dir" ] && [[ "$(basename "$id_dir")" != ._* ]]; then
    id=$(basename "$id_dir")
    input_path="$id_dir"
    output_path="$id_dir/histogram"
    echo "Processing ID: $id"
    python3 data/scripts/python/histogram.py -i "$input_path" -o "$output_path"
  fi
done