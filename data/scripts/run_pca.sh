#!/bin/bash

print_help() {
  echo ""
  echo "Usage: $0 -i <input_folder> [-p <pca_path>]"
  echo ""
  echo "Description:"
  echo "  Iterates through each subdirectory (assumed to be sample IDs) inside the input folder"
  echo "  and runs PCA on the 'layers' directory of each ID. The results are saved in a 'pca' subfolder."
  echo ""
  echo "  If --pca-path is provided:"
  echo "    - PCA will be loaded if it exists"
  echo "    - PCA will be trained and saved if it does not exist"
  echo ""
  echo "Required Arguments:"
  echo "  -i    Path to the input folder containing subfolders named by ID"
  echo ""
  echo "Optional Arguments:"
  echo "  -p    Path to PCA model (.joblib) to reuse or create"
  echo "  -h, --help    Show this help message and exit"
  echo ""
  echo "Example:"
  echo "  $0 -i /path/to/data -p /path/to/pca_trained.joblib"
  echo ""
  exit 0
}

# Defaults
PCA_PATH=""

# Manually check for --help
for arg in "$@"; do
  if [[ "$arg" == "--help" ]]; then
    print_help
  fi
done

# Parse options
while getopts "i:p:h" opt; do
  case ${opt} in
    i )
      INPUT_FOLDER="$OPTARG"
      ;;
    p )
      PCA_PATH="$OPTARG"
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
  print_help
fi

# Check if the input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
  echo "Error: Directory '$INPUT_FOLDER' does not exist."
  exit 1
fi

# Iterate through subdirectories (assumed to be IDs)
for id_dir in "$INPUT_FOLDER"/*; do
  if [ -d "$id_dir" ]; then
    id=$(basename "$id_dir")
    input_path="$id_dir/layers"
    output_path="$id_dir/pca"

    echo "Processing ID: $id"

    if [ -n "$PCA_PATH" ]; then
      python3 data/scripts/python/pca.py \
        -i "$input_path" \
        -o "$output_path" \
        -p "$PCA_PATH"
    else
      echo "without pca-path"
      python3 data/scripts/python/pca.py \
        -i "$input_path" \
        -o "$output_path"
    fi
  fi
done
