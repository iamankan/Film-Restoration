#!/bin/bash

show_help() {
    echo "Usage: $0 -d <parent_folder> -e <extension>"
    echo
    echo "Options:"
    echo "  -d DIR    Parent directory containing subfolders with raw files"
    echo "  -e EXT    Extension of raw files (e.g. NEF, CR2)"
    echo "  -h        Show this help message and exit"
}

# Parse arguments
while getopts "d:e:h" opt; do
  case $opt in
    d )
      PARENT_DIR=$OPTARG
      ;;
    e )
      EXT=$OPTARG
      ;;
    h )
      show_help
      exit 0
      ;;
    \? )
      show_help
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$PARENT_DIR" ] || [ -z "$EXT" ]; then
    echo "Error: Both parent directory and extension must be specified."
    show_help
    exit 1
fi

# Check if directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: '$PARENT_DIR' is not a valid directory."
    exit 1
fi

# Iterate over subfolders
for subfolder in "$PARENT_DIR"/*/; do
    # Remove trailing slash
    subfolder=${subfolder%/}

    if [ -d "$subfolder" ]; then
	subfolder_name=$(basename "$subfolder")
        echo "Processing subfolder: $subfolder (using base name: $subfolder_name)"

        # Call raw2tiff.sh with subfolder and extension
        ./raw2tiff.sh -d "$subfolder" -e "$EXT" -n "$subfolder_name" -o tiff
    fi
done

