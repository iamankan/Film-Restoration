#!/bin/bash

show_help() {
    echo "Usage: $0 -i <parent_folder> -l <lightfield_folder> -d <darkfield_folder>"
    echo
    echo "Options:"
    echo "  -i DIR     Input parent directory containing subfolders with tiff files converted from raw"
    echo "  -l DIR    Directory for lightfield"
    echo "  -d DIR    Directory for darkfield"

    echo "  -h        Show this help message and exit"
}

# Parse arguments
while getopts "i:l:d:h" opt; do
  case $opt in
    i )
      PARENT_DIR=$OPTARG
      ;;
    l )
      LF=$OPTARG
      ;;
    d )
      DF=$OPTARG
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
if [ -z "$PARENT_DIR" ] || [ -z "$LF" ] || [ -z "$DF" ]; then
    echo "Error: Parent directory and light/dark field directory must be specified."
    show_help
    exit 1
fi

# Check if directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: '$PARENT_DIR' is not a valid directory."
    exit 1
fi

if [ ! -d "$LF" ]; then
    echo "Error: '$LF' is not a valid directory."
    exit 1
fi

if [ ! -d "$DF" ]; then
    echo "Error: '$DF' is not a valid directory."
    exit 1
fi

# Iterate over subfolders
for subfolder in "$PARENT_DIR"/*/; do
    # Remove trailing slash
    subfolder=${subfolder%/}

    if [ -d "$subfolder" ]; then
	      subfolder_name=$(basename "$subfolder")
        echo "Processing subfolder: $subfolder (using base name: $subfolder_name)"

        python3 flat_field_correction.py --input-image-directory $subfolder/tiff \
        --input-darkfield-directory $DF --input-lightfield-directory $LF \
        --output-flatfield-directory $subfolder/ffc

        python3 hdr_stacking.py --exposures-dir $subfolder/ffc --output-dir $subfolder/hdr --dark-level 0 \
        --raw-bit 16 --normalized --debayered --prefix-name $subfolder_name

    fi
done

