#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 --parent_dir DIR --k VAL --n VAL --t VAL --output_dir DIR"
    echo ""
    echo "Named Arguments:"
    echo "  --parent_dir     Path to the parent directory containing folders with volumes"
    echo "  --k              Number of clusters for KMeans"
    echo "  --n              Min number of segmentaion points"
    echo "  --t              Threshold factor (1/t)"
    echo "  --output_dir     Output directory for segmentation results"
    echo ""
    echo "Example:"
    echo "  $0 --parent_dir /media/.../VolPkgs --k 3 --n 100 --t 3 --output_dir /localdisk0/..."
}

# Default values (optional)
PARENT_DIR=""
K=""
N=""
T=""
OUTPUT_DIR=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --parent_dir)
            PARENT_DIR="$2"
            shift; shift
            ;;
        --k)
            K="$2"
            shift; shift
            ;;
        --n)
            N="$2"
            shift; shift
            ;;
        --t)
            T="$2"
            shift; shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift; shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$PARENT_DIR" || -z "$K" || -z "$N" || -z "$T" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required arguments."
    show_help
    exit 1
fi

# Main loop
for folder in "$PARENT_DIR"/*; do
    if [[ -d "$folder" && -d "$folder/volumes" ]]; then
        volume_subfolders=("$folder"/volumes/*)
        if [[ ${#volume_subfolders[@]} -gt 0 ]]; then
            first_volume="${volume_subfolders[0]}"
            input_file="$first_volume/0000.tif"
            if [[ -f "$input_file" ]]; then
                echo "Running kmeans on: $input_file"
                python3 segmentation/scripts/kmeans.py \
                    -s "$input_file" \
                    -k "$K" \
                    -n "$N" \
                    -t "$T" \
                    -o "$OUTPUT_DIR"
            else
                echo "Warning: File $input_file not found"
            fi
        else
            echo "Warning: No subfolders in $folder/volumes"
        fi
    else
        echo "Skipping: $folder (missing volumes/ directory)"
    fi
done
