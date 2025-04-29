#!/bin/bash

# Default empty
rendered=""
optical=""
output=""

show_help() {
    echo "Usage: $0 --rendered RENDERED_FOLDER --optical OPTICAL_FOLDER --output OUTPUT_FOLDER"
    echo
    echo "Arguments:"
    echo "  --rendered   Path to the rendered images folder (e.g., 001_render.jpg)"
    echo "  --optical    Path to the optical images folder  (e.g., 001_reg.jpg)"
    echo "  --output     Path where the output images will be saved"
    echo "  --help       Show this help message and exit"
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rendered) rendered="$2"; shift ;;
        --optical) optical="$2"; shift ;;
        --output) output="$2"; shift ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# If missing any argument, show help
if [[ -z "$rendered" || -z "$optical" || -z "$output" ]]; then
    echo "Error: Missing required arguments."
    echo
    show_help
    exit 1
fi

# Make output directory if it doesn't exist
mkdir -p "$output"

# Process each optical image
for optical_img in "$optical"/*_reg.jpg; do
    base=$(basename "$optical_img")          # e.g., 001_reg.jpg
    id="${base%_reg.jpg}"                    # Extract '001'
    rendered_img="$rendered/${id}_render.jpg"

    if [[ -f "$rendered_img" ]]; then
        # optical - rendered
        magick "$optical_img" "$rendered_img" -compose MinusSrc -composite "$output/optical_minus_rendered_${id}.jpg"

        # rendered - optical
        magick "$rendered_img" "$optical_img" -compose MinusSrc -composite "$output/rendered_minus_optical_${id}.jpg"

        echo "Processed ID $id"
    else
        echo "Warning: No matching rendered image for ID $id"
    fi
done
