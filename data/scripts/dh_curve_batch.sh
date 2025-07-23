#!/bin/bash

# Help function
print_help() {
    echo "Usage: $0 --dataset-dir PATH --roi-json PATH --output-dir PATH"
    echo
    echo "Arguments:"
    echo "  --dataset-dir     Root directory containing 'train', 'val', 'test' subfolders."
    echo "  --roi-json        Path to the ROI JSON file."
    echo "  --output-dir      Output directory to store the dh_curve results."
    echo
    echo "Example:"
    echo "  $0 --dataset-dir /path/to/data --roi-json /path/to/roi.json --output-dir /localdisk0/dh_results"
    exit 1
}

# Default values
DATASET_DIR=""
ROI_JSON=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --roi-json)
            ROI_JSON="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            ;;
    esac
done

# Validate inputs
if [[ -z "$DATASET_DIR" || -z "$ROI_JSON" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required arguments."
    print_help
fi

# Iterate through train, val, test folders
for SPLIT in train val test; do
    SPLIT_DIR="$DATASET_DIR/$SPLIT"
    if [[ ! -d "$SPLIT_DIR" ]]; then
        echo "Warning: $SPLIT_DIR does not exist. Skipping..."
        continue
    fi

    for FILM_ID in "$SPLIT_DIR"/*; do
        ID=$(basename "$FILM_ID")
        INPUT_CT="$SPLIT_DIR/$ID/layers/07.png"
        INPUT_OPTICAL="$SPLIT_DIR/$ID/match/${ID}_reg.tif"
        OUTPUT_PATH="$OUTPUT_DIR/dh_curve_${ID}"
        TITLE="Film-${ID}"

        if [[ -f "$INPUT_CT" && -f "$INPUT_OPTICAL" ]]; then
            echo "Running for ID: $ID"
            python3 data/scripts/python/dh_curve.py \
                --input-ct-image "$INPUT_CT" \
                --input-optical-image "$INPUT_OPTICAL" \
                --k 1 --A 1 \
                --output "$OUTPUT_PATH" \
                --roi "$ROI_JSON" \
                --roi-key "$ID" \
                --title "$TITLE"
        else
            echo "Warning: Missing CT or optical image for ID $ID. Skipping..."
        fi
    done
done
