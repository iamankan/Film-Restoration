#!/bin/bash

# Default parameters
THRESHOLD=2
NUM_CLUSTERS=1
NUM_SEG_POINTS=1000
TOTAL_SEG_POINTS=1000
GAUSSIAN_KERNEL=1
INTENSITY_ALPHA=3
MASK_THICKNESS=2
SLICE_NAME="0005.tif"

print_help() {
    echo "Usage: $0 --input-dir <path> --output-dir <path> [--help]"
    echo ""
    echo "Required:"
    echo "  --input-dir        Path to the directory containing *.volpkg folders"
    echo "  --output-dir       Path to the base output directory"
    echo ""
    echo "Optional (editable inside script):"
    echo "  --help             Show this help message and exit"
    echo ""
    echo "Fixed parameters:"
    echo "  -t $THRESHOLD  -k $NUM_CLUSTERS  -n $NUM_SEG_POINTS  -s $TOTAL_SEG_POINTS"
    echo "  --gaussian-kernel $GAUSSIAN_KERNEL"
    echo "  --intensity-alpha $INTENSITY_ALPHA"
    echo "  --mask-thickness $MASK_THICKNESS"
    echo "  --slice-name $SLICE_NAME"
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input-dir) INPUT_DIR="$2"; shift ;;
        --output-dir) OUTPUT_DIR_BASE="$2"; shift ;;
        --help) print_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; print_help; exit 1 ;;
    esac
    shift
done

# Check required args
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR_BASE" ]; then
    echo "Error: --input-dir and --output-dir are required."
    echo ""
    print_help
    exit 1
fi

# Ensure base output folder exists
mkdir -p "$OUTPUT_DIR_BASE"

# Iterate over each volpkg
for volpkg_path in "$INPUT_DIR"/*.volpkg; do
    if [ -d "$volpkg_path" ]; then
        volpkg_name=$(basename "$volpkg_path" .volpkg)
        echo "🔍 Processing $volpkg_name"

        VOLUME_DIR="$volpkg_path/volumes"
        if [ -d "$VOLUME_DIR" ]; then
            volume_id=$(ls "$VOLUME_DIR" | head -n 1)
            if [ -z "$volume_id" ]; then
                echo "⚠️  Warning: No volume found in $VOLUME_DIR"
                continue
            fi
        else
            echo "⚠️  Warning: volumes/ directory missing in $volpkg_path"
            continue
        fi

        # Prepare output folder
        OUTPUT_DIR="$OUTPUT_DIR_BASE/${volpkg_name}_${volume_id}"
        mkdir -p "$OUTPUT_DIR"

        # Track execution time
        start_time=$(date +%s)

        python3 segmentation/scripts/kmeans_vcps_nx_dijkstra.py \
            --volpkg "$volpkg_path" \
            --volume "$volume_id" \
            --slice-name "$SLICE_NAME" \
            -t "$THRESHOLD" \
            -k "$NUM_CLUSTERS" \
            -n "$NUM_SEG_POINTS" \
            -s "$TOTAL_SEG_POINTS" \
            -o "$OUTPUT_DIR" \
            --gaussian-kernel "$GAUSSIAN_KERNEL" \
            --intensity-alpha "$INTENSITY_ALPHA" \
            --mask-thickness "$MASK_THICKNESS"


        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✅ Finished ${volpkg_name} (${volume_id}) in ${duration} seconds."
        echo ""
    fi
done
