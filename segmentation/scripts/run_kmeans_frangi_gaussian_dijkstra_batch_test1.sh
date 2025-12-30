#!/bin/bash

# Default parameters
THRESHOLD=5
NUM_CLUSTERS=1
NUM_SEG_POINTS=100
TOTAL_SEG_POINTS=100
GAUSSIAN_KERNEL=3
INTENSITY_ALPHA=5
MASK_THICKNESS=1
SLICE_NAME="0000.tif"

FRANGI_SIGMA_MIN=4
FRANGI_SIGMA_MAX=14
FRANGI_SIGMA_STEP=2
FRANGI_ALPHA=0.3
FRANGI_BETA=0.3
FRANGI_GAMMA=1

TRIM_VAL=10

WRITE_VCPS=false

print_help() {
    echo "Usage: $0 --input-dir <path> --output-dir <path> [--write-vcps] [--help]"
    echo ""
    echo "Required:"
    echo "  --input-dir        Path to the directory containing *.volpkg folders"
    echo "  --output-dir       Path to the base output directory"
    echo ""
    echo "Optional:"
    echo "  --write-vcps       Save VCPS intermediate files (adds --write-vcps to Python call)"
    echo "  --help             Show this help message and exit"
    echo ""
    echo "Fixed parameters:"
    echo "  -t $THRESHOLD  -k $NUM_CLUSTERS  -n $NUM_SEG_POINTS  -s $TOTAL_SEG_POINTS"
    echo "  --gaussian-kernel $GAUSSIAN_KERNEL"
    echo "  --intensity-alpha $INTENSITY_ALPHA"
    echo "  --mask-thickness $MASK_THICKNESS"
    echo "  --slice-name $SLICE_NAME"
    echo "  --frangi-sigma-min $FRANGI_SIGMA_MIN  --frangi-sigma-max $FRANGI_SIGMA_MAX"
    echo "  --frangi-sigma-step $FRANGI_SIGMA_STEP  --frangi-alpha $FRANGI_ALPHA"
    echo "  --frangi-beta $FRANGI_BETA  --frangi-gamma $FRANGI_GAMMA --trim-val $TRIM_VAL"
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input-dir) INPUT_DIR="$2"; shift ;;
        --output-dir) OUTPUT_DIR_BASE="$2"; shift ;;
        --write-vcps) WRITE_VCPS=true ;; 
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

# Record global start time
global_start_time=$(date +%s)

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

        OUTPUT_DIR="$OUTPUT_DIR_BASE/${volpkg_name}_${volume_id}_frangi_on_gaussian_${GAUSSIAN_KERNEL}_a${FRANGI_ALPHA}_b${FRANGI_BETA}_g${FRANGI_GAMMA}"
        mkdir -p "$OUTPUT_DIR"

        start_time=$(date +%s)

        CMD=(python3 segmentation/scripts/frangi_gaussian_vcps_nx_dijkstra.py
            --volpkg "$volpkg_path"
            --volume "$volume_id"
            --slice-name "$SLICE_NAME"
            -t "$THRESHOLD"
            -k "$NUM_CLUSTERS"
            -n "$NUM_SEG_POINTS"
            -s "$TOTAL_SEG_POINTS"
            -o "$OUTPUT_DIR"
            --gaussian-kernel "$GAUSSIAN_KERNEL"
            --intensity-alpha "$INTENSITY_ALPHA"
            --mask-thickness "$MASK_THICKNESS"
            --frangi-sigma-min "$FRANGI_SIGMA_MIN"
            --frangi-sigma-max "$FRANGI_SIGMA_MAX"
            --frangi-sigma-step "$FRANGI_SIGMA_STEP"
            --frangi-alpha "$FRANGI_ALPHA"
            --frangi-beta "$FRANGI_BETA"
            --frangi-gamma "$FRANGI_GAMMA"
            --trim-val "$TRIM_VAL"
        )

        if [ "$WRITE_VCPS" = true ]; then
            CMD+=(--write-vcps)
        fi

        echo "▶️  Running: ${CMD[@]}"
        "${CMD[@]}"

        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✅ Finished ${volpkg_name} (${volume_id}) in ${duration} seconds."
        echo ""
    fi
done

# Global end time
global_end_time=$(date +%s)
total_time=$((global_end_time - global_start_time))

echo "🎯 All processing complete in $total_time seconds."
