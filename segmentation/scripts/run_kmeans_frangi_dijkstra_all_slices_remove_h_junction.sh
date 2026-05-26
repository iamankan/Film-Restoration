#!/bin/bash

# Default parameters
THRESHOLD=2
NUM_CLUSTERS=1
NUM_SEG_POINTS=1000
TOTAL_SEG_POINTS=500
GAUSSIAN_KERNEL=1
INTENSITY_ALPHA=5
MASK_THICKNESS=2

START_SLICE_NAME="1000.tif"
END_SLICE_NAME="1100.tif"
STEP_SIZE=1

FRANGI_SIGMA_MIN=4
FRANGI_SIGMA_MAX=14
FRANGI_SIGMA_STEP=2
FRANGI_ALPHA=0.3
FRANGI_BETA=0.3
FRANGI_GAMMA=1
FRANGI_WEIGHT=1.0

TRIM_VAL=10

WRITE_VCPS=false

print_help() {
    echo "Usage: $0 --volpkg <path> --output-dir <path> [--write-vcps] [--help]"
    echo ""
    echo "Required:"
    echo "  --volpkg        Path to the directory containing *.volpkg folders"
    echo "  --output-dir       Path to the base output directory"
    echo "  --volume-id        Volume ID to process (e.g., 20250416174927)"
    echo "  --start-slice-name Start slice name (e.g., 1000.tif)"
    echo "  --end-slice-name   End slice name (e.g., 1100.tif)"
    echo "  --step-size        Step size for slice iteration (default: 1)"
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
    echo "  --frangi-sigma-min $FRANGI_SIGMA_MIN  --frangi-sigma-max $FRANGI_SIGMA_MAX"
    echo "  --frangi-sigma-step $FRANGI_SIGMA_STEP  --frangi-alpha $FRANGI_ALPHA"
    echo "  --frangi-beta $FRANGI_BETA  --frangi-gamma $FRANGI_GAMMA --trim-val $TRIM_VAL"
    echo "  --frangi-weight $FRANGI_WEIGHT"
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --volpkg) VOLPKG="$2"; shift ;;
        --output-dir) OUTPUT_DIR_BASE="$2"; shift ;;
        --write-vcps) WRITE_VCPS=true ;;
        --volume-id) VOLUME_ID="$2"; shift ;;
        --start-slice-name) START_SLICE_NAME="$2"; shift ;;
        --end-slice-name) END_SLICE_NAME="$2"; shift ;;
        --step-size) STEP_SIZE="$2"; shift ;;
        -t) THRESHOLD="$2"; shift ;;
        -k) NUM_CLUSTERS="$2"; shift ;;
        -n) NUM_SEG_POINTS="$2"; shift ;;
        -s) TOTAL_SEG_POINTS="$2"; shift ;;
        --gaussian-kernel) GAUSSIAN_KERNEL="$2"; shift ;;
        --intensity-alpha) INTENSITY_ALPHA="$2"; shift ;;
        --mask-thickness) MASK_THICKNESS="$2"; shift ;;
        --frangi-sigma-min) FRANGI_SIGMA_MIN="$2"; shift ;;
        --frangi-sigma-max) FRANGI_SIGMA_MAX="$2"; shift ;;
        --frangi-sigma-step) FRANGI_SIGMA_STEP="$2"; shift ;;
        --frangi-alpha) FRANGI_ALPHA="$2"; shift ;;
        --frangi-beta) FRANGI_BETA="$2"; shift ;;
        --frangi-gamma) FRANGI_GAMMA="$2"; shift ;;
        --frangi-weight) FRANGI_WEIGHT="$2"; shift ;;
        --trim-val) TRIM_VAL="$2"; shift ;;
        --help) print_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; print_help; exit 1 ;;
    esac
    shift
done

# Check required args
if [ -z "$VOLPKG" ] || [ -z "$OUTPUT_DIR_BASE" ]  || [ -z "$VOLUME_ID" ]; then
    echo "Error: --input-dir, --output-dir, and --volume-id are required."
    echo ""
    print_help
    exit 1
fi

# Ensure base output folder exists
mkdir -p "$OUTPUT_DIR_BASE"

# Record global start time
global_start_time=$(date +%s)

echo "Volume ID: $VOLUME_ID"
echo "Start Slice: $START_SLICE_NAME"
echo "End Slice: $END_SLICE_NAME"
echo "Step Size: $STEP_SIZE"
echo "Output Directory: $OUTPUT_DIR_BASE"
echo "Write VCPS: $WRITE_VCPS"
echo "--------------------------------------------------------"

# 1. Get the raw string first
START_STR=$(basename "$START_SLICE_NAME" .tif)
END_STR=$(basename "$END_SLICE_NAME" .tif)

# 2. Determine the padding length (e.g., "0000" is 4 characters)
PADDING=${#START_STR}

# 3. Convert to base-10 numbers for the math loop
# The 10# prefix prevents "Illegal number" errors with 0008 or 0009
START_NUM=$((10#$START_STR))
END_NUM=$((10#$END_STR))

echo "Detected padding length: $PADDING"

# Safety check: ensure we actually got numbers
if [[ -z "$START_NUM" || -z "$END_NUM" ]]; then
    echo "ERROR: Could not extract numbers from $START_SLICE_NAME or $END_SLICE_NAME. Please ensure they are in the format 'NNNN.tif'."
    exit 1
fi

# --- Iteration Loop ---
echo "Processing Volume: $VOLUME_ID from $START_SLICE_NAME to $END_SLICE_NAME with step size $STEP_SIZE"
echo "--------------------------------------------------------"




for (( i=$START_NUM; i<=$END_NUM; i+=$STEP_SIZE )); do
    # CURRENT_SLICE=$(printf "%d.tif" $i)
    CURRENT_SLICE=$(printf "%0${PADDING}d.tif" $i)


    OUTPUT_DIR="$OUTPUT_DIR_BASE/${VOLUME_ID}_${CURRENT_SLICE}"
    mkdir -p "$OUTPUT_DIR"
    RUN_CONFIG="$OUTPUT_DIR/run_params_${VOLUME_ID}.txt"

    echo "Saving session parameters to $RUN_CONFIG..."

    {
        echo "# --- Auto-generated Config for Volume: $VOLUME_ID ---"
        echo "# Date: $(date)"
        echo ""
        echo "VOLPKG=\"$VOLPKG\""
        echo "OUTPUT_DIR_BASE=\"$OUTPUT_DIR_BASE\""
        echo "WRITE_VCPS=$WRITE_VCPS"
        echo "VOLUME_ID=\"$VOLUME_ID\""
        echo "START_SLICE_NAME=\"$START_SLICE_NAME\""
        echo "END_SLICE_NAME=\"$END_SLICE_NAME\""
        echo "STEP_SIZE=$STEP_SIZE"
        echo ""
        echo "# --- Segmentation Parameters ---"
        echo "THRESHOLD=$THRESHOLD"
        echo "NUM_CLUSTERS=$NUM_CLUSTERS"
        echo "NUM_SEG_POINTS=$NUM_SEG_POINTS"
        echo "TOTAL_SEG_POINTS=$TOTAL_SEG_POINTS"
        echo "GAUSSIAN_KERNEL=$GAUSSIAN_KERNEL"
        echo "INTENSITY_ALPHA=$INTENSITY_ALPHA"
        echo "MASK_THICKNESS=$MASK_THICKNESS"
        echo "TRIM_VAL=$TRIM_VAL"
        echo ""
        echo "# --- Frangi Filter Parameters ---"
        echo "FRANGI_SIGMA_MIN=$FRANGI_SIGMA_MIN"
        echo "FRANGI_SIGMA_MAX=$FRANGI_SIGMA_MAX"
        echo "FRANGI_SIGMA_STEP=$FRANGI_SIGMA_STEP"
        echo "FRANGI_ALPHA=$FRANGI_ALPHA"
        echo "FRANGI_BETA=$FRANGI_BETA"
        echo "FRANGI_GAMMA=$FRANGI_GAMMA"
        echo "FRANGI_WEIGHT=$FRANGI_WEIGHT"
    } > "$RUN_CONFIG"

    echo "Config saved successfully."
    
    echo ">>> Running slice: $CURRENT_SLICE"


    start_time=$(date +%s)

        CMD=(python3 segmentation/scripts/frangi_vcps_nx_dijkstra_with_h_removal.py
            --volpkg "$VOLPKG"
            --volume "$VOLUME_ID"
            --slice-name "$CURRENT_SLICE"
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
            --frangi-weight "$FRANGI_WEIGHT"
            --txt-coord
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
done

echo "--------------------------------------------------------"
echo "Done!"


# Global end time
global_end_time=$(date +%s)
total_time=$((global_end_time - global_start_time))

echo "🎯 All processing complete in $total_time seconds."
