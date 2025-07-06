#!/bin/bash

# Help function
print_help() {
    echo "Usage: $0 --reconstruction-directory <path> --volpkg-directory <path>"
    echo ""
    echo "Arguments:"
    echo "  --reconstruction-directory   Path to the directory containing reconstructions"
    echo "  --volpkg-directory           Path to the directory where .volpkg files will be created"
    echo ""
    echo "Example:"
    echo "  $0 --reconstruction-directory /path/to/recon --volpkg-directory /path/to/volpkgs"
    exit 1
}

# Initialize variables
RECON_DIR=""
VOLPKG_DIR=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reconstruction-directory)
            RECON_DIR="$2"
            shift 2
            ;;
        --volpkg-directory)
            VOLPKG_DIR="$2"
            shift 2
            ;;
        --help|-h)
            print_help
            ;;
        *)
            echo "Unknown parameter: $1"
            print_help
            ;;
    esac
done

# Check if required arguments were provided
if [[ -z "$RECON_DIR" || -z "$VOLPKG_DIR" ]]; then
    echo "Error: Both arguments are required."
    print_help
fi

# Create volpkg directory if it does not exist
if [ ! -d "$VOLPKG_DIR" ]; then
    echo "Creating volpkg directory: $VOLPKG_DIR"
    mkdir -p "$VOLPKG_DIR"
fi

# Iterate through reconstruction directory
for ID_DIR in "$RECON_DIR"/*; do
    if [ -d "$ID_DIR" ]; then
        ID=$(basename "$ID_DIR")
        LOG_PATH="$RECON_DIR/$ID/IBW_${ID}_10um_60kV_Rec_ROI/IBW_${ID}_10um_60kV__rec.log"
        VOLPKG_PATH="$VOLPKG_DIR/IBW_${ID}_10um_60kV.volpkg"
        NAME="IBW_${ID}_10um_60kV"

        # echo "Log file name for $ID is $LOG_PATH"

        # ls "$RECON_DIR/${ID}/IBW_${ID}_10um_60kV_Rec_ROI/IBW_${ID}_10um_60kV__rec.log"


        if [ -f "$LOG_PATH" ]; then
            echo "Processing $ID..."
            vc_packager -v "$VOLPKG_PATH" --name "$NAME" -m 140 -s "$LOG_PATH" -n roi -u 10
        else
            echo "Warning: Log file not found for ID: $ID, skipping."
        fi
    fi
done