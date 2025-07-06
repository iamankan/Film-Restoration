#!/bin/bash

# Help function
print_help() {
    echo "Usage: $0 --src <path> --dest <path>"
    echo ""
    echo "Arguments:"
    echo "  --src   Path to the directory having the recons"
    echo "  --dest  Path to the directory where recon will be copied"
    echo ""
    echo "Example:"
    echo "  $0 --reconstruction-directory /path/to/recon --volpkg-directory /path/to/volpkgs"
    exit 1
}

# Initialize variables
SRC=""
DEST=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --src)
            SRC="$2"
            shift 2
            ;;
        --dest)
            DEST="$2"
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
if [[ -z "$SRC" || -z "$DEST" ]]; then
    echo "Error: Both arguments are required."
    print_help
fi

# Create recon directory if it does not exist
if [ ! -d "$DEST" ]; then
    echo "Creating recon directory: $DEST"
    mkdir -p "$DEST"
fi

# Iterate through reconstruction directory
for ID_DIR in "$SRC"/*; do
    if [ -d "$ID_DIR" ]; then
        ID=$(basename "$ID_DIR")
        SRC_DIR="$SRC/$ID/IBW_${ID}_10um_60kV/IBW_${ID}_10um_60kV_Rec_ROI"
        DEST_DIR="$DEST/$ID"
        if [ ! -d "$DEST_DIR" ]; then
            echo "Creating recon directory: $DEST_DIR"
            mkdir -p "$DEST_DIR"
        fi
        # echo "$DEST_DIR,$SRC_DIR"
        cp -r "$SRC_DIR" "$DEST_DIR"
    fi
done