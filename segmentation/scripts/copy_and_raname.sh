#!/bin/bash

show_help() {
  echo "Usage: $0 --source SOURCE_DIR --destination DEST_DIR --postfix POSTFIX"
  echo ""
  echo "Copies from each cluster_0 directory under SOURCE_DIR:"
  echo "  - binary_<N>.jpg, gaussian_<N>.jpg, skeleton_<N>.jpg → copied as-is"
  echo "  - segmented_*.jpg → renamed to segment_<POSTFIX>_<N>"
  echo "  - total_colored_*.jpg → renamed to total_segment_<POSTFIX>_<N>"
  echo ""
  echo "Only filenames that strictly match name_<number>.jpg are accepted."
  exit 0
}

# Parse named args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --destination) DEST="$2"; shift 2 ;;
    --postfix) POSTFIX="$2"; shift 2 ;;
    --help) show_help ;;
    *) echo "Unknown argument: $1"; show_help ;;
  esac
done

# Validate
if [[ -z "$SOURCE" || -z "$DEST" || -z "$POSTFIX" ]]; then
  echo "❌ Error: --source, --destination, and --postfix are required."
  show_help
fi

mkdir -p "$DEST"

# Find all cluster_0 folders
CLUSTERS=$(find "$SOURCE" -type d -name "cluster_0")

for CLUSTER in $CLUSTERS; do
  echo "📂 Processing: $CLUSTER"

  # Find binary_<number>.jpg
  BINARY_FILE=$(find "$CLUSTER" -maxdepth 1 -type f -regex ".*/binary_[0-9]+\.jpg" | head -n1)

  if [[ ! -f "$BINARY_FILE" ]]; then
    echo "⚠ No binary_<N>.jpg found in $CLUSTER. Skipping."
    continue
  fi

  NUMBER=$(basename "$BINARY_FILE" | grep -oP '(?<=binary_)\d+(?=\.jpg)')
  if [[ -z "$NUMBER" ]]; then
    echo "⚠ Couldn't extract number from $BINARY_FILE. Skipping."
    continue
  fi

  # Copy strict files
  for TYPE in binary gaussian skeleton; do
    FILE="$CLUSTER/${TYPE}_${NUMBER}.jpg"
    if [[ -f "$FILE" ]]; then
      cp "$FILE" "$DEST/"
      echo "✔ Copied $(basename "$FILE")"
    fi
  done

  # Rename segmented_* → segment_<postfix>_<number>
  SEG=$(find "$CLUSTER" -maxdepth 1 -type f -name "segmented_*.jpg" | head -n1)
  if [[ -f "$SEG" ]]; then
    DEST_SEG="$DEST/segment_${POSTFIX}_${NUMBER}.jpg"
    cp "$SEG" "$DEST_SEG"
    echo "✔ Copied and renamed segmented_* → $(basename "$DEST_SEG")"
  fi

  # Rename total_colored_* → total_segment_<postfix>_<number>
  TCOL=$(find "$CLUSTER" -maxdepth 1 -type f -name "total_colored_*.jpg" | head -n1)
  if [[ -f "$TCOL" ]]; then
    DEST_TCOL="$DEST/total_segment_${POSTFIX}_${NUMBER}.jpg"
    cp "$TCOL" "$DEST_TCOL"
    echo "✔ Copied and renamed total_colored_* → $(basename "$DEST_TCOL")"
  fi
done