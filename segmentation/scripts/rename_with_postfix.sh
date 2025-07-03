#!/bin/bash

show_help() {
  echo "Usage: $0 --folder FOLDER_PATH --postfix POSTFIX"
  echo ""
  echo "Renames files in FOLDER_PATH from 'name.ext' to 'name_POSTFIX.ext'"
  exit 0
}

# Parse named args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --folder) FOLDER="$2"; shift 2 ;;
    --postfix) POSTFIX="$2"; shift 2 ;;
    --help) show_help ;;
    *) echo "Unknown argument: $1"; show_help ;;
  esac
done

# Check args
if [[ -z "$FOLDER" || -z "$POSTFIX" ]]; then
  echo "Error: --folder and --postfix are required."
  show_help
fi

# Check if folder exists
if [[ ! -d "$FOLDER" ]]; then
  echo "Error: Folder '$FOLDER' does not exist."
  exit 1
fi

# Rename files
shopt -s nullglob
for FILE in "$FOLDER"/*.*; do
  BASENAME=$(basename "$FILE")
  EXT="${BASENAME##*.}"
  NAME="${BASENAME%.*}"

  NEW_NAME="${NAME}_${POSTFIX}.${EXT}"
  mv -v "$FILE" "$FOLDER/$NEW_NAME"
done