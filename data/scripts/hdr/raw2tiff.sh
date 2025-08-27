#!/bin/bash

show_help() {
    echo "Usage: $0 -d <directory> -e <extension>"
    echo
    echo "Options:"
    echo "  -d DIR    Specify the directory to process"
    echo "  -e EXT    Specify the file extension to filter (e.g. NEF, jpg, txt)"
    echo "  -n NEWNAME Optional: base name for all output files"
    echo "  -o OUTPUT_FOLDER Optional: directory to save converted TIFFs"
    echo "  -h        Show this help message and exit"
}

# Parse arguments
while getopts "d:e:n:o:h" opt; do
  case ${opt} in
    d )
      DIR=$OPTARG
      ;;
    e )
      EXT=$OPTARG
      ;;
    n )
      NEWNAME=$OPTARG
      ;;
    o ) 
      OUTPUT_DIR=$OPTARG
      ;;
    h )
      show_help
      exit 0
      ;;
    \? )
      show_help
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$DIR" ] || [ -z "$EXT" ]; then
    echo "Error: Both directory and extension must be specified."
    show_help
    exit 1
fi

# If output folder is provided, create it if it doesn't exist
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$DIR/$OUTPUT_DIR"
else
    OUTPUT_DIR="$DIR"
fi

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: '$DIR' is not a valid directory."
    exit 1
fi

# Go into the directory
cd "$DIR" || exit

# Process files with given extension
shopt -s nullglob   # prevents literal *.EXT if no matches
FILES=(*."$EXT")

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files found with extension .$EXT"
    exit 0
fi

for file in "${FILES[@]}"; do
    # Read exposure difference using exiftool (value only)
    expdiff=$(exiftool -s3 -ExposureDifference "$file")

    # Make it filename-safe
    if [[ "$expdiff" == "0" ]]; then
    	expdiff_safe="p0"
    else
	# Replace characters
	expdiff_safe=${expdiff//\//p}    # replace /
	expdiff_safe=${expdiff_safe//+/p} # replace +
	expdiff_safe=${expdiff_safe// /_} # replace spaces
	expdiff_safe=${expdiff_safe//-/m} # replace - with m
    fi

    # Base filename without extension
    if [ -n "$NEWNAME" ]; then
        base="$NEWNAME"
    else
        base="${file%.*}"
    fi

    # New filename for TIFF output
    # newfile="${base}_expdiff${expdiff_safe}.tiff"
    newfile="${OUTPUT_DIR}/${base}_expdiff${expdiff_safe}.tiff"

    # Run dcraw and save as new filename
    echo "Processing $file → $newfile"
    dcraw -4 -D -d -T -j -c "$file" > "$newfile" # No Debayering
    # dcraw -4 -T -d -c "$file" > "$newfile"
    exiftool -TagsFromFile "$file" "$newfile" -overwrite_original
    echo "Processing $file → $newfile (metadata copied)"


done

