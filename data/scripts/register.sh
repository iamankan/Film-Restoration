#!/bin/bash

# Help function
print_help() {
    echo "Usage: $0 --id <film-id>"
    echo ""
    echo "Arguments:"
    echo "  --id   Film id"
    echo "Example:"
    echo "  $0 --id 001"
    exit 1
}

# Initialize variables
ID=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --id)
            ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            print_help
            ;;
    esac
done

# /Volumes/Working_4TB/utils/registration-toolkit/build/bin/rt_register -m /Volumes/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/optical/${ID}.jpg -f /Volumes/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/xray_render/edited/tif/${ID}_render.tif -o /Volumes/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/registered/${ID}_reg.tif


# echo /Volumes/Working_4TB/utils/registration-toolkit/build/bin/rt_register -m /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/match/${ID}_hdr_positive.tiff -f /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/render/${ID}_max.tif -o /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/match/${ID}_hdr_positive_registered.tiff

/Volumes/Working_4TB/utils/registration-toolkit/build/bin/rt_register \
-m /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/match/${ID}_hdr_positive.png \
-f /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/render/${ID}_max.tif \
-o /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/match/${ID}_hdr_positive_registered.png \
--output-ldm /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/match/${ID}_hdr_positive_ldm.ldm

/Volumes/Working_4TB/utils/registration-toolkit/build/bin/rt_register \
-m /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/match/${ID}_hdr_positive.tiff \
-f /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/render/${ID}_max.tif \
-o /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/match/${ID}_hdr_positive_registered.tiff \
-l /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/${ID}/match/${ID}_hdr_positive_ldm.ldm


# /Volumes/Working_4TB/utils/registration-toolkit/build/bin/rt_register \
# -m /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/013/match/013_hdr_positive.tiff \
# -f /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset/IBW_no_seam/013/render/013_max.tif \
# -o /Volumes/Working_4TB/IBW_no_Seam/no_seam_dataset_win_2/013/match/013_hdr_positive_registered.tiff