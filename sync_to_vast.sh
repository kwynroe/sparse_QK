#!/bin/bash

# Configuration
SOURCE_DIR="."
DEST="vast:/workspace/sparse_QK"
EXCLUDE_PATTERN="*.ipynb"

# Function to perform sync
sync_to_vast() {
    # Check if there are actual changes
    changes=$(rsync -avzn --exclude="$EXCLUDE_PATTERN" "$SOURCE_DIR"/* "$DEST" 2>&1)

   if echo "$changes" | grep -qvE '^(sending|receiving|$)'; then
        echo "Changes detected at $(date). Syncing..."
        rsync -avz --progress --exclude="$EXCLUDE_PATTERN" "$SOURCE_DIR"/* "$DEST"
        echo "Sync completed at $(date)"
        echo "-------------------"
    fi
}

# Check if fswatch is installed
if ! command -v fswatch &> /dev/null; then
    echo "fswatch is not installed. Please install it using 'brew install fswatch'"
    exit 1
fi

# Main loop
echo "Starting sync watch on $SOURCE_DIR"
fswatch -o "$SOURCE_DIR" | while read -r file; do
    sync_to_vast
done
