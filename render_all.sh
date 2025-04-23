#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p output

# Process all .bin files in the lights directory
for file in lights/*.bin; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        ./render "$file"
    fi
done

echo "All files processed!" 