#!/bin/bash

# Set the input directory and output file name
input_dir=$1
output_file=$2

# Check if the input directory exists
if [ ! -d "$input_dir" ]; then
  echo "Input directory '$input_dir' does not exist."
  exit 1
fi

# Check if there are image files in the directory
if ! [ "$(ls -A $input_dir)" ]; then
  echo "No image files found in '$input_dir'."
  exit 1
fi

# Create the GIF animation
convert -delay 10 -loop 0 "${input_dir}/*" "$output_file"

echo "GIF animation '$output_file' created."