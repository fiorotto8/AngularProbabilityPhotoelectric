#!/bin/bash

# Set the input directory and output file name
input_dir=$1
output_file=$2

montage "${input_dir}/*" -tile 2x2 -geometry +1+1 "$output_file"