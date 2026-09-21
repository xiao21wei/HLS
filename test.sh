#!/bin/bash


INPUT_DIR="experiment/datasets1"
OUTPUT_DIR="../test_save/drl1"

INPUT_DIR1="/data/luopw/production_scheduling/build/datasets1"

mkdir -p "$OUTPUT_DIR"

# for file in "$INPUT_DIR1"/*; do
#     filename=$(basename "$file")
#     output_file="$OUTPUT_DIR/$filename"
#     echo "Processing input: $file"
#     echo "Saving output to: $output_file"
#     ./production_scheduling "$INPUT_DIR1/$filename" 0.5 0.5 100 30 20 10 > "$output_file" &
# done

# for file in "$INPUT_DIR1"/*; do
#     for i in {10..60..5}; do
#         filename=$(basename "$file")
#         output_file="$OUTPUT_DIR/${i}/$filename"
#         mkdir -p "$OUTPUT_DIR/${i}"
#         echo "Processing input: $file with tabu_list_length=$i"
#         echo "Saving output to: $output_file"
#         ./production_scheduling "$INPUT_DIR1/$filename" 30 20 "$i" 100 > "$output_file" &
#     done
# done
# 0.5 0.5 100 30 20 25

# for i in {6..10}; do
#     for file in "$INPUT_DIR1"/*; do
#         filename=$(basename "$file")
#         mkdir -p "$OUTPUT_DIR/${i}"
#         output_file="$OUTPUT_DIR/${i}/${filename}"
#         echo "Saving output to: $output_file"
#         ./production_scheduling "$INPUT_DIR1/$filename" 0.5 0.5 100 30 20 25 > "$output_file" &
#     done
# done

for file in "$INPUT_DIR1"/*; do
    filename=$(basename "$file")
    mkdir -p "$OUTPUT_DIR/11"
    output_file="$OUTPUT_DIR/11/${filename}"
    echo "Saving output to: $output_file"
    ./production_scheduling "$INPUT_DIR1/$filename" 0.5 0.5 100 30 20 25 > "$output_file" &
done

