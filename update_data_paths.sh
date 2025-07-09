#!/bin/bash

echo "Updating data paths from 'data/full_lob_data' to 'data/data/full_lob_data'..."

# Update raw data path in prepare_attention_model_data_v2.py
echo "Updating prepare_attention_model_data_v2.py..."
sed -i "s|'data_path': 'data/full_lob_data/resampled_5s'|'data_path': 'data/data/full_lob_data/resampled_5s'|g" prepare_attention_model_data_v2.py

# Update any other references to data/full_lob_data in Python files
echo "Updating other Python files with data/full_lob_data references..."
find . -name "*.py" -exec sed -i "s|data/full_lob_data|data/data/full_lob_data|g" {} \;

# Update any references in markdown files
echo "Updating markdown files..."
find . -name "*.md" -exec sed -i "s|data/full_lob_data|data/data/full_lob_data|g" {} \;

echo "Path updates completed!"
echo ""
echo "Summary of changes:"
echo "   Raw data path: data/full_lob_data/resampled_5s -> data/data/full_lob_data/resampled_5s"
echo "   Processed data path: data/final_attention (unchanged)"
echo ""
echo "To verify changes, run:"
echo "   grep -r 'data/full_lob_data' . --include='*.py'"
echo "   grep -r 'data/data/full_lob_data' . --include='*.py'" 