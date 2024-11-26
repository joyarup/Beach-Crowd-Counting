# Create README.md
echo "# Beach Crowd Detection

This project implements computer vision techniques to detect and count people in beach scenes using different methods:

## Methods Implemented
1. Background Subtraction
2. Edge Detection with CLAHE
3. Adaptive Thresholding

## Project Structure
- \`crowd_detection_background.py\`: Implementation using background subtraction
- \`crowd_detection_edge.py\`: Implementation using edge detection and CLAHE
- \`Dataset/\`: Folder containing beach images (not included in repo)
- \`detection_results/\`: Output folder for background subtraction method
- \`alternative_detection_results/\`: Output folder for edge detection method

## Requirements
- OpenCV
- NumPy
- SciPy
- Matplotlib

## Usage
1. Place your beach images in the 'Dataset' folder
2. Run background subtraction method:
   \`\`\`
   python crowd_detection_background.py
   \`\`\`
3. Run edge detection method:
   \`\`\`
   python crowd_detection_edge.py
   \`\`\`
" > README.md

# Add and commit README
git add README.md
git commit -m "Add README"
git push
