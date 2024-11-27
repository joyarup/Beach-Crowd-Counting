import cv2
import numpy as np
from scipy import ndimage
import os
import glob
import matplotlib.pyplot as plt

def load_images_from_folder(folder_path):
    """Load all images from the specified folder"""
    image_files = glob.glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) + \
                 glob.glob(os.path.join(folder_path, '*.[pP][nN][gG]'))
    
    images = []
    filenames = []
    
    for image_path in sorted(image_files):
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
            filenames.append(os.path.basename(image_path))
            print(f"Loaded: {os.path.basename(image_path)}")
        else:
            print(f"Failed to load: {image_path}")
    
    return images, filenames

def apply_clahe_processing(image, clip_limit=2.0, tile_size=(8,8)):
    """Apply CLAHE to enhance image contrast"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    # Apply CLAHE to L channel
    l_clahe = clahe.apply(l)
    
    # Merge channels back
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return enhanced

def detect_people_edge_based(image, min_area=100, max_area=1000):
    """Detect people using edge detection and contour analysis"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = []
    mask = np.zeros_like(gray)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, -1)
    
    return len(valid_contours), mask, edges

def detect_people_adaptive_thresh(image, min_area=100, max_area=1000):
    """Detect people using adaptive thresholding"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = []
    mask = np.zeros_like(gray)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, -1)
    
    return len(valid_contours), mask, thresh

def process_image_all_methods(image):
    """Process image using all detection methods"""
    # Enhance image using CLAHE
    enhanced = apply_clahe_processing(image)
    
    # Apply edge-based detection
    edge_count, edge_mask, edges = detect_people_edge_based(enhanced)
    
    # Apply adaptive threshold detection
    adapt_count, adapt_mask, thresh = detect_people_adaptive_thresh(enhanced)
    
    return {
        'enhanced': enhanced,
        'edge_detection': {
            'count': edge_count,
            'mask': edge_mask,
            'edges': edges
        },
        'adaptive_thresh': {
            'count': adapt_count,
            'mask': adapt_mask,
            'thresh': thresh
        }
    }

def visualize_results_for_presentation(image, results, filename):
    """Create comprehensive visualization of all detection methods"""
    # Create figure with 3x2 subplots
    plt.figure(figsize=(20, 12))
    
    # 1. Original Image
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # 2. Enhanced Image (CLAHE)
    plt.subplot(232)
    plt.imshow(cv2.cvtColor(results['enhanced'], cv2.COLOR_BGR2RGB))
    plt.title('CLAHE Enhanced')
    plt.axis('off')
    
    # 3. Edge Detection
    plt.subplot(233)
    plt.imshow(results['edge_detection']['edges'], cmap='gray')
    plt.title(f'Edge Detection\n(Count: {results["edge_detection"]["count"]} people)')
    plt.axis('off')
    
    # 4. Edge Detection Mask
    plt.subplot(234)
    plt.imshow(results['edge_detection']['mask'], cmap='gray')
    plt.title('Edge Detection Mask')
    plt.axis('off')
    
    # 5. Adaptive Threshold
    plt.subplot(235)
    plt.imshow(results['adaptive_thresh']['thresh'], cmap='gray')
    plt.title(f'Adaptive Threshold\n(Count: {results["adaptive_thresh"]["count"]} people)')
    plt.axis('off')
    
    # 6. Adaptive Threshold Mask
    plt.subplot(236)
    plt.imshow(results['adaptive_thresh']['mask'], cmap='gray')
    plt.title('Adaptive Threshold Mask')
    plt.axis('off')
    
    # Add overall title
    plt.suptitle(f'People Detection Analysis - {filename}', fontsize=16)
    plt.tight_layout()
    
    return plt.gcf()

def create_trend_analysis(results_list, filenames, output_folder):
    """Create trend analysis comparing different methods"""
    plt.figure(figsize=(12, 6))
    
    # Extract counts
    edge_counts = [r['edge_detection']['count'] for r in results_list]
    adapt_counts = [r['adaptive_thresh']['count'] for r in results_list]
    
    # Plot trends
    plt.plot(range(len(edge_counts)), edge_counts, 'b-o', 
             label='Edge Detection', linewidth=2, markersize=8)
    plt.plot(range(len(adapt_counts)), adapt_counts, 'r-o', 
             label='Adaptive Threshold', linewidth=2, markersize=8)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Customize plot
    plt.title('People Count Comparison Across Methods', fontsize=14)
    plt.xlabel('Image Sequence', fontsize=12)
    plt.ylabel('Number of People Detected', fontsize=12)
    
    # Add value labels
    for i, (edge_count, adapt_count) in enumerate(zip(edge_counts, adapt_counts)):
        plt.annotate(str(edge_count), (i, edge_count), 
                    textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(str(adapt_count), (i, adapt_count), 
                    textcoords="offset points", xytext=(0,-15), ha='center')
    
    # Rotate x-axis labels
    plt.xticks(range(len(edge_counts)), [f'Image {i+1}' for i in range(len(edge_counts))], 
               rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'methods_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_results_with_presentation(output_folder, results_list, images, filenames):
    """Save detection results with presentation-quality visualizations"""
    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    presentation_folder = os.path.join(output_folder, 'presentation_visuals')
    os.makedirs(presentation_folder, exist_ok=True)
    
    # Save visualizations and create summary
    summary = []
    
    for result, image, filename in zip(results_list, images, filenames):
        # Create visualization figure
        fig = visualize_results_for_presentation(image, result, filename)
        
        # Save visualization
        base_name = os.path.splitext(filename)[0]
        fig.savefig(os.path.join(presentation_folder, f'analysis_{base_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Add to summary
        summary.append(f"{filename}:")
        summary.append(f"  Edge Detection: {result['edge_detection']['count']} people")
        summary.append(f"  Adaptive Threshold: {result['adaptive_thresh']['count']} people")
        summary.append("")
    
    # Create and save trend analysis
    create_trend_analysis(results_list, filenames, presentation_folder)
    
    # Save summary to file
    with open(os.path.join(output_folder, 'detection_summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
    
    # Print summary to console
    print("\nDetection Summary:")
    print("------------------")
    for line in summary:
        print(line)

def main():
    # Define input and output folders
    dataset_folder = "Dataset"
    output_folder = "alternative_detection_results"
    
    # Load images
    print("Loading images from dataset...")
    images, filenames = load_images_from_folder(dataset_folder)
    
    if not images:
        print("No images found in the dataset folder!")
        return
    
    print(f"\nProcessing {len(images)} images...")
    
    # Process images using all methods
    results_list = []
    for image in images:
        results = process_image_all_methods(image)
        results_list.append(results)
    
    # Save results with enhanced visualizations
    print("\nSaving results and creating visualizations...")
    save_results_with_presentation(output_folder, results_list, images, filenames)
    
    print(f"\nProcessing complete! Results saved in '{output_folder}' folder")
    print("Presentation-quality visualizations are in the 'presentation_visuals' subfolder")

if __name__ == "__main__":
    main()