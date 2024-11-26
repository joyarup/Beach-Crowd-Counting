import cv2
import numpy as np
from scipy import ndimage
import os
import glob
import matplotlib.pyplot as plt

def load_images_from_folder(folder_path):
    """Load all images from the specified folder"""
    # Get list of image files (supports common image formats)
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

def create_background_model(images):
    """Create a background model by averaging multiple frames"""
    if len(images) == 0:
        return None
    
    # Convert list of images to numpy array
    image_array = np.array(images)
    # Calculate median across all frames
    background = np.median(image_array, axis=0).astype(np.uint8)
    return background

def detect_and_count_people(image, background, min_person_size=25):
    """Detect and count people using background subtraction and morphological operations"""
    
    # Convert images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    # Perform background subtraction
    diff = cv2.absdiff(gray_background, gray_image)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove noise and connect components
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Label connected components
    labeled, num_objects = ndimage.label(thresh)
    
    # Filter objects by size to remove noise
    sizes = ndimage.sum(thresh, labeled, range(1, num_objects + 1))
    mask = np.zeros_like(thresh)
    
    # Only keep objects larger than min_person_size
    for i, size in enumerate(sizes):
        if size >= min_person_size:
            mask[labeled == i + 1] = 255
    
    # Count remaining objects (potential people)
    labeled_filtered, num_people = ndimage.label(mask)
    
    return num_people, mask

def process_beach_images(image_list):
    """Process multiple beach images to detect and count people"""
    
    # Create background model
    background = create_background_model(image_list)
    if background is None:
        return []
    
    results = []
    for i, image in enumerate(image_list):
        # Detect and count people
        count, mask = detect_and_count_people(image, background)
        
        results.append({
            'image_index': i,
            'people_count': count,
            'detection_mask': mask
        })
        
    return results

def visualize_results_for_presentation(image, background, mask, count, filename):
    """Create a comprehensive visualization for presentation"""
    # Create figure with 2x2 subplots
    plt.figure(figsize=(15, 10))
    
    # 1. Original Image
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # 2. Background Model
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    plt.title('Background Model')
    plt.axis('off')
    
    # 3. Detection Mask
    plt.subplot(223)
    plt.imshow(mask, cmap='gray')
    plt.title('Detection Mask')
    plt.axis('off')
    
    # 4. Final Result with Overlay
    overlay = image.copy()
    overlay[mask > 0] = [0, 0, 255]  # Mark detections in red
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f'Final Detection (Count: {count} people)')
    plt.axis('off')
    
    # Add overall title
    plt.suptitle(f'People Detection Analysis - {filename}', fontsize=16)
    plt.tight_layout()
    
    return plt.gcf()

def create_trend_analysis(counts, filenames, output_folder):
    """Create trend analysis visualization"""
    plt.figure(figsize=(12, 6))
    
    # Plot people count trend
    plt.plot(range(len(counts)), counts, 'b-o', linewidth=2, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize plot
    plt.title('People Count Trend Across Images', fontsize=14)
    plt.xlabel('Image Sequence', fontsize=12)
    plt.ylabel('Number of People Detected', fontsize=12)
    
    # Add value labels on points
    for i, count in enumerate(counts):
        plt.annotate(str(count), (i, count), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    # Rotate x-axis labels for better readability
    plt.xticks(range(len(counts)), [f'Image {i+1}' for i in range(len(counts))], 
               rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'count_trend_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_results_with_presentation(output_folder, results, images, filenames, background):
    """Save detection results with presentation-quality visualizations"""
    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    presentation_folder = os.path.join(output_folder, 'presentation_visuals')
    os.makedirs(presentation_folder, exist_ok=True)
    
    # Save visualizations and create summary
    summary = []
    all_counts = []
    
    for result, image, filename in zip(results, images, filenames):
        # Create visualization figure
        fig = visualize_results_for_presentation(
            image,
            background,
            result['detection_mask'],
            result['people_count'],
            filename
        )
        
        # Save visualization
        base_name = os.path.splitext(filename)[0]
        fig.savefig(os.path.join(presentation_folder, f'analysis_{base_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Add to summary
        summary.append(f"{filename}: {result['people_count']} people detected")
        all_counts.append(result['people_count'])
    
    # Create and save trend analysis
    create_trend_analysis(all_counts, filenames, presentation_folder)
    
    # Save summary to file
    with open(os.path.join(output_folder, 'detection_summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
        
    # Save background model
    cv2.imwrite(os.path.join(output_folder, 'background_model.png'), background)
    
    # Print summary to console
    print("\nDetection Summary:")
    print("------------------")
    for line in summary:
        print(line)
    print(f"\nAverage count: {np.mean(all_counts):.1f} people")
    print(f"Maximum count: {max(all_counts)} people")
    print(f"Minimum count: {min(all_counts)} people")

def main():
    # Define input and output folders
    dataset_folder = "Dataset"
    output_folder = "detection_results"
    
    # Load images
    print("Loading images from dataset...")
    images, filenames = load_images_from_folder(dataset_folder)
    
    if not images:
        print("No images found in the dataset folder!")
        return
    
    print(f"\nProcessing {len(images)} images...")
    # Create background model first
    background = create_background_model(images)
    
    # Process images
    results = process_beach_images(images)
    
    # Save results with enhanced visualizations
    print("\nSaving results and creating visualizations...")
    save_results_with_presentation(output_folder, results, images, filenames, background)
    
    print(f"\nProcessing complete! Results saved in '{output_folder}' folder")
    print("Presentation-quality visualizations are in the 'presentation_visuals' subfolder")

if __name__ == "__main__":
    main()