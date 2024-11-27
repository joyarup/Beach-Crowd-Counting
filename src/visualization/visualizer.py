import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd

def visualize_results_for_presentation(image, background, mask, count, filename, ground_truth_coords=None):
    """Create a comprehensive visualization including ground truth if available."""
    plt.figure(figsize=(20, 10))

    # Original Image
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Background Model
    plt.subplot(232)
    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    plt.title('Background Model')
    plt.axis('off')

    # Detection Mask
    plt.subplot(233)
    plt.imshow(mask, cmap='gray')
    plt.title('Detection Mask')
    plt.axis('off')

    # Ground Truth Visualization
    plt.subplot(234)
    img_with_gt = image.copy()
    if ground_truth_coords:
        for x, y in ground_truth_coords:
            cv2.circle(img_with_gt, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green dots
            cv2.circle(img_with_gt, (int(x), int(y)), 7, (0, 255, 0), 2)   # Green circles
    plt.imshow(cv2.cvtColor(img_with_gt, cv2.COLOR_BGR2RGB))
    plt.title(f'Ground Truth ({len(ground_truth_coords) if ground_truth_coords else 0} people)')
    plt.axis('off')

    # Detection Result
    plt.subplot(235)
    overlay = image.copy()
    overlay[mask > 0] = [0, 0, 255]  # Mark detections in red
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f'Detection Result ({count} people)')
    plt.axis('off')

    # Combined Visualization
    plt.subplot(236)
    combined = image.copy()
    # Add ground truth (green)
    if ground_truth_coords:
        for x, y in ground_truth_coords:
            cv2.circle(combined, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.circle(combined, (int(x), int(y)), 7, (0, 255, 0), 2)
    # Add detections (red)
    detection_overlay = image.copy()
    detection_overlay[mask > 0] = [0, 0, 255]
    combined = cv2.addWeighted(combined, 0.7, detection_overlay, 0.3, 0)
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title('Combined (Green: Ground Truth, Red: Detections)')
    plt.axis('off')

    plt.suptitle(f'People Detection Analysis - {filename}', fontsize=16)
    plt.tight_layout()
    return plt.gcf()

def save_results_with_presentation(output_folder, results, images, filenames, background, ground_truth_file=None):
    """Save detection results with presentation-quality visualizations including ground truth."""
    os.makedirs(output_folder, exist_ok=True)
    presentation_folder = os.path.join(output_folder, 'presentation_visuals')
    os.makedirs(presentation_folder, exist_ok=True)

    # Load ground truth data if available
    ground_truth = {}
    if ground_truth_file:
        df = pd.read_csv(ground_truth_file, header=None, 
                        names=['label', 'x', 'y', 'image', 'width', 'height'])
        for image_name in df['image'].unique():
            image_data = df[df['image'] == image_name]
            ground_truth[image_name] = list(zip(image_data['x'], image_data['y']))

    summary = []
    all_counts = []
    gt_counts = []

    for result, image, filename in zip(results, images, filenames):
        # Get ground truth coordinates for this image
        gt_coords = ground_truth.get(filename, [])
        
        fig = visualize_results_for_presentation(
            image,
            background,
            result['detection_mask'],
            result['people_count'],
            filename,
            gt_coords
        )
        base_name = os.path.splitext(filename)[0]
        fig.savefig(os.path.join(presentation_folder, f'analysis_{base_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        summary.append(f"{filename}: {result['people_count']} people detected (Ground Truth: {len(gt_coords)})")
        all_counts.append(result['people_count'])
        gt_counts.append(len(gt_coords))

    # Save detection summary with ground truth comparison
    with open(os.path.join(output_folder, 'detection_summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
        if all_counts:
            f.write('\n\nSummary Statistics:\n')
            f.write(f'Average detected count: {np.mean(all_counts):.1f}\n')
            f.write(f'Average ground truth count: {np.mean(gt_counts):.1f}\n')
            f.write(f'Average count difference: {np.mean(np.abs(np.array(all_counts) - np.array(gt_counts))):.1f}\n')
            f.write(f'Maximum detected count: {max(all_counts)}\n')
            f.write(f'Maximum ground truth count: {max(gt_counts)}\n')
            f.write(f'Minimum detected count: {min(all_counts)}\n')
            f.write(f'Minimum ground truth count: {min(gt_counts)}\n')

    cv2.imwrite(os.path.join(output_folder, 'background_model.png'), background)

    print("\nDetection Summary:")
    for line in summary:
        print(line)
    if all_counts:
        print(f"\nAverage detected count: {np.mean(all_counts):.1f}")
        print(f"Average ground truth count: {np.mean(gt_counts):.1f}")
        print(f"Average count difference: {np.mean(np.abs(np.array(all_counts) - np.array(gt_counts))):.1f}")
        print(f"Maximum detected count: {max(all_counts)}")
        print(f"Maximum ground truth count: {max(gt_counts)}")
        print(f"Minimum detected count: {min(all_counts)}")
        print(f"Minimum ground truth count: {min(gt_counts)}")