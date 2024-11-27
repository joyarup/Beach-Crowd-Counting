from .background import create_background_model
import cv2
import numpy as np
def detect_and_count_people(image, background=None):
    """Detect and count people with multi-scale detection for varying distances."""
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Background subtraction if available
    if background is not None:
        gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_background, gray_image)
        _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    else:
        motion_mask = np.ones_like(gray_image) * 255
    
    # Multi-scale detection
    scales = [1.0, 0.75, 0.5]  # Multiple scales for different distances
    all_contours = []
    image_height, image_width = gray_image.shape
    
    for scale in scales:
        # Resize image for different scales
        if scale != 1.0:
            width = int(gray_image.shape[1] * scale)
            height = int(gray_image.shape[0] * scale)
            scaled_image = cv2.resize(gray_image, (width, height))
            scaled_mask = cv2.resize(motion_mask, (width, height))
        else:
            scaled_image = gray_image.copy()
            scaled_mask = motion_mask.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(scaled_image, (5, 5), 0)
        
        # Separate processing for upper (distant) and lower (near) regions
        h, w = scaled_image.shape
        upper_region = blurred[0:int(h/2), :]
        lower_region = blurred[int(h/2):, :]
        
        # Different thresholding parameters for different regions
        upper_thresh = cv2.adaptiveThreshold(
            upper_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 4)  # More sensitive for distant objects
        
        lower_thresh = cv2.adaptiveThreshold(
            lower_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 8)  # Less sensitive for near objects
        
        thresh = np.vstack((upper_thresh, lower_thresh))
        
        # Combine with motion mask
        combined_mask = cv2.bitwise_and(thresh, scaled_mask)
        
        # Scale-dependent morphological operations
        kernel_size = max(2, int(3 * scale))
        kernel_open = np.ones((kernel_size, kernel_size), np.uint8)
        kernel_close = np.ones((kernel_size + 2, kernel_size + 2), np.uint8)
        
        # Remove noise and connect components
        processed = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Scale-specific size filtering
        min_size = int(100 * scale * scale)  # Smaller min size for distant objects
        max_size = int(3000 * scale * scale)
        
        # Filter contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size <= area <= max_size:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / float(w)
                
                # Ignore contours in the top 40% of the image
                bounding_box_center_y = (y + y + h) // 2  # Calculate bounding box center y-coordinate
                if bounding_box_center_y < 0.4 * image_height:
                    continue
                
                # More lenient aspect ratio for distant objects
                if y < h / 2:  # Upper half of image
                    aspect_ratio_range = (0.2, 4.0)
                    solidity_threshold = 0.2
                else:  # Lower half of image
                    aspect_ratio_range = (0.3, 3.0)
                    solidity_threshold = 0.3
                
                if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                    # Additional shape analysis
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    if solidity > solidity_threshold:
                        # Scale contour back to original size
                        if scale != 1.0:
                            contour = (contour / scale).astype(np.int32)
                        all_contours.append(contour)
    
    # Create final mask
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, all_contours, -1, 255, thickness=cv2.FILLED)
    
    # Final cleanup
    kernel_final = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_final)
    
    return len(all_contours), mask

def process_beach_images(images, background=None):
    """Process multiple beach images to detect and count people."""
    if background is None:
        return []

    results = []
    print("\nProcessing individual frames...")
    for i, image in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}...")
        count, mask = detect_and_count_people(image, background)
        results.append({
            'image_index': i,
            'people_count': count,
            'detection_mask': mask
        })

    return results