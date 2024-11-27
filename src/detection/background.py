import cv2
import numpy as np

def create_background_model(images):
    """Create a robust background model using multiple techniques."""
    if len(images) == 0:
        return None
    
    # Convert images to float32 for better precision
    float_images = [img.astype(np.float32) for img in images]
    
    # Create masks for different regions (water, sand, etc.)
    h, w = float_images[0].shape[:2]
    water_region = np.zeros((h, w), dtype=np.uint8)
    water_region[int(h/3):] = 255  # Assume lower 2/3 might contain water
    
    # Initialize arrays for different statistical measures
    median_model = np.zeros_like(float_images[0])
    mean_model = np.zeros_like(float_images[0])
    mode_model = np.zeros_like(float_images[0])
    
    # Process each color channel separately
    for channel in range(3):
        channel_data = np.array([img[:,:,channel] for img in float_images])
        
        # Median computation
        median_model[:,:,channel] = np.median(channel_data, axis=0)
        
        # Mean with outlier removal
        sorted_data = np.sort(channel_data, axis=0)
        trimmed_mean = np.mean(sorted_data[1:-1], axis=0)  # Exclude highest and lowest values
        mean_model[:,:,channel] = trimmed_mean
        
        # Approximate mode using histogram
        for i in range(0, h, 10):  # Process in blocks for efficiency
            for j in range(0, w, 10):
                block = channel_data[:, i:min(i+10,h), j:min(j+10,w)]
                hist, bins = np.histogram(block.ravel(), bins=50)
                mode_idx = np.argmax(hist)
                mode_value = (bins[mode_idx] + bins[mode_idx + 1]) / 2
                mode_model[i:min(i+10,h), j:min(j+10,w), channel] = mode_value
    
    # Combine models based on region
    background = np.zeros_like(float_images[0])
    
    # Use different techniques for different regions
    for channel in range(3):
        # Water regions: use median (handles reflections better)
        background[:,:,channel] = np.where(
            water_region == 255,
            median_model[:,:,channel],
            mean_model[:,:,channel]
        )
    
    # Apply bilateral filter to preserve edges while smoothing
    background = cv2.bilateralFilter(
        background.astype(np.uint8), 
        9,  # Diameter of pixel neighborhood
        75, # Sigma color
        75  # Sigma space
    )
    
    # Enhance contrast in shadow areas
    lab = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to luminance channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l)
    
    # Merge back
    enhanced_lab = cv2.merge([enhanced_l, a, b])
    enhanced_background = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Final noise removal
    final_background = cv2.fastNlMeansDenoisingColored(
        enhanced_background,
        None,
        10, # Luminance component
        10, # Color components
        7,  # Template window size
        21  # Search window size
    )
    
    return final_background