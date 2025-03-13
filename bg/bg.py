import cv2
import numpy as np
from PIL import Image
from rembg import remove
import matplotlib.pyplot as plt

def remove_background(image_path):
    """Removes the background using rembg and returns an image with transparency."""
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    output_path = "temp_no_bg.png"
    output_image.save(output_path)  # Save the output with transparency
    return output_path

def segment_flower(image_path):
    """Segments the flower using the alpha channel from rembg output."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print("Error: Could not load image. Check the file path.")
        return None

    # If image has an alpha channel, use it to create the mask
    if image.shape[-1] == 4:  # Check if alpha channel exists
        alpha_channel = image[:, :, 3]  # Extract the alpha channel
        mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)[1]  # Convert it to binary

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Apply the mask to extract the flower
        result = cv2.bitwise_and(image[:, :, :3], image[:, :, :3], mask=cleaned_mask)
        
        return result

    print("Error: No alpha channel found. Background removal may have failed.")
    return None

# 1️⃣ Step 1: Remove Background
input_image_path = "data/IMG-20250310-WA0043.jpg"  # Replace with your image path
bg_removed_path = remove_background(input_image_path)

# 2️⃣ Step 2: Segment the Flower
final_output = segment_flower(bg_removed_path)

# 3️⃣ Step 3: Display and Save Final Result
if final_output is not None:
    # Convert to RGB format for Matplotlib
    final_output_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(final_output_rgb)
    plt.axis('off')
    plt.title("Final Flower Output")
    plt.show()

    # Save the final segmented flower
    cv2.imwrite("FinalFlower.png", final_output)
else:
    print("Segmentation failed.")
