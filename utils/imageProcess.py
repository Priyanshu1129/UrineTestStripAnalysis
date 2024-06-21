import sys
import cv2
import json

def extract_colors(image_path, num_patches=10, gap_width=2):
    labels = ['URO', 'BIL', 'KET', 'BLD', 'PRO', 'NIT', 'LEU', 'GLU', 'SG', 'PH']
    
    image = cv2.imread(image_path)
    if image is None:
        return {'error': 'Image not found'}

    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find the region containing the color patches
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the color patches are in a straight line vertically
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

    if len(contours) < num_patches:
        return {'error': 'Not enough color patches found'}

    # Extract patches from the identified region
    colors = {}
    cropped_images = []

    for i in range(num_patches):
        x, y, w, h = cv2.boundingRect(contours[i])
        x_start = max(0, x - i * gap_width)
        x_end = min(image_rgb.shape[1], x + w + (num_patches - 1 - i) * gap_width)
        patch = image_rgb[y:y+h, x_start:x_end]

        # Calculate the average color of the patch
        average_color = patch.mean(axis=0).mean(axis=0)
        colors[labels[i]] = average_color.astype(int).tolist()

        # Optionally, save the cropped patch as a separate image
        cropped_images.append(patch)

    return colors

if __name__ == '__main__':
    image_path = sys.argv[1]
    result = extract_colors(image_path)
    print(json.dumps(result))
    

# import sys
# import cv2
# import json
# import numpy as np

# def extract_colors(image_path, num_patches=10):
#     labels = ['URO', 'BIL', 'KET', 'BLD', 'PRO', 'NIT', 'LEU', 'GLU', 'SG', 'PH']
    
#     # List of background colors (in RGB)
#     background_colors = [
#         (184, 181, 176), (180, 179, 177), (178, 177, 172), (181, 180, 176), (179, 178, 174),
#         (195, 194, 190), (179, 178, 176), (177, 174, 173), (183, 178, 172), (176, 171, 168),
#         (167, 163, 162), (185, 184, 182), (175, 170, 169), (190, 181, 176), (174, 170, 169),
#         (185, 183, 184), (197, 186, 184), (193, 179, 178), (198, 186, 188), (206, 192, 189),
#         (191, 177, 176), (205, 191, 188), (183, 177, 177), (195, 193, 196), (193, 187, 189),
#         (190, 186, 183), (190, 184, 182), (186, 184, 187), (172, 162, 161), (179, 173, 173),
#         (167, 159, 157)
#     ]
    
#     # Convert image to RGB (OpenCV uses BGR by default)
#     image = cv2.imread(image_path)
#     if image is None:
#         return {'error': 'Image not found'}
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Create a mask for the background
#     mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
#     tolerance = 10  # Tolerance for color range

#     for color in background_colors:
#         lower_bound = np.array([max(0, c - tolerance) for c in color], dtype=np.uint8)
#         upper_bound = np.array([min(255, c + tolerance) for c in color], dtype=np.uint8)
#         current_mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
#         mask = cv2.bitwise_or(mask, current_mask)
    
#     # Invert mask to get the patches
#     mask_inv = cv2.bitwise_not(mask)

#     # Find contours in the mask
#     contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Sort contours by their y-coordinate to process patches from top to bottom
#     contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

#     if len(contours) < num_patches:
#         return {'error': f'Only {len(contours)} patches found, expected {num_patches}'}

#     colors = {}
#     cropped_images = []

#     for i in range(num_patches):
#         # Extract bounding box of the contour
#         x, y, w, h = cv2.boundingRect(contours[i])

#         # Extract patch from original RGB image
#         patch = image_rgb[y:y+h, x:x+w, :]

#         # Calculate the average color of the patch
#         average_color = patch.mean(axis=0).mean(axis=0)
#         colors[labels[i]] = average_color.astype(int).tolist()

#         # Optionally, save the cropped patch as a separate image
#         cropped_images.append(patch)

#     # Save cropped images (optional)
#     for i, cropped_image in enumerate(cropped_images):
#         cv2.imwrite(f'patch_{i+1}.jpg', cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

#     return colors

# if __name__ == '__main__':
#     image_path = sys.argv[1]
#     result = extract_colors(image_path)
#     print(json.dumps(result))

# import sys
# import cv2
# import numpy as np
# import json
# import os

# def extract_colors(image_path, output_path, num_patches=10):
#     labels = ['URO', 'BIL', 'KET', 'BLD', 'PRO', 'NIT', 'LEU', 'GLU', 'SG', 'PH']
    
#     # Try to read the image
#     try:
#         image = cv2.imread(image_path)
#     except Exception as e:
#         return {'error': f'Failed to read image: {str(e)}'}

#     if image is None:
#         return {'error': 'Image not found or cannot be opened'}
    
#     # Convert image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply thresholding to isolate the patches
#     _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Filter contours by size and aspect ratio
#     patch_contours = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = h / float(w)
#         if 0.8 < aspect_ratio < 1.2 and 20 < w < 50 and 20 < h < 50:  # Adjust size range as needed
#             patch_contours.append(contour)

#     if len(patch_contours) < num_patches:
#         return {'error': f'Only {len(patch_contours)} patches found, expected {num_patches}'}

#     # Sort contours by their y-coordinate to get top-to-bottom order
#     patch_contours = sorted(patch_contours, key=lambda c: cv2.boundingRect(c)[1])

#     colors = {}
#     for i, contour in enumerate(patch_contours[:num_patches]):
#         x, y, w, h = cv2.boundingRect(contour)
#         patch = image[y:y+h, x:x+w]
        
#         # Calculate the average color of the patch
#         average_color = patch.mean(axis=0).mean(axis=0)
#         colors[labels[i]] = average_color.astype(int).tolist()

#         # Optionally, draw bounding box around each patch for visualization
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#     # Save the image with bounding boxes (optional)
#     try:
#         cv2.imwrite(output_path, image)
#     except Exception as e:
#         return {'error': f'Failed to save processed image: {str(e)}'}

#     return colors

# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print(json.dumps({'error': 'Usage: python3 extract_colors.py <image_path>'}))
#         sys.exit(1)
    
#     image_path = sys.argv[1]
#     output_path = os.path.join(os.path.dirname(image_path), 'patches_detected.jpg')
#     result = extract_colors(image_path, output_path)
#     print(json.dumps(result, indent=4))
