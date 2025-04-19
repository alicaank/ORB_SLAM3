import cv2
import numpy as np
import glob
import os

# Load the person image (ensure it has an alpha channel if needed)
person_img = cv2.imread("/home/ak/GuidedResearch/data/ry.png", cv2.IMREAD_UNCHANGED)

# If the person has a depth map, load it (otherwise, assign a fixed depth value)
person_depth = None
use_fixed_depth = person_depth is None
fixed_depth_value = 5037  # Arbitrary depth value (adjust based on your range)

# Get sequence of RGB and depth images
rgb_images = sorted(glob.glob("/home/ak/GuidedResearch/data/rgbd_dataset_freiburg1_desk2/rgb/*.png"))
depth_images = sorted(glob.glob("/home/ak/GuidedResearch/data/rgbd_dataset_freiburg1_desk2/depth/*.png"))  # Assume depth images are grayscale PNG

# Ensure output directories exist
os.makedirs("/home/ak/GuidedResearch/data/output/rgb", exist_ok=True)
os.makedirs("/home/ak/GuidedResearch/data/output/depth", exist_ok=True)

# Define movement parameters
start_x, start_y = 50, 50  # Initial position
move_x, move_y = 5, 0  # Move 5 pixels per frame to the right
disappear_frame = 100  # Frame number after which the person disappears

# Iterate over the sequence of images
for i, (rgb_path, depth_path) in enumerate(zip(rgb_images, depth_images)):
    frame = cv2.imread(rgb_path)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Extract the original filenames
    rgb_filename = os.path.basename(rgb_path)
    depth_filename = os.path.basename(depth_path)

    # Output paths
    output_rgb_path = os.path.join("/home/ak/GuidedResearch/data/output/rgb", rgb_filename)
    output_depth_path = os.path.join("/home/ak/GuidedResearch/data/output/depth", depth_filename)

    # If within the first 200 frames, overlay the person
    if i < disappear_frame:
        # Resize the person image if necessary
        person_resized = cv2.resize(person_img, (100, 200))  # Adjust dimensions
        if not use_fixed_depth:
            person_depth_resized = cv2.resize(person_depth, (100, 200))  # Resize depth if available

        # Compute new position (ensuring it doesn't go out of bounds)
        x = start_x + i * move_x
        y = start_y + i * move_y
        x = min(x, frame.shape[1] - person_resized.shape[1])  # Keep within frame width
        y = min(y, frame.shape[0] - person_resized.shape[0])  # Keep within frame height

        # Extract the alpha channel if available
        if person_resized.shape[2] == 4:  # Check if it has an alpha channel
            person_rgb = person_resized[:, :, :3]
            mask = person_resized[:, :, 3] / 255.0  # Normalize alpha to 0-1

            # Get ROI in the background image
            h, w, _ = person_rgb.shape
            roi_rgb = frame[y:y+h, x:x+w]
            roi_depth = depth_map[y:y+h, x:x+w]

            # Blend using the alpha mask
            for c in range(3):
                roi_rgb[:, :, c] = roi_rgb[:, :, c] * (1 - mask) + person_rgb[:, :, c] * mask

            # Update depth values
            if use_fixed_depth:
                roi_depth[mask > 0] = fixed_depth_value  # Assign fixed depth value
            else:
                roi_depth[mask > 0] = person_depth_resized[:, :, 0][mask > 0]  # Use depth map

            # Place the blended regions back into the frame
            frame[y:y+h, x:x+w] = roi_rgb
            depth_map[y:y+h, x:x+w] = roi_depth

        else:  # If no alpha channel, just overlay the person
            frame[y:y+person_resized.shape[0], x:x+person_resized.shape[1]] = person_resized
            if use_fixed_depth:
                depth_map[y:y+person_resized.shape[0], x:x+person_resized.shape[1]] = fixed_depth_value
            else:
                depth_map[y:y+person_resized.shape[0], x:x+person_resized.shape[1]] = person_depth_resized[:, :, 0]

    # Save the modified RGB and depth images with the same names
    cv2.imwrite(output_rgb_path, frame)
    cv2.imwrite(output_depth_path, depth_map)

    # Show frame (optional)
    cv2.imshow("RGB Frame", frame)
    cv2.imshow("Depth Frame", depth_map)
    cv2.waitKey(50)  # Adjust speed of movement

cv2.destroyAllWindows()
