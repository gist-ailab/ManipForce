# Make ArUco marker sheet for gripper

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# A4 paper size in mm
A4_WIDTH_MM  = 210.0
A4_HEIGHT_MM = 297.0

# Marker settings (small)
border_size_mm = 12  # outer border 13mm
marker_size_mm = border_size_mm * (24 / 28) # preserve original 24/28 ratio
border_thickness_mm = 0.2

DPI = 300
MM_PER_INCH = 25.4

def mm_to_px(mm_val):
    return int(mm_val * DPI / MM_PER_INCH)

# Canvas size in pixels
A4_WIDTH_PX  = mm_to_px(A4_WIDTH_MM)   # 2480
A4_HEIGHT_PX = mm_to_px(A4_HEIGHT_MM)  # 3508

marker_size_px     = mm_to_px(marker_size_mm)
border_size_px     = mm_to_px(border_size_mm)
border_thickness_px = max(1, mm_to_px(border_thickness_mm))

# Create A4 canvas (white background)
a4_image = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype=np.uint8) * 255

# Select ArUco marker dictionary
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

# Marker spacing
margin = 100  # gap between markers (pixels)

# Starting position for marker layout (top-left corner)
x_offset = 100  # left margin
y_offset = 100  # top margin

# Draw 6 sets of markers (each set contains IDs 6 and 7)
set_count  = 6
marker_ids = [6, 7]  # marker IDs per set

for set_idx in range(set_count):
    row = set_idx // 3  # 0 or 1
    col = set_idx % 3   # 0, 1, or 2

    for i, marker_id in enumerate(marker_ids):
        x = x_offset + col * (2 * border_size_px + margin) + i * (border_size_px + 50)
        y = y_offset + row * (border_size_px + margin)

        # Black border rectangle
        cv2.rectangle(a4_image, (x, y), (x + border_size_px, y + border_size_px),
                      0, border_thickness_px)

        # Generate ArUco marker (generateImageMarker replaces drawMarker in OpenCV 4.7+)
        marker_image = aruco.generateImageMarker(dictionary, marker_id, marker_size_px)

        # Place marker at the centre of the border
        marker_offset = (border_size_px - marker_size_px) // 2
        marker_x = x + marker_offset
        marker_y = y + marker_offset
        a4_image[marker_y:marker_y + marker_size_px, marker_x:marker_x + marker_size_px] = marker_image

# --- Save as PDF with exact A4 physical dimensions ---
fig = plt.figure(figsize=(A4_WIDTH_MM / MM_PER_INCH, A4_HEIGHT_MM / MM_PER_INCH))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(a4_image, cmap='gray', vmin=0, vmax=255,
          extent=[0, A4_WIDTH_MM, A4_HEIGHT_MM, 0])
ax.set_xlim(0, A4_WIDTH_MM)
ax.set_ylim(A4_HEIGHT_MM, 0)
ax.axis('off')

output_folder = "calibration"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "aruco_markers_gripper.pdf")

fig.savefig(output_path, format='pdf', dpi=DPI, bbox_inches=None)
plt.close(fig)

print(f"Saved to: {output_path}")
print(f"Page size: {A4_WIDTH_MM}mm x {A4_HEIGHT_MM}mm  |  Marker: {marker_size_mm}mm  |  Border: {border_size_mm}mm")
print("Print at 100% (Actual Size) — do NOT use 'Fit to page'.")
