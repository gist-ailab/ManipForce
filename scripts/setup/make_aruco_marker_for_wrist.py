
# Make ArUco marker sheet for wrist
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

# Marker settings (physical size in mm)
border_size_mm = 59   # black border ~5.9cm
marker_size_mm = 53   # marker slightly smaller than border

DPI = 300
MM_PER_INCH = 25.4

def mm_to_px(mm_val):
    return int(mm_val * DPI / MM_PER_INCH)

# Canvas size in pixels
A4_WIDTH_PX  = mm_to_px(A4_WIDTH_MM)   # 2480
A4_HEIGHT_PX = mm_to_px(A4_HEIGHT_MM)  # 3508

border_size_px = mm_to_px(border_size_mm)
marker_size_px = mm_to_px(marker_size_mm)
border_thickness_px = max(1, mm_to_px(0.2))  # at least 1px

# Create A4 canvas (white background)
a4_image = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype=np.uint8) * 255

# Select ArUco marker dictionary
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

# Arrange 5 markers on a 3x3 grid layout (only top 5 used)
num_markers = 5
margin_px = 80  # gap between markers in pixels

x_offset = (A4_WIDTH_PX  - (3 * border_size_px + 2 * margin_px)) // 2
y_offset = (A4_HEIGHT_PX - (3 * border_size_px + 2 * margin_px)) // 2

marker_id = 0
for row in range(3):
    for col in range(3):
        if marker_id >= num_markers:
            break

        x = x_offset + col * (border_size_px + margin_px)
        y = y_offset + row * (border_size_px + margin_px)

        # Draw black border rectangle
        cv2.rectangle(a4_image, (x, y), (x + border_size_px, y + border_size_px),
                      0, border_thickness_px)

        # Generate ArUco marker (generateImageMarker replaces drawMarker in OpenCV 4.7+)
        marker_image = aruco.generateImageMarker(dictionary, marker_id, marker_size_px)

        # Centre marker inside border
        marker_offset = (border_size_px - marker_size_px) // 2
        mx = x + marker_offset
        my = y + marker_offset
        a4_image[my:my + marker_size_px, mx:mx + marker_size_px] = marker_image

        marker_id += 1
        print(f"Marker {marker_id} placed.")

# --- Save as PDF with exact A4 physical dimensions ---
# figsize is in inches; A4 = 210mm x 297mm
fig = plt.figure(figsize=(A4_WIDTH_MM / MM_PER_INCH, A4_HEIGHT_MM / MM_PER_INCH))
ax = fig.add_axes([0, 0, 1, 1])   # axes fills the entire figure
ax.imshow(a4_image, cmap='gray', vmin=0, vmax=255,
          extent=[0, A4_WIDTH_MM, A4_HEIGHT_MM, 0])
ax.set_xlim(0, A4_WIDTH_MM)
ax.set_ylim(A4_HEIGHT_MM, 0)
ax.axis('off')

output_folder = "calibration"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "aruco_markers_wrist.pdf")

# bbox_inches=None preserves the exact figure size (no padding/cropping)
fig.savefig(output_path, format='pdf', dpi=DPI, bbox_inches=None)
plt.close(fig)

print(f"\nSaved to: {output_path}")
print(f"Page size: {A4_WIDTH_MM}mm x {A4_HEIGHT_MM}mm  |  Marker: {marker_size_mm}mm  |  Border: {border_size_mm}mm")
print("Print at 100% (Actual Size) — do NOT use 'Fit to page'.")
