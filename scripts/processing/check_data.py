
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from scipy.spatial.transform import Rotation

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Check and visualize pose data')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the data directory (e.g., data/0314/episode_3)')
    args = parser.parse_args()
    
    # Load both JSON files
    base_save_path = args.data_path

    with open(os.path.join(base_save_path, 'raw_pose.json'), 'r') as f:
        raw_data = json.load(f)

    with open(os.path.join(base_save_path, 'pose_data.json'), 'r') as f:
        refined_data = json.load(f)

    # Extract data with image file matching
    def extract_matched_data(raw_data, refined_data):
        # Create dictionaries with image_file as key
        raw_dict = {entry['image_file']: entry for entry in raw_data if 'pose' in entry and entry['pose'] is not None}
        refined_dict = {entry['image_file']: entry for entry in refined_data}
        
        # Find common image files
        common_files = sorted(set(raw_dict.keys()) & set(refined_dict.keys()))
        
        timestamps = []
        raw_positions = []
        raw_orientations = []
        raw_gripper_states = []
        ref_positions = []
        ref_orientations = []
        ref_gripper_states = []
        
        for img_file in common_files:
            raw_entry = raw_dict[img_file]
            ref_entry = refined_dict[img_file]
            
            timestamps.append(raw_entry['timestamp'])
            raw_positions.append(raw_entry['pose']['position'])
            raw_orientations.append(raw_entry['pose']['orientation'])
            raw_gripper_states.append(raw_entry['pose'].get('gripper_state', 1.0))
            ref_positions.append(ref_entry['pose']['position'])  # absolute position
            ref_orientations.append(ref_entry['pose']['orientation'])  # absolute orientation
            ref_gripper_states.append(ref_entry['pose']['gripper_state'])
        
        return (np.array(timestamps), 
                np.array(raw_positions), 
                np.array(raw_orientations),
                np.array(raw_gripper_states),
                np.array(ref_positions), 
                np.array(ref_orientations),
                np.array(ref_gripper_states))

    def fix_quaternion_discontinuities(q):
        """Fixes discontinuities in quaternion data"""
        q_fixed = q.copy()
        for i in range(1, len(q)):
            # Calculate dot product with previous quaternion
            dot = np.sum(q[i-1] * q[i])
            # If the dot product is negative, negate this quaternion
            if dot < 0:
                q_fixed[i] = -q[i]
        return q_fixed

    # Extract matched data
    timestamps, raw_pos, raw_ori, raw_grip, ref_pos, ref_ori, ref_grip = extract_matched_data(raw_data, refined_data)

    # Fix quaternion discontinuities
    raw_ori_fixed = fix_quaternion_discontinuities(raw_ori)
    ref_ori_fixed = fix_quaternion_discontinuities(ref_ori)

    # Convert quaternions to Euler angles (degrees)
    raw_euler = Rotation.from_quat(raw_ori_fixed).as_euler('xyz', degrees=True)
    ref_euler = Rotation.from_quat(ref_ori_fixed).as_euler('xyz', degrees=True)

    # Create subplots: 3x3 grid, last row spans all columns
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, :])  # last row spans all columns

    # Plot positions
    ax1.plot(timestamps, raw_pos[:, 0], 'b-', label='Raw', alpha=0.5)
    ax1.plot(timestamps, ref_pos[:, 0], 'r-', label='Refined', alpha=0.5)
    ax1.set_title('Position X')
    ax1.legend()

    ax2.plot(timestamps, raw_pos[:, 1], 'b-', alpha=0.5)
    ax2.plot(timestamps, ref_pos[:, 1], 'r-', alpha=0.5)
    ax2.set_title('Position Y')

    ax3.plot(timestamps, raw_pos[:, 2], 'b-', alpha=0.5)
    ax3.plot(timestamps, ref_pos[:, 2], 'r-', alpha=0.5)
    ax3.set_title('Position Z')

    # Plot orientations (Euler angles)
    ax4.plot(timestamps, raw_euler[:, 0], 'b-', alpha=0.5)
    ax4.plot(timestamps, ref_euler[:, 0], 'r-', alpha=0.5)
    ax4.set_title('Roll (X rotation)')
    ax4.set_ylabel('Degrees')

    ax5.plot(timestamps, raw_euler[:, 1], 'b-', alpha=0.5)
    ax5.plot(timestamps, ref_euler[:, 1], 'r-', alpha=0.5)
    ax5.set_title('Pitch (Y rotation)')
    ax5.set_ylabel('Degrees')

    ax6.plot(timestamps, raw_euler[:, 2], 'b-', alpha=0.5)
    ax6.plot(timestamps, ref_euler[:, 2], 'r-', alpha=0.5)
    ax6.set_title('Yaw (Z rotation)')
    ax6.set_ylabel('Degrees')
    
    # Plot gripper state
    ax7.plot(timestamps, raw_grip, 'b-', label='Raw', alpha=0.5)
    ax7.plot(timestamps, ref_grip, 'r-', label='Refined', alpha=0.5)
    ax7.set_title('Gripper State')
    ax7.set_ylabel('State')
    ax7.legend()

    # Adjust overall layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()