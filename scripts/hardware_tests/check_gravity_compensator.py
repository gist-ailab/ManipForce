#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import queue
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter
from tqdm import trange
import os
import cv2
from collections import deque
from utils.ft_capture import AidinFTSensorUDP
import argparse

import rospy
from geometry_msgs.msg import WrenchStamped
import pyrealsense2 as rs
from ahrs.filters import Madgwick

import torch

# Import gravity compensation utilities and RealSense IMU helper functions
from utils.gravity_compensation_utils import GravityCompensator

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Gravity compensation calibration')
    parser.add_argument('--gravity_compensate', action='store_true',
                       help='Enable gravity compensation (default: False, show raw values only)')
    parser.add_argument('--robot', action='store_true',
                       help='Robot mode (default: False, Gumi mode)')
    parser.add_argument('--use_ft_imu', action='store_true',
                       help='Use FT built-in IMU instead of RealSense for gravity compensation (experimental)')
    parser.add_argument('--global_top_setup', action='store_true',
                       help='Use GlobalTop setup parameters (same mass/COM as Gumi default)')
    parser.add_argument('--port', type=int, default=50000, help='FT sensor port')
    args = parser.parse_args()
    
    print("Starting program")
    print(f"Gravity compensation: {'ON' if args.gravity_compensate else 'OFF (raw values only)'}")

    rospy.init_node('gravity_compensation_calibration', anonymous=True)
    pub = rospy.Publisher('/ft300/wrench', WrenchStamped, queue_size=10)
    rate = rospy.Rate(200)  # target 200 Hz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== Start FT sensor ======
    print("Initializing FT sensor...")
    # Ports: 8890, 50000
    port = args.port
    ft_reader = AidinFTSensorUDP('172.27.190.4', port)
    ft_reader.start()
    print("FT sensor initialized.")

    # ====== IMU setup (RealSense; for comparison with FT IMU) ======
    print("Initializing IMU (RealSense)...")
    imu_pipe = rs.pipeline()
    imu_cfg = rs.config()
    imu_cfg.enable_stream(rs.stream.accel)
    imu_cfg.enable_stream(rs.stream.gyro)
    imu_pipe.start(imu_cfg)
    print("IMU initialized.")

    # ====== Initialize gravity compensator ======
    gravity_compensator = None
    global_top_setup = args.global_top_setup

    if args.gravity_compensate:
        # Initialize gravity compensator
        # Gumi Setup
        if args.robot:
            # Robot Setup
            gravity_compensator = GravityCompensator(
                mass_for_x=0.73,
                mass_for_y=0.73,
                mass_for_z=0.73,
                com_ft=np.array([0.0, 0.01, 0.03]),
                g_const=9.81,
                madgwick_frequency=200.0,
                filter_fs=200.0
            )
        elif global_top_setup:
            gravity_compensator = GravityCompensator(
                mass_for_x=0.0,
                mass_for_y=0.0,
                mass_for_z=0.0,
                com_ft=np.array([0.0, 0.0, 0.0]),
                g_const=9.81
            )
        else:
            # Gumi Setup
            gravity_compensator = GravityCompensator(
                mass_for_x=0.58,
                mass_for_y=0.53, 
                mass_for_z=0.7,
                com_ft=np.array([0.01, 0.01, 0.03]),
                g_const=9.81
            )
        
        if args.use_ft_imu:
            # Initialize for FT IMU-based gravity compensation (no RealSense needed)
            print("[FT-IMU] Initializing for FT IMU-based gravity compensation (gyro bias, pose warmup)...")
            gravity_compensator.calibrate_ft_imu_bias(
                ft_reader,
                warmup_sec=5.0,
                gyro_scale=np.array([0.1, 0.1, 0.1])
            )
            # Explicitly set the internal flag (safety guard)
            gravity_compensator.use_ft_imu = True
            print("[FT-IMU] Initialization complete.")
        else:
            # Collect baseline using existing RealSense IMU path
            print("Collecting baseline data (RealSense IMU)...")
            gravity_compensator.calibrate_baseline(imu_pipe, ft_reader, warmup_sec=5.0)
            gravity_compensator.use_ft_imu = False
            print("Baseline data collected.")
    

    # Initial bias (used across all modes)
    f_bias_initial = None
    t_bias_initial = None
    settle_count = 0  # Frame counter for the IMU convergence warm-up period

    # Real-time control loop
    while not rospy.is_shutdown():
        if args.gravity_compensate:
            # Gravity-compensation mode
            # Pass use_ft_imu flag into process_single_frame so it selects RealSense or FT IMU internally
            compensated_force, compensated_torque, debug_info = gravity_compensator.process_single_frame(
                imu_pipe=imu_pipe,
                ft_reader=ft_reader,
                use_ft_imu=gravity_compensator.use_ft_imu,
            )
            
            if compensated_force is not None:
                # Remove initial bias
                if f_bias_initial is None:
                    # In FT IMU mode, wait for the IMU to converge (skip early frames)
                    if gravity_compensator.use_ft_imu:
                        settle_count += 1
                        if settle_count < 10000:  # wait roughly 1–2 seconds then capture once
                            continue
                        # After the initial bias is captured, no need to count again
                        settle_count = 0

                    f_bias_initial = compensated_force.copy()
                    t_bias_initial = compensated_torque.copy()

                # Remove initial bias
                f_final = compensated_force - f_bias_initial
                # Subtract additional fy offset
                if args.robot:
                    f_final[1] -= 3.0
                else:
                    f_final[1] -= 0.0
                t_final = compensated_torque - t_bias_initial
                
 
                # Publish final compensated wrench
                msg = WrenchStamped()
                msg.header.stamp = rospy.Time.now()
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z   = f_final
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = t_final
                pub.publish(msg)
            
            gravity_compensator.increment_loop_count()
            
        else:
            # Raw value mode (zero-out via bias subtraction)
            try:
                ts, f_raw, t_raw = ft_reader.read_latest(timeout=0.001)
                
                # Convert to numpy arrays
                f_raw = np.array(f_raw)
                t_raw = np.array(t_raw)
                
                # Simple bias removal (initialise from the first sample)
                if f_bias_initial is None:
                    f_bias_initial = f_raw.copy()
                    t_bias_initial = t_raw.copy()
                    print(f"Raw initial bias set: Force={f_bias_initial}, Torque={t_bias_initial}")
                
                # Subtract bias
                f_zeroed = f_raw - f_bias_initial
                t_zeroed = t_raw - t_bias_initial
                
                # Publish bias-subtracted values
                msg = WrenchStamped()
                msg.header.stamp = rospy.Time.now()
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z   = f_zeroed
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = t_zeroed
                pub.publish(msg)
                
            except queue.Empty:
                continue
        
        rate.sleep()
        
        