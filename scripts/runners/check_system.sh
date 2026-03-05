#!/bin/bash

# =============================================================================
# Pre-flight System Verification Script
# =============================================================================

# 1. Vision: Aruco Marker Real-time Tracking
# Checks DJI camera stream and Reprojection accuracy
python scripts/hardware_tests/realtime_aruco_tracking.py

# 2. Force/Tactile: F/T Sensor & Gravity Compensation
# Requires ROS environment for Aidin FT Bridge
conda activate ros_env && roscore
conda activate ros_env && python scripts/hardware_tests/check_gravity_compensator.py --gravity_compensate --robot

# 3. Control: Franka Panda Teleoperation
# Verifies SpaceMouse integrated Shared Memory control
python scripts/hardware_tests/teleop_panda_sm.py
