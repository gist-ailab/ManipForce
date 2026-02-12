# Aruco marker realtime tracking
python scripts/hardware_tests/realtime_aruco_tracking.py

# Check FT data w gravity compensation
conda activate ros_env && roscore
conda activate ros_env && python scripts/hardware_tests/check_gravity_compensator.py --robot

# Franka panda Teleop
python scripts/hardware_tests/teleop_panda.py
