"""
Minimal script to read MechArm 270 positions.
Uses pymycobot API: power_on, release_all_servos, get_angles, power_off.
Press Enter to read position; ^C to exit (robot is cleanly shut down).
On exit, prints a copy-pasteable array of all collected positions.
"""
import sys
import time
from pymycobot import MyCobot

MIN_ANGLES = [-165, -90, -180, -160, -115, -175]
MAX_ANGLES = [165, 90, 65, 160, 115, 175]

# Port/baud per API: M5=COM3/115200, Pi=/dev/ttyAMA0/1000000
PORT = "COM3" if sys.platform == "win32" else "/dev/ttyAMA0"
BAUD = 115200 if sys.platform == "win32" else 1000000

def within_bounds(position, min_angles, max_angles):
    angles = position[-6:] if len(position) >= 6 else position
    if len(angles) != 6:
        return None
    for i in range(6):
        if angles[i] < min_angles[i] or angles[i] > max_angles[i]:
            return None
    return position

collected = []  # [x, y, z, rx, ry, rz, q1..q6] per position
robot = None
try:
    robot = MyCobot(PORT, str(BAUD))
    robot.power_on()
    time.sleep(0.5)
    robot.release_all_servos()
    print("Connected. Press Enter to read position (^C to exit)...")
    while True:
        sys.stdin.readline()
        angles = robot.get_angles()
        coords = robot.get_coords()
        # Store full pose: coords [x,y,z,rx,ry,rz] + angles [q1..q6]
        # if coords and angles:
        #     row = list(coords) + list(angles)
        #     collected.append(row)
        # print(f"  [{len(collected)}] angles: {angles}  |  coords: {coords}")
        if coords and angles:
            row = list(coords) + list(angles)
            if within_bounds(row, MIN_ANGLES, MAX_ANGLES):
                collected.append(row[:6])
                print(f"  [{len(collected)}] angles: {angles}  |  coords: {coords}")
except KeyboardInterrupt:
    print("\nExiting.")
    if collected:
        print("\n--- Copy-paste array below ---")
        print(collected)
        print("--- End ---")
finally:
    if robot:
        robot.release_all_servos()
        time.sleep(0.5)
        robot.power_off()
        print("Robot shut down.")
