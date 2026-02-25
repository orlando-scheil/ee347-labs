from pymycobot import MechArm270
from pymycobot import PI_PORT, PI_BAUD
from pymycobot.mycobot import MyCobot
import time
import sys
import csv
import select
import os

# Initialize the robot arm
mycobot = MyCobot(PI_PORT, PI_BAUD)
mycobot.power_on()
mycobot.release_all_servos()

# CSV file setup - append if exists, otherwise create with headers
file_exists = os.path.isfile('robot_poses.csv')
csv_file = open('robot_poses.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)

def angle_is_valid(angles):
    min_angle = [-165, -90, -180, -160, -115, -175]
    max_angle = [165, 90, 65, 160, 115, 175]

    for i in range(len(angles)):
        if angles[i] < min_angle[i] or angles[i] > max_angle[i]:
            return False

    return True

try:
    while True:
        coords = mycobot.get_coords()
        angles = mycobot.get_angles()

        # Check for input
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.readline().strip()
            if key == ' ' or key == '':  # Spacebar or Enter
                row = coords + angles
                if angle_is_valid(angles):
                    csv_writer.writerow(row)
                    csv_file.flush()
                    print(f"{coords}", "{angles}")
                    print(" ============== Pose captured ==============")
            elif key == 'q':
                break
except KeyboardInterrupt:
    print('\nExiting.')
finally:
    csv_file.close()
    mycobot.power_off()
    print('Saved to robot_poses.csv')

