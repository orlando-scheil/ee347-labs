from pymycobot import MechArm270
from pymycobot import PI_PORT, PI_BAUD
from pymycobot.mycobot import MyCobot
import numpy as np
from sympy import symbols, cos, sin, pi, simplify, Matrix
import csv
import time

# Initialize robot
mycobot = MyCobot(PI_PORT, PI_BAUD)
mycobot.power_on()
mycobot.release_all_servos()

# Define symbolic joint angles
q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

# D-H Table
dh_table = [
    # [a(n-1), alpha(n-1), d(n),    theta(n)]
    [0.0,      0,           87,       q1],
    [0,       -pi/2,        0.0,      q2],
    [110.0,    0.0,         0.0,      q3],
    [20,      -pi/2,        90,       q4],
    [0.0,      pi/2,        0,        q5],
    [0.0,     -pi/2,        55,       q6]
]

# Transformation matrix function (as provided in your class)
def get_transformation_matrix(a, alpha, d, theta):
    """
    Creates a 4x4 homogeneous transformation matrix from D-H parameters
    Note: alpha and a correspond to link (n-1)
    """
    M = Matrix([
        [cos(theta),            -sin(theta),             0,           a],
        [sin(theta)*cos(alpha),  cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
        [sin(theta)*sin(alpha),  cos(theta)*sin(alpha),  cos(alpha),  cos(alpha)*d],
        [0,                      0,                       0,                     1]
    ])
    return M

# Calculate overall transformation matrix
print("Calculating transformation matrices")
T_total = Matrix.eye(4)  # Start with identity matrix

for row in dh_table:
    a, alpha, d, theta = row
    T_i = get_transformation_matrix(a, alpha, d, theta)
    T_total = T_total * T_i

print("Simplifying transformation matrix")
T_total = simplify(T_total)

# Apply offset accommodations
# MechArm 270 has a 90-degree offset on Joint 2 (from vertical to horizontal)
offsets = [0, -pi/2, 0, 0, 0, 0]  # Joint 2 has -90 degree offset

# Extract end effector position function
def forward_kinematics(joint_angles_deg):
    """
    Calculate end effector position from joint angles
    joint_angles_deg: list of 6 joint angles in DEGREES
    Returns: [x, y, z] in mm
    """
    # Convert degrees to radians
    joint_angles_rad = [np.radians(angle) for angle in joint_angles_deg]

    # Create substitution dictionary with offsets
    subs_dict = {
        q1: offsets[0] + joint_angles_rad[0],
        q2: offsets[1] + joint_angles_rad[1],
        q3: offsets[2] + joint_angles_rad[2],
        q4: offsets[3] + joint_angles_rad[3],
        q5: offsets[4] + joint_angles_rad[4],
        q6: offsets[5] + joint_angles_rad[5]
    }

    # Substitute values into transformation matrix
    T_numeric = T_total.subs(subs_dict)

    # Extract position (first 3 elements of last column)
    x = float(T_numeric[0, 3])
    y = float(T_numeric[1, 3])
    z = float(T_numeric[2, 3])

    return [x, y, z]

# Validation - Load your Challenge 2 data
print("\n=== VALIDATION ===")
print("Reading data from Challenge 2...")


with open('robot_poses.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        angles = list(map(float, row[6:12]))
        angles[1] = angles[1] - 7

        predicted_pos = forward_kinematics(angles)
        print(f"Your FK predicts position: {predicted_pos}")

        # Move robot to those angles
        mycobot.send_angles(angles, 50)
        time.sleep(1)

        # Get actual position from robot
        actual_pos = mycobot.get_coords()[0:3]  # Only x, y, z
        print(f"Robot reports position: {actual_pos}")

        error = np.linalg.norm(np.array(predicted_pos) - np.array(actual_pos))
        print(f"Position error: {error:.2f} mm")


# Return to zero position
print("\nReturning to home position...")
mycobot.send_angles([0, 0, 0, 0, 0, 0], 50)
time.sleep(3)
mycobot.power_off()
print("Done!")
