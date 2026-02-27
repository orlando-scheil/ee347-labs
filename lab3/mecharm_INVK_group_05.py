import os
import numpy as np
from sympy  import symbols, cos, sin, atan2, pi, Matrix, lambdify
from sympy.simplify import simplify
from scipy.optimize import least_squares

# Define symbolic joint angles
q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

# Apply offset accommodations
# MechArm 270 has a 90-degree offset on Joint 2 (from vertical to horizontal)
offsets = [0, -pi/2, 0, 0, 0, 0]  # Joint 2 has -90 degree offset

# D-H Table
dh_table = [
    # [a(n-1), alpha(n-1), d(n),    theta(n)]
    [0.0,      0,           87,       q1],
    [0,       -pi/2,        0.0,      q2 + offsets[1] + 0.1221],
    [110.0,    0.0,         0.0,      q3],
    [20,      -pi/2,        90,       q4],
    [0.0,      pi/2,        0,        q5],
    [0.0,     -pi/2,        55,       q6]
]

# To get rid of the warning, we can use the following line of code
link_lengths = dh_table

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

# Define symbolic joint variables for differentiation
q_sym = symbols('q1:7')

# calculate the forward kinematics symbolically using the DH parameters. This will help us obtain the end-effector position and orientation as symbolic expressions, which are later used for inverse kinematics.
def symbolic_forward_kinematics(q_values):
    """
    Calculate the symbolic forward kinematics transformation matrix (end-effector frame)
    using the symbolic DH-parameter transformation matrix with substituted joint values.
    Args:
        q_values: list or tuple of 6 numerical joint values [q1, q2, q3, q4, q5, q6]
    Returns:
        T_symbolic: 4x4 numpy or sympy Matrix, forward kinematics result
    """
    # Prepare the mapping from q1...q6 symbols to provided values
    q_syms = [q1, q2, q3, q4, q5, q6]
    subs_dict = {k: v for k, v in zip(q_syms, q_values)}
    T_symbolic = T_total.subs(subs_dict)
    return T_symbolic

# convert the symbolic forward kinematics to a numerical function that can be used for optimization.
forward_kinematics_func = lambdify(q_sym, symbolic_forward_kinematics(q_sym), 'numpy')

def position_error(q_position, x_target, y_target, z_target):
    """
    Compute the position error between forward kinematics result and a target XYZ position.

    Args:
        q_position: iterable of 3 joint values [q1, q2, q3].
        x_target, y_target, z_target: desired end-effector coordinates.

    Returns:
        list of [x_err, y_err, z_err].
    """
    # Calculate forward kinematics for the first three joints, with last three at zero
    T_values = forward_kinematics_func(q_position[0], q_position[1], q_position[2], 0, 0, 0)
    # Extract XYZ from the resulting transformation matrix
    X = T_values[0, 3]
    Y = T_values[1, 3]
    Z = T_values[2, 3]
    # Return the difference
    return [X - x_target, Y - y_target, Z - z_target]

def orientation_error(q_orientation, rx_d, ry_d, rz_d):
    """
    Compute the orientation error between desired Euler angles (rx_d, ry_d, rz_d)
    and the robot's current orientation determined by the last three joint values.

    Args:
        q_orientation: iterable of 3 joint values [q4, q5, q6] for the last three joints.
        rx_d, ry_d, rz_d: desired end-effector Euler angles (roll, pitch, yaw) in radians.

    Returns:
        list: [roll_error, pitch_error, yaw_error]
    """
    # Calculate forward kinematics with the last three joints set as q_orientation.
    # The first three joints are held at zero.
    T_values = forward_kinematics_func(0, 0, 0, q_orientation[0], q_orientation[1], q_orientation[2])

    # Extract the 3x3 rotation matrix from the 4x4 transformation matrix
    R = np.array(T_values[0:3, 0:3]).astype(np.float64)

    # Compute Euler angles (roll, pitch, yaw) using XYZ convention (roll-pitch-yaw)
    # roll (rx)
    roll = np.arctan2(R[2,1], R[2,2])
    # pitch (ry)
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    # yaw (rz)
    yaw = np.arctan2(R[1,0], R[0,0])

    # Compute orientation errors
    roll_error = roll - rx_d
    pitch_error = pitch - ry_d
    yaw_error = yaw - rz_d

    return [roll_error, pitch_error, yaw_error]


def inverse_kinematics(x_target, y_target, z_target, rx_d, ry_d, rz_d, q_init, link_lengths, max_iterations=100, tolerance=1e-6):
    """
    Perform numerical inverse kinematics to solve for joint angles that achieve the desired
    end-effector position (x_target, y_target, z_target) and orientation (rx_d, ry_d, rz_d).

    Args:
        x_target, y_target, z_target: desired end-effector position.
        rx_d, ry_d, rz_d: desired end-effector orientation (roll, pitch, yaw).
        q_init: initial guess for all 6 joint angles [q1, q2, q3, q4, q5, q6].
        link_lengths: robot's DH parameters or link lengths (not used explicitly here, but kept for interface compatibility).
        max_iterations: maximum number of optimization steps.
        tolerance: solution convergence tolerance.

    Returns:
        joint_angles: array of 6 joint angles [q1, q2, q3, q4, q5, q6].
    """

    # Perform numerical inverse kinematics for position
    position_args = (x_target, y_target, z_target)
    q_position_solution = least_squares(
        position_error,
        q_init[:3],
        args=position_args,
        method='lm',
        max_nfev=max_iterations,
        ftol=tolerance
    ).x

    # Perform numerical inverse kinematics for orientation
    orientation_args = (rx_d, ry_d, rz_d)
    q_orientation_solution = least_squares(
        orientation_error,
        q_init[3:],
        args=orientation_args,
        method='lm',
        max_nfev=max_iterations,
        ftol=tolerance
    ).x

    # Combine the position and orientation components to get the final joint angles
    joint_angles = np.concatenate((q_position_solution, q_orientation_solution))
    return joint_angles


if __name__ == "__main__":
    # Load robot poses from Lab 2 CSV: first 6 columns are
    # [X, Y, Z, roll_deg, pitch_deg, yaw_deg], next 6 are joint angles (deg).
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "robot_poses.csv")
    data = np.loadtxt(csv_path, delimiter=",")

    print("Testing inverse_kinematics on poses from robot_poses.csv\n")

    for idx, row in enumerate(data, start=1):
        x_target, y_target, z_target = row[0:3]
        roll_deg, pitch_deg, yaw_deg = row[3:6]
        # q_measured_deg = row[6:12]
        q_measured_deg = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # Convert pose and initial guess angles to radians
        rx_d, ry_d, rz_d = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
        q_init = np.deg2rad(q_measured_deg)

        # Solve IK
        joint_angles = inverse_kinematics(
            x_target, y_target, z_target, rx_d, ry_d, rz_d, q_init, link_lengths
        )
        joint_angles_deg = np.degrees(joint_angles)

        # Forward kinematics of IK solution to get resulting pose
        T_fk = forward_kinematics_func(*joint_angles)
        X_fk = T_fk[0, 3]
        Y_fk = T_fk[1, 3]
        Z_fk = T_fk[2, 3]
        R_fk = np.array(T_fk[0:3, 0:3]).astype(np.float64)
        roll_fk = np.arctan2(R_fk[2, 1], R_fk[2, 2])
        pitch_fk = np.arctan2(-R_fk[2, 0], np.sqrt(R_fk[2, 1] ** 2 + R_fk[2, 2] ** 2))
        yaw_fk = np.arctan2(R_fk[1, 0], R_fk[0, 0])
        roll_fk_deg, pitch_fk_deg, yaw_fk_deg = np.degrees([roll_fk, pitch_fk, yaw_fk])

        # Clean, comparable printout
        print(f"Pose {idx}:")
        print(
            f"  Target pose (mm, deg): "
            f"X={x_target:.2f}, Y={y_target:.2f}, Z={z_target:.2f}, "
            f"roll={roll_deg:.2f}, pitch={pitch_deg:.2f}, yaw={yaw_deg:.2f}"
        )
        print(f"  Measured joints (deg): {q_measured_deg}")
        print(f"  IK joints (deg):       {joint_angles_deg}")
        print(
            f"  FK from IK (mm, deg):  "
            f"X={X_fk:.2f}, Y={Y_fk:.2f}, Z={Z_fk:.2f}, "
            f"roll={roll_fk_deg:.2f}, pitch={pitch_fk_deg:.2f}, yaw={yaw_fk_deg:.2f}"
        )
        # Position error (mm)
        pos_err_x = X_fk - x_target
        pos_err_y = Y_fk - y_target
        pos_err_z = Z_fk - z_target
        pos_err_mm = np.sqrt(pos_err_x**2 + pos_err_y**2 + pos_err_z**2)
        print(
            f"  Position error (mm):   "
            f"ΔX={pos_err_x:.4f}, ΔY={pos_err_y:.4f}, ΔZ={pos_err_z:.4f}, "
            f"|err|={pos_err_mm:.4f}"
        )
        # Orientation error (deg)
        orient_err_roll = roll_fk_deg - roll_deg
        orient_err_pitch = pitch_fk_deg - pitch_deg
        orient_err_yaw = yaw_fk_deg - yaw_deg
        orient_err_deg = np.sqrt(orient_err_roll**2 + orient_err_pitch**2 + orient_err_yaw**2)
        print(
            f"  Orientation error (deg): "
            f"Δroll={orient_err_roll:.4f}, Δpitch={orient_err_pitch:.4f}, Δyaw={orient_err_yaw:.4f}, "
            f"|err|={orient_err_deg:.4f}"
        )
        print()
