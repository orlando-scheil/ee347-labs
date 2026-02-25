import os
import numpy as np
from sympy  import symbols, cos, sin, atan2, pi, Matrix, lambdify
from scipy.optimize import least_squares
from mecharm_INVK_group_05 import symbolic_forward_kinematics
from mecharm_INVK_group_05 import q_sym
from mecharm_INVK_group_05 import dh_table

# ! NOTE: Roll, pitch, and yaw are more susceptible to error correction than the position
# ! FIX IS TO EXPLICITELY WEIGHT THE POSITION ERROR MORE THAN THE ORIENTATION ERROR

# To get rid of the warning, we can use the following line of code
link_lengths = dh_table

forward_kinematics_func = lambdify(q_sym, symbolic_forward_kinematics(q_sym), 'numpy')

# Converts a homogeneous transformation matrix into a pose vector [X, Y, Z, roll, pitch, yaw]
# The rotation convention matches the one used in mecharm_INVK_group_05.py: XYZ (roll-pitch-yaw)
def transf_to_pose(t_matrix):
    # Extract position components
    X = t_matrix[0, 3]
    Y = t_matrix[1, 3]
    Z = t_matrix[2, 3]

    # Extract rotation matrix
    R = np.array(t_matrix[0:3, 0:3]).astype(np.float64)

    # Compute Euler angles using XYZ convention (roll-pitch-yaw)
    # roll (rx)
    roll = np.arctan2(R[2, 1], R[2, 2])
    # pitch (ry)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    # yaw (rz)
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return X, Y, Z, roll, pitch, yaw

# Checks the current joint angles against the target pose and returns their delta
def target_pose_error(joint_angles, *args):
    target_pose = args[0] # target pose vector [X, Y, Z, roll, pitch, yaw]
    current_fk = forward_kinematics_func(*joint_angles) # current forward kinematics transformation matrix

    # Convert translation matrix to pose vector
    current_pose = transf_to_pose(current_fk) # current pose vector [X, Y, Z, roll, pitch, yaw]

    error = target_pose - current_pose # difference between target and current pose
    return error

# Finds a solution to inverse kinematics for a target pose using nonlinear least-squares
def ik(target_pose, init_joints, max_iter=1000, tolerance=1e-5, bounds=None):
    """
    Solves for joint angles that achieve a given end-effector pose.

    Args:
        target_pose: desired pose vector [X, Y, Z, roll, pitch, yaw]
        init_joints: initial guess for joint angles (length 6)
        max_iter: maximum number of optimization iterations
        tolerance: convergence tolerance
        bounds: optional tuple for joint angle limits (lower, upper); defaults to no bounds

    Returns:
        joint_angles: array of 6 joint values if solution found, else None
    """
    if bounds is None:
        # Example: no bounds
        bounds = (-np.pi, np.pi)
    result = least_squares(
        target_pose_error,
        init_joints,
        args=(np.array(target_pose),),
        method='trf',
        max_nfev=max_iter,
        ftol=tolerance,
        bounds=bounds
    )
    if result.success:
        # print(f"Inverse kinematics converged after {result.nfev} function evaluations.")
        return result.x
    else:
        print("Inverse kinematics did not converge.")
        return None


if __name__ == "__main__":
    # Load robot poses from Lab 2 CSV: first 6 columns are
    # [X, Y, Z, roll_deg, pitch_deg, yaw_deg], next 6 are joint angles (deg).
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "..", "lab2", "robot_poses.csv")
    data = np.loadtxt(csv_path, delimiter=",")

    print("Testing advanced IK on poses from robot_poses.csv\n")

    for idx, row in enumerate(data, start=1):
        x_target, y_target, z_target = row[0:3]
        roll_deg, pitch_deg, yaw_deg = row[3:6]
        q_measured_deg = row[6:12]

        # Convert pose and initial guess angles to radians
        rx_d, ry_d, rz_d = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
        q_init = np.deg2rad(q_measured_deg)

        joint_angles = ik(
            target_pose=[x_target, y_target, z_target, rx_d, ry_d, rz_d],
            init_joints=q_init,
            max_iter=1000,
            tolerance=1e-5,
            bounds=None,
        )
        joint_angles_deg = np.degrees(joint_angles)

        # Forward kinematics and resulting pose
        T_fk = forward_kinematics_func(*joint_angles)
        X_fk = T_fk[0, 3]
        Y_fk = T_fk[1, 3]
        Z_fk = T_fk[2, 3]
        fk_pose = transf_to_pose(T_fk)
        _, _, _, roll_fk, pitch_fk, yaw_fk = fk_pose
        roll_fk_deg, pitch_fk_deg, yaw_fk_deg = np.degrees(
            [roll_fk, pitch_fk, yaw_fk]
        )

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
        print()
