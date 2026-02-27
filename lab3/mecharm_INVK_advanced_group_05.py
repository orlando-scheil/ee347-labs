# Where was AI used?
# AI was used to organize the code and break challenge 2 parts down into helper functions.
# AI was also used to fix the code and make it more readable.
# AI was also used to add comments to the code to make it more readable.
# AI was also used to add the warning and status messages to the code to make it more readable.
#

import time
import os
import numpy as np
from sympy  import symbols, cos, sin, atan2, pi, Matrix, lambdify
from scipy.optimize import least_squares
from mecharm_INVK_group_05 import symbolic_forward_kinematics
from mecharm_INVK_group_05 import q_sym
from mecharm_INVK_group_05 import dh_table
from pymycobot import MyCobot
from pymycobot import PI_BAUD, PI_PORT

# ! NOTE: Roll, pitch, and yaw are more susceptible to error correction than the position
# ! FIX IS TO EXPLICITELY WEIGHT THE POSITION ERROR MORE THAN THE ORIENTATION ERROR

min_angle = [-165, -90, -180, -145, -115, -175]
max_angle = [165, 90, 65, 145, 115, 175]
angle_bounds = (np.deg2rad(min_angle), np.deg2rad(max_angle))

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

    error = current_pose - target_pose # difference between target and current pose
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


# ---------------------------------------------------------------------------
# CSV test (callable, no robot needed)
# ---------------------------------------------------------------------------

def run_csv_test(csv_path=None):
    """
    Validate the advanced IK against recorded poses from a CSV file.
    Prints target vs. FK-reconstructed pose for each row.
    """
    from mecharm_INVK_group_05 import load_poses_from_csv

    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "robot_poses.csv")
    poses, measured_joints = load_poses_from_csv(csv_path)
    print("Testing advanced IK on poses from", csv_path, "\n")

    for idx, (pose, q_measured_deg) in enumerate(zip(poses, measured_joints), start=1):
        x, y, z = pose[0], pose[1], pose[2]
        roll_deg, pitch_deg, yaw_deg = pose[3], pose[4], pose[5]
        rx_d, ry_d, rz_d = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
        q_init = np.deg2rad(q_measured_deg)

        joint_angles = ik(
            target_pose=[x, y, z, rx_d, ry_d, rz_d],
            init_joints=q_init,
        )
        if joint_angles is None:
            print(f"Pose {idx}: IK did not converge — skipping\n")
            continue
        joint_angles_deg = np.degrees(joint_angles)

        T_fk = forward_kinematics_func(*joint_angles)
        fk_pose = transf_to_pose(T_fk)
        X_fk, Y_fk, Z_fk = fk_pose[0], fk_pose[1], fk_pose[2]
        roll_fk_deg = np.degrees(fk_pose[3])
        pitch_fk_deg = np.degrees(fk_pose[4])
        yaw_fk_deg = np.degrees(fk_pose[5])

        print(f"Pose {idx}:")
        print(f"  Target (mm, deg):    X={x:.2f}, Y={y:.2f}, Z={z:.2f}, "
              f"roll={roll_deg:.2f}, pitch={pitch_deg:.2f}, yaw={yaw_deg:.2f}")
        print(f"  Measured joints:     {q_measured_deg}")
        print(f"  IK joints (deg):     {joint_angles_deg}")
        print(f"  FK from IK (mm,deg): X={X_fk:.2f}, Y={Y_fk:.2f}, Z={Z_fk:.2f}, "
              f"roll={roll_fk_deg:.2f}, pitch={pitch_fk_deg:.2f}, yaw={yaw_fk_deg:.2f}")
        print()


# ---------------------------------------------------------------------------
# IK solver: batch of waypoints → joint angle arrays
# ---------------------------------------------------------------------------

def solve_waypoints(waypoints, q_init_deg=None, angle_bounds=None):
    """
    Run IK on a list of waypoints and return the solved joint angles.

    Args:
        waypoints: list of [x, y, z, rx_deg, ry_deg, rz_deg]  (mm and degrees).
        q_init_deg: optional initial guess in degrees (length-6 list).
                    Each solution is used as the guess for the next waypoint.
    Returns:
        List of length-6 joint-angle lists in degrees (or None where IK failed).
    """
    if q_init_deg is not None:
        q_prev = np.deg2rad(q_init_deg)
    else:
        q_prev = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    results = []
    for wp in waypoints:
        x, y, z, rx_deg, ry_deg, rz_deg = wp
        rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
        q_sol = ik(target_pose=[x, y, z, rx, ry, rz], init_joints=q_prev, bounds=angle_bounds)
        if q_sol is not None:
            q_prev = q_sol
            results.append(np.degrees(q_sol).tolist())
        else:
            print(f"WARNING: IK failed for waypoint {wp}")
            results.append(None)
    return results


def print_solved_waypoints(waypoints, solved_angles):
    """Pretty-print the IK results for a list of waypoints."""
    print(f"{'#':>3}  {'X':>8} {'Y':>8} {'Z':>8}  "
          f"{'rx':>8} {'ry':>8} {'rz':>8}  "
          f"{'q1':>8} {'q2':>8} {'q3':>8} {'q4':>8} {'q5':>8} {'q6':>8}")
    print("-" * 120)
    for i, (wp, ang) in enumerate(zip(waypoints, solved_angles), start=1):
        if ang is None:
            print(f"{i:>3}  {wp[0]:>8.2f} {wp[1]:>8.2f} {wp[2]:>8.2f}  "
                  f"{wp[3]:>8.2f} {wp[4]:>8.2f} {wp[5]:>8.2f}  *** IK FAILED ***")
        else:
            print(f"{i:>3}  {wp[0]:>8.2f} {wp[1]:>8.2f} {wp[2]:>8.2f}  "
                  f"{wp[3]:>8.2f} {wp[4]:>8.2f} {wp[5]:>8.2f}  "
                  f"{ang[0]:>8.2f} {ang[1]:>8.2f} {ang[2]:>8.2f} "
                  f"{ang[3]:>8.2f} {ang[4]:>8.2f} {ang[5]:>8.2f}")


# ---------------------------------------------------------------------------
# Main: waypoints → IK → solved_joint_angles (no robot execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    WAYPOINTS = [[198.0, 77.1, 75.2, -175.85, -17.85, 27.87], [155.0, 12.9, 256.2, 172.07, -64.7, 20.48], [133.4, -63.9, 266.2, 160.72, -68.28, -3.28], [179.2, -74.5, 76.1, 178.36, -11.13, -12.74], [152.7, -51.6, 222.2, -174.41, -53.0, -15.81], [150.1, 63.2, 230.7, -178.68, -60.85, 27.46], [189.8, 73.1, 66.1, -179.09, -20.42, 26.64]]

    # Initial joint guess in degrees [q1..q6]; passed as init_joints to ik for first waypoint.

    mycobot = MyCobot(PI_PORT, PI_BAUD)
    # Connect to the robot
    mycobot.power_on()
    mycobot.release_all_servos()

    mycobot.send_coords([10.63, -10.54, -13.0, 9.58, 65.91, 174.63], 50)

    time.sleep(2)

    q_init_deg = mycobot.get_angles()

    print(q_init_deg)

    solved_joint_angles = solve_waypoints(
        WAYPOINTS,
        angle_bounds=angle_bounds,
    )
    print_solved_waypoints(WAYPOINTS, solved_joint_angles)

    for i, joint_angles in enumerate(solved_joint_angles):
        mycobot.send_angles(joint_angles, 50)
        time.sleep(2)

    mycobot.power_off()
