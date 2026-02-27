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

# Orientation residuals (radians) are ~100x smaller than position residuals (mm),
# so w_ori in target_pose_error scales them up for balanced optimisation.

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

def wrap_to_pi(angles):
    """Wrap angles into the [-pi, pi] range to handle ±180° discontinuities."""
    return (angles + np.pi) % (2 * np.pi) - np.pi

def target_pose_error(joint_angles, *args):
    target_pose = args[0]
    w_ori = args[1] if len(args) > 1 else 50.0

    current_fk = forward_kinematics_func(*joint_angles)
    current_pose = np.array(transf_to_pose(current_fk), dtype=np.float64)

    error = current_pose - target_pose
    error[3:6] = wrap_to_pi(error[3:6])
    error[3:6] *= w_ori
    return error

# Finds a solution to inverse kinematics for a target pose using nonlinear least-squares
def ik(target_pose, init_joints, max_iter=1000, tolerance=1e-5, bounds=None, w_ori=50.0):
    """
    Solves for joint angles that achieve a given end-effector pose.

    Args:
        target_pose: desired pose vector [X, Y, Z, roll, pitch, yaw]
        init_joints: initial guess for joint angles (length 6)
        max_iter: maximum number of optimization iterations
        tolerance: convergence tolerance
        bounds: optional tuple for joint angle limits (lower, upper); defaults to no bounds
        w_ori: weight applied to orientation residuals so they contribute
               comparably to position residuals (mm vs rad scaling)

    Returns:
        joint_angles: array of 6 joint values if solution found, else None
    """
    if bounds is None:
        bounds = (-np.pi, np.pi)
    result = least_squares(
        target_pose_error,
        init_joints,
        args=(np.array(target_pose), w_ori),
        method='trf',
        max_nfev=max_iter,
        ftol=tolerance,
        bounds=bounds
    )
    if result.success:
        return result.x
    else:
        print("Inverse kinematics did not converge.")
        return None


# ---------------------------------------------------------------------------
# CSV test (callable, no robot needed)
# ---------------------------------------------------------------------------

def run_csv_test(csv_path=None):
    """
    Validate advanced IK against recorded poses from robot_poses.csv.

    For each row the CSV supplies a target pose [X,Y,Z,roll,pitch,yaw] and the
    measured joint angles.  The test runs IK on the target pose (seeded with the
    measured angles), then forward-kinematics on the IK result, and reports the
    pose error and joint-angle delta.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_poses.csv")

    data = np.loadtxt(csv_path, delimiter=",")
    n_poses = data.shape[0]

    print(f"Advanced IK test — {n_poses} poses from {csv_path}")
    print("=" * 100)

    pos_errors = []
    ori_errors = []
    joint_deltas = []

    for idx, row in enumerate(data, start=1):
        target_xyz = row[0:3]
        target_rpy_deg = row[3:6]
        measured_deg = row[6:12]

        target_rpy_rad = np.deg2rad(target_rpy_deg)
        target_pose = np.concatenate([target_xyz, target_rpy_rad])

        q_init = np.deg2rad(measured_deg)
        q_sol = ik(target_pose=target_pose, init_joints=q_init)

        if q_sol is None:
            print(f"\nPose {idx:>2}: IK did not converge — skipped\n")
            continue

        q_sol_deg = np.degrees(q_sol)
        fk_pose = np.array(transf_to_pose(forward_kinematics_func(*q_sol)), dtype=np.float64)
        fk_xyz = fk_pose[0:3]
        fk_rpy_deg = np.degrees(fk_pose[3:6])

        d_pos = fk_xyz - target_xyz
        d_ori = (fk_rpy_deg - target_rpy_deg + 180) % 360 - 180
        d_joints = q_sol_deg - measured_deg

        pos_norm = np.linalg.norm(d_pos)
        ori_norm = np.linalg.norm(d_ori)

        pos_errors.append(pos_norm)
        ori_errors.append(ori_norm)
        joint_deltas.append(d_joints)

        print(f"\nPose {idx:>2}")
        print(f"  Target  (mm, deg) : X={target_xyz[0]:.2f}\tY={target_xyz[1]:.2f}\tZ={target_xyz[2]:.2f}"
              f"\troll={target_rpy_deg[0]:.2f}\tpitch={target_rpy_deg[1]:.2f}\tyaw={target_rpy_deg[2]:.2f}")
        print(f"  FK→IK   (mm, deg) : X={fk_xyz[0]:.2f}\tY={fk_xyz[1]:.2f}\tZ={fk_xyz[2]:.2f}"
              f"\troll={fk_rpy_deg[0]:.2f}\tpitch={fk_rpy_deg[1]:.2f}\tyaw={fk_rpy_deg[2]:.2f}")
        print(f"  Pos err (mm)      : ΔX={d_pos[0]:+.4f}\tΔY={d_pos[1]:+.4f}\tΔZ={d_pos[2]:+.4f}"
              f"\t‖err‖={pos_norm:.4f}")
        print(f"  Ori err (deg)     : Δr={d_ori[0]:+.4f}\tΔp={d_ori[1]:+.4f}\tΔy={d_ori[2]:+.4f}"
              f"\t‖err‖={ori_norm:.4f}")
        print(f"      Measured joints   : [\t{','.join(f'{a:.2f}' for a in measured_deg)}]")
        print(f"  IK joints         : [\t{','.join(f'{a:.2f}' for a in q_sol_deg)}]")
        print(f"  Δ joints (deg)    : [\t{','.join(f'{a:+.2f}' for a in d_joints)}]")

    if not pos_errors:
        print("\nNo poses converged — nothing to summarise.")
        return

    pos_errors = np.array(pos_errors)
    ori_errors = np.array(ori_errors)
    joint_deltas = np.array(joint_deltas)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("-" * 100)
    print(f"  Position error  (mm) : mean={pos_errors.mean():.4f}\tmax={pos_errors.max():.4f}")
    print(f"  Orientation err (deg): mean={ori_errors.mean():.4f}\tmax={ori_errors.max():.4f}")
    print(f"  Mean |Δjoint| (deg)  : [\t{','.join(f'{v:.4f}' for v in np.mean(np.abs(joint_deltas), axis=0))}]")
    print("=" * 100)


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
    # run_csv_test()
    WAYPOINTS = [[177.6, -7.6, 99.9, -175.45, -4.66, -19.76], [149.9, 74.4, 183.1, -171.73, -25.7, -3.66], [185.8, 80.7, 97.6, -173.61, -20.95, -9.22], [188.8, 82.2, 67.2, -174.73, -21.32, -8.01], [111.5, 38.5, 278.7, -155.98, -68.94, -14.42], [134.5, -15.6, 225.0, 179.15, -32.26, -17.72], [172.9, -32.5, 174.4, 167.77, -22.22, -10.83], [170.8, -102.1, 126.8, -177.33, -22.67, -9.2], [176.9, -96.7, 108.4, -176.02, -19.62, -8.53], [121.4, -8.4, 276.2, -179.5, -72.92, 1.9], [165.3, -8.9, 68.2, 179.46, -23.2, 1.44]]

    # best values so far: [[178.6, -3.1, 105.5, 177.04, 4.02, -8.23], [164.1, 7.1, 175.5, -178.78, 20.35, 176.47], [199.9, 71.9, 70.3, -172.04, -15.13, -3.0], [176.5, 11.0, 162.4, -171.24, -33.34, 0.9], [153.4, 3.8, 177.9, -177.93, -24.24, 4.41], [95.6, -43.8, 243.5, 165.76, -26.45, -11.02], [186.0, -78.9, 114.4, 174.85, -14.32, -2.69], [120.4, -10.0, 235.0, 176.21, -27.81, 1.92], [180.3, -10.1, 72.6, 179.21, -22.98, 1.49]]

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

    time.sleep(1.5)
    mycobot.set_gripper_value(100, 50)
    time.sleep(1.5)
    mycobot.send_angles(solved_joint_angles[0], 50)
    time.sleep(1.5)
    mycobot.send_angles(solved_joint_angles[1], 50)
    time.sleep(1.5)
    mycobot.send_angles(solved_joint_angles[2], 50)
    time.sleep(1.5)
    mycobot.send_angles(solved_joint_angles[3], 50)
    time.sleep(1.5)
    mycobot.set_gripper_value(0, 50)
    time.sleep(1.5)
    mycobot.send_angles(solved_joint_angles[4], 50)
    time.sleep(1.5)
    mycobot.send_angles(solved_joint_angles[5], 50)
    time.sleep(1.5)
    mycobot.send_angles(solved_joint_angles[6], 50)
    time.sleep(1.5)
    mycobot.send_angles(solved_joint_angles[7], 50)
    time.sleep(1.5)
    mycobot.set_gripper_value(100, 50)
    time.sleep(1.5)
    mycobot.send_angles(solved_joint_angles[9], 50)
    time.sleep(1.5)
    mycobot.send_angles(solved_joint_angles[10], 50)
    time.sleep(1.5)



    mycobot.power_off()
