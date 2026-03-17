import time
import sys
import numpy as np
from sympy import symbols, cos, sin, atan2, pi, Matrix, lambdify
from scipy.optimize import least_squares
from pymycobot import MechArm270
from pymycobot import PI_BAUD, PI_PORT


# ---------------------------------------------------------------------------
# Symbolic setup (DH parameters, forward kinematics)
# ---------------------------------------------------------------------------

q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

offsets = [0, -pi/2 + np.deg2rad(3), 0, 0, 0, 0]
# offsets = [0, -pi/2 + np.deg2rad(4), np.deg2rad(2), 0, np.deg2rad(4), 0]


dh_table = [ # added angles for group 5 robot
    [0.0,      0,           114,      q1 + offsets[0]],
    [0,       -pi/2,        0.0,      q2 + offsets[1]],  #  + np.deg2rad(3)
    [105.0,    0.0,         0.0,      q3 + offsets[2]], #  + np.deg2rad(2)
    [10,      -pi/2,        95,       q4 + offsets[3]],
    [0.0,      pi/2,        0,        q5 + offsets[4]], #  + np.deg2rad(4)
    [0.0,     -pi/2,        58,       q6 + offsets[5]]
]

link_lengths = dh_table


def get_transformation_matrix(a, alpha, d, theta):
    """Creates a 4x4 homogeneous transformation matrix from D-H parameters."""
    M = Matrix([
        [cos(theta),            -sin(theta),             0,           a],
        [sin(theta)*cos(alpha),  cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
        [sin(theta)*sin(alpha),  cos(theta)*sin(alpha),  cos(alpha),  cos(alpha)*d],
        [0,                      0,                       0,                     1]
    ])
    return M


print("Calculating transformation matrices")
T_total = Matrix.eye(4)
for row in dh_table:
    a, alpha, d, theta = row
    T_i = get_transformation_matrix(a, alpha, d, theta)
    T_total = T_total * T_i

q_sym = symbols('q1:7')


def symbolic_forward_kinematics(q_values):
    """
    Symbolic forward kinematics using DH-parameter transformation matrix
    with substituted joint values.
    """
    q_syms = [q1, q2, q3, q4, q5, q6]
    subs_dict = {k: v for k, v in zip(q_syms, q_values)}
    T_symbolic = T_total.subs(subs_dict)
    return T_symbolic


forward_kinematics_func = lambdify(q_sym, symbolic_forward_kinematics(q_sym), 'numpy')


# ---------------------------------------------------------------------------
# Joint angle constraints
# ---------------------------------------------------------------------------

min_angle = [-160, -90, -180, -145, -115, -175] # -165, -90, -180, -145, -115, -175
max_angle = [160, 90, 65, 145, 115, 175] # 165, 90, 65, 145, 115, 175
angle_bounds = (np.deg2rad(min_angle), np.deg2rad(max_angle))


def within_bounds(position, min_angles=None, max_angles=None):
    """Return the position if all joint angles are within bounds, else None."""
    if min_angles is None:
        min_angles = min_angle
    if max_angles is None:
        max_angles = max_angle
    angles = position[-6:] if len(position) >= 6 else position
    if len(angles) != 6:
        return None
    for i in range(6):
        if angles[i] < min_angles[i] or angles[i] > max_angles[i]:
            return None
    return position


# ---------------------------------------------------------------------------
# Pose / angle utilities
# ---------------------------------------------------------------------------

def transf_to_pose(t_matrix):
    """Convert a 4x4 homogeneous transformation matrix to [X, Y, Z, roll, pitch, yaw]."""
    X = t_matrix[0, 3]
    Y = t_matrix[1, 3]
    Z = t_matrix[2, 3]

    R = np.array(t_matrix[0:3, 0:3]).astype(np.float64)

    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return X, Y, Z, roll, pitch, yaw


def wrap_to_pi(angles):
    """Wrap angles into the [-pi, pi] range to handle +/-180 deg discontinuities."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Inverse kinematics (advanced, full-pose least-squares)
# ---------------------------------------------------------------------------

def target_pose_error(joint_angles, *args):
    """Residual vector between current FK pose and a target pose."""
    target_pose = args[0]
    w_ori = args[1] if len(args) > 1 else 50.0

    current_fk = forward_kinematics_func(*joint_angles)
    current_pose = np.array(transf_to_pose(current_fk), dtype=np.float64)

    error = current_pose - target_pose
    error[3:6] = wrap_to_pi(error[3:6])
    error[3:6] *= w_ori
    return error


def ik(target_pose, init_joints, max_iter=1000, tolerance=1e-5, bounds=None, w_ori=50.0):
    """
    Solve for joint angles that achieve a given end-effector pose using
    nonlinear least-squares.

    Args:
        target_pose: desired pose vector [X, Y, Z, roll, pitch, yaw]
        init_joints: initial guess for joint angles (length 6)
        max_iter: maximum number of optimization iterations
        tolerance: convergence tolerance
        bounds: optional (lower, upper) for joint angle limits; defaults to (-pi, pi)
        w_ori: weight applied to orientation residuals

    Returns:
        joint_angles array (length 6) if solution found, else None
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


def solve_waypoints(waypoints, q_init_deg=None, angle_bounds=None):
    """
    Run IK on a list of waypoints and return the solved joint angles.

    Args:
        waypoints: list of [x, y, z, rx_deg, ry_deg, rz_deg] (mm and degrees).
        q_init_deg: optional initial guess in degrees (length-6 list).
                    Each solution seeds the next waypoint.
        angle_bounds: optional (lower, upper) tuple for joint limits in radians.

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

# ---------------------------------------------------------------------------
# Waypoint execution with gripper control
# ---------------------------------------------------------------------------

def execute_waypoints(robot, solutions, open_after=None, close_after=None, speed=50):
    """
    Send each solved waypoint to the robot, opening/closing the gripper
    at specified indices.

    Args:
        robot: MyCobot instance.
        solutions: list of joint-angle lists from solve_waypoints().
        open_after: list of waypoint indices after which to open the gripper.
        close_after: list of waypoint indices after which to close the gripper.
        speed: movement speed (0-100).
    """
    if open_after is None:
        open_after = []
    if close_after is None:
        close_after = []

    for i, sol in enumerate(solutions):
        if sol is None:
            print(f"Skipping waypoint {i} (IK failed)")
            continue

        formatted_sol = [f"{value:.3f}" for value in sol]
        print("Angle ", i, ": ", formatted_sol)
        robot.send_angles(sol, speed) # Set to send_coords to bypass ik
        time.sleep(2.0)

        if i in open_after:
            robot.set_gripper_value(100, 30, 1)
            time.sleep(2.0)
        if i in close_after:
            robot.set_gripper_value(82, 30, 1)
            time.sleep(2.0)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        dec_top = [121.7, 59.0, 200.0, 171.33, 53.24, -163.56]
        rec_bot = [142.0, -58.0, 52.7, -177.54, 56.81, 158.29]

        # increasing x moves forward
        # increasing y moves right

        # blocks are 38cm big 

        WAYPOINTS = [
            [121.7, 59.0, 250.0, 171.33, 53.24, -163.56], # Dec Top Above
            [121.7, 59.0, 200.0, 171.33, 53.24, -163.56], # Dec Top 1
            [121.7, 59.0, 250.0, 171.33, 53.24, -163.56], # Dec Top Above
            [142.0, -58.0, 108.7, -177.54, 56.81, 158.29], # Con Bottom Above
            [142.0, -58.0, 52.7, -177.54, 56.81, 158.29], # Con Bottom 4
            [142.0, -58.0, 108.7, -177.54, 56.81, 158.29], # Con Bottom Above
            
            [121.7, 59.0, 180.0, 171.33, 53.24, -163.56], # Dec Middle Above
            [121.7, 59.0, 135.0, 171.33, 53.24, -163.56], # Dec Middle 7
            [121.7, 59.0, 180.0, 171.33, 53.24, -163.56], # Dec Middle Above
            [131.8, -53.6, 174.0, -177.54, 56.81, 158.29], # Con Middle Above
            [131.8, -53.6, 134.0, -177.54, 56.81, 158.29], # Con Middle 10
            [131.8, -53.6, 174.0, -177.54, 56.81, 158.29], # Con Middle Above

            [139, 69.0, 108.7, 171.33, 53.24, -163.56], # Dec Bottom Above
            [139, 69.0, 68.7, 171.33, 53.24, -163.56], # Dec Bottom 13
            [139, 69.0, 108.7, 171.33, 53.24, -163.56], # Dec Bottom Above
            [135.8, -52, 260.0, -177.54, 56.81, 158.29], # Con Middle Above
            [135.8, -52, 220.0, -177.54, 56.81, 158.29], # Con Middle 16
            [135.8, -52, 250.0, -177.54, 56.81, 158.29], # Con Middle Above
        ]

        robot = MechArm270(PI_PORT, PI_BAUD)
        robot.power_on()
        time.sleep(2.0)

        robot.send_angles([0,0,0,0,0,180], 30) # Set last variable to 0 if bypassing ik
        time.sleep(2.0)
        angles = robot.get_angles()
        coords = robot.get_coords()
        print("Coords: ", coords, ", angles: ", angles)
        
        robot.set_gripper_value(100, 50, 1)

        solutions = solve_waypoints(WAYPOINTS, angle_bounds=angle_bounds)
        open_after  = [0, 4, 10, 16]
        close_after = [1, 7, 13]

        # Set solutions parameter to WAYPOINTS to bypass IK
        execute_waypoints(robot, solutions, open_after=open_after, close_after=close_after, speed=30)

        robot.send_coords([0, 0, 0, 0, 0, 0], 50)

        robot.power_off()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        if robot:
            time.sleep(0.5)
            robot.power_off()
            print("Robot shut down.")





