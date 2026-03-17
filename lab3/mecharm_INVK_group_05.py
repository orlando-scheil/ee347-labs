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
    [0,       -pi/2,        0.0,      q2 + offsets[1]],
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


def transf_to_pose(t_matrix):
    """Extract [X, Y, Z, roll, pitch, yaw] from a 4x4 homogeneous transformation matrix."""
    X = t_matrix[0, 3]
    Y = t_matrix[1, 3]
    Z = t_matrix[2, 3]

    R = np.array(t_matrix[0:3, 0:3]).astype(np.float64)

    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return X, Y, Z, roll, pitch, yaw


# ---------------------------------------------------------------------------
# CSV test (callable, no robot needed)
# ---------------------------------------------------------------------------

def run_csv_test(csv_path=None):
    """
    Validate basic (split position/orientation) IK against recorded poses
    from robot_poses.csv.

    For each row the CSV supplies a target pose [X,Y,Z,roll,pitch,yaw] and the
    measured joint angles.  The test runs IK on the target pose (seeded with the
    measured angles), then forward-kinematics on the IK result, and reports the
    pose error and joint-angle delta.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_poses.csv")

    data = np.loadtxt(csv_path, delimiter=",")
    n_poses = data.shape[0]

    print(f"Basic IK test — {n_poses} poses from {csv_path}")
    print("=" * 100)

    pos_errors = []
    ori_errors = []
    joint_deltas = []

    for idx, row in enumerate(data, start=1):
        target_xyz = row[0:3]
        target_rpy_deg = row[3:6]
        measured_deg = row[6:12]

        target_rpy_rad = np.deg2rad(target_rpy_deg)
        q_init = np.deg2rad(measured_deg)

        q_sol = inverse_kinematics(
            target_xyz[0], target_xyz[1], target_xyz[2],
            target_rpy_rad[0], target_rpy_rad[1], target_rpy_rad[2],
            q_init, link_lengths
        )

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
        print(f"  Target  (mm, deg) : X={target_xyz[0]:.2f}  Y={target_xyz[1]:.2f}  Z={target_xyz[2]:.2f}"
              f"  roll={target_rpy_deg[0]:.2f}  pitch={target_rpy_deg[1]:.2f}  yaw={target_rpy_deg[2]:.2f}")
        print(f"  FK→IK   (mm, deg) : X={fk_xyz[0]:.2f}  Y={fk_xyz[1]:.2f}  Z={fk_xyz[2]:.2f}"
              f"  roll={fk_rpy_deg[0]:.2f}  pitch={fk_rpy_deg[1]:.2f}  yaw={fk_rpy_deg[2]:.2f}")
        print(f"  Pos err (mm)      : ΔX={d_pos[0]:+.4f}  ΔY={d_pos[1]:+.4f}  ΔZ={d_pos[2]:+.4f}"
              f"  ‖err‖={pos_norm:.4f}")
        print(f"  Ori err (deg)     : Δr={d_ori[0]:+.4f}  Δp={d_ori[1]:+.4f}  Δy={d_ori[2]:+.4f}"
              f"  ‖err‖={ori_norm:.4f}")
        print(f"  Measured joints   : [{', '.join(f'{a:.2f}' for a in measured_deg)}]")
        print(f"  IK joints         : [{', '.join(f'{a:.2f}' for a in q_sol_deg)}]")
        print(f"  Δ joints (deg)    : [{', '.join(f'{a:+.2f}' for a in d_joints)}]")

    if not pos_errors:
        print("\nNo poses converged — nothing to summarise.")
        return

    pos_errors = np.array(pos_errors)
    ori_errors = np.array(ori_errors)
    joint_deltas = np.array(joint_deltas)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("-" * 100)
    print(f"  Position error  (mm) : mean={pos_errors.mean():.4f}  max={pos_errors.max():.4f}")
    print(f"  Orientation err (deg): mean={ori_errors.mean():.4f}  max={ori_errors.max():.4f}")
    print(f"  Mean |Δjoint| (deg)  : [{', '.join(f'{v:.4f}' for v in np.mean(np.abs(joint_deltas), axis=0))}]")
    print("=" * 100)


if __name__ == "__main__":
    run_csv_test()
