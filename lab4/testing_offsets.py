import time
import sys
import numpy as np
from sympy import symbols, cos, sin, atan2, pi, Matrix, lambdify
from scipy.optimize import least_squares
from pymycobot import MechArm270
from pymycobot import PI_BAUD, PI_PORT


q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

offsets = [0, -pi/2, 0, 0, 0, 0]

dh_table = [ # added angles for group 5 robot
    [0.0,      0,           114,      q1 + offsets[0]],
    [0,       -pi/2,        0.0,      q2 + offsets[1]], 
    [105.0,    0.0,         0.0,      q3 + offsets[2]],
    [10,      -pi/2,        95,       q4 + offsets[3]],
    [0.0,      pi/2,        0,        q5 + offsets[4]]
    [0.0,     -pi/2,        58,       q6 + offsets[5]]
]

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

link_lengths = dh_table

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


def test(robot, desired):
    time.sleep(1.5)
    robot.send_angles(desired, 40)
    time.sleep(1.5)
    print(np.array(desired) - np.array(robot.get_angles()))

def test_fk(robot, angles):


if __name__ == "__main__":
    robot = MechArm270(PI_PORT, PI_BAUD)
    robot.power_on()
    time.sleep(0.5)

    # desired = [-22, 40, -13, 0, 30, 0]
    # test(robot, desired)

    # desired = [-22, 25, -13, 0, 35, 0]
    # test(robot, desired)
    
    # desired = [-22, 11, -13, 0, 38, 0]
    # test(robot, desired)

    # desired = [-22, -4, -13, 0, 40, 0]
    # test(robot, desired)

    # desired = [-22, 4, 23, 0, 20, 0]
    # test(robot, desired)

    # desired = [-22, 2, 0, 0, 35, 0]
    # test(robot, desired)

    # desired = [-22, 4, -24, 0, 55, 0]
    # test(robot, desired)



    robot.power_off()


