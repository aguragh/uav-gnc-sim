import numpy as np

# Constants for Crazyflie 2.1 (drone simulated)

m = 0.030
g = 9.81

Ixx = 1.4e-5 # roll and pitch have same I by symmetry
Iyy = 1.4e-5
Izz = 2.17e-5 # Yaw axis

T_max = 0.60 # total thrust for 4 actuators at max thrust 150 mN
Tau_max = np.array([0.015, 0.015, 0.005]) # max torques for roll, pitch, yaw

Theta_max = np.radians(30) 


def rotation_matrices(phi, theta, psi):
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)  
    cy, sy = np.cos(psi), np.sin(psi)

    R_x = np.array([[1, 0,   0  ],
                    [0, cp, -sp ],
                    [0, sp,  cp ]])
    R_y = np.array([[ ct, 0, st],
                    [  0, 1,  0],
                    [-st, 0, ct]])
    R_z = np.array([[cy, -sy, 0],
                    [sy,  cy, 0],
                    [ 0,   0, 1]])

    R = R_z @ R_y @ R_x  

    phi_dot   = np.array([1, 0, 0])
    theta_dot = R_x.T @ R_y.T @ np.array([0, 1, 0])           # e_y
    psi_dot   = R_x.T @ R_y.T @ R_z.T @ np.array([0, 0, 1])   # e_z

    B = np.column_stack([phi_dot, theta_dot, psi_dot])
    W = np.linalg.inv(B)

    return R, W


def eom(state, T, tau, F_wind=None):

    vx = state[1]; vy = state[3]; vz = state[5]
    phi = state[6]; p = state[7]
    theta = state[8]; q = state[9]
    psi = state[10]; r = state[11]

    if F_wind is None:
        F_wind = np.zeros(3)

    R, W = rotation_matrices(phi, theta, psi)

    accel = (T * R[:, 2] + F_wind) / m - np.array([0, 0, g])

    J = np.array([Ixx, Iyy, Izz])
    w = np.array([p, q, r])
    wdot = (tau - np.cross(w, J * w)) / J

    euler_dot = W @ w

    return np.array([
        vx, accel[0],
        vy, accel[1],
        vz, accel[2],
        euler_dot[0], wdot[0],
        euler_dot[1], wdot[1],
        euler_dot[2], wdot[2]
    ])

def rk4_step(state, T, tau, dt, F_wind=None):
    f = lambda s: eom(s, T, tau, F_wind)
    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
