import numpy as np

# Each function returns a setpoint [x_des, y_des, z_des, psi_des] at time t.
# Defaults are shared with web/index.html's client-side reference-path draw --
# change one, change the other, or the drawn tube and the actual setpoint
# drift apart.


def hover_path(t, x0=0.0, y0=0.0, z0=2.0):
    return np.array([x0, y0, z0, 0.0])


def circle_path(t, R=2.5, omega=0.4, z0=2.0):
    return np.array([R * np.cos(omega * t), R * np.sin(omega * t), z0, 0.0])


def figure_eight_path(t, R=2.5, omega=0.4, z0=2.0):
    th = omega * t
    return np.array([R * np.sin(th), R * np.sin(2 * th), z0, 0.0])


def rose_path(t, R=2.0, k=3, omega=0.15, z0=2.0, z_amp=0.8):
    '''3-petal rhodonea curve with per-petal height variation.'''
    th = omega * t
    r = R * np.cos(k * th)
    return np.array([r * np.cos(th), r * np.sin(th), z0 + z_amp * np.sin(k * th), 0.0])


def lissajous_path(t, Rx=2.5, Ry=2.5, Rz=1.2, omega=0.3, z0=2.0):
    '''3D Lissajous figure, frequency ratios 1:2:3.'''
    return np.array([
        Rx * np.sin(omega * t),
        Ry * np.sin(2 * omega * t + np.pi / 4),
        z0 + Rz * np.sin(3 * omega * t),
        0.0,
    ])


def trefoil_path(t, R=0.75, z_amp=0.7, omega=0.3, z0=2.0):
    th = omega * t
    return np.array([
        R * (np.sin(th) + 2 * np.sin(2 * th)),
        R * (np.cos(th) - 2 * np.cos(2 * th)),
        z0 + z_amp * np.sin(3 * th),
        0.0,
    ])


def spiral_path(t, R=1.5, omega=0.4, z0=1.0, climb_rate=0.1):
    z_range = 2.0
    z_ph = (t * climb_rate) % (2 * z_range)
    z = z0 + (z_ph if z_ph < z_range else 2 * z_range - z_ph)
    return np.array([R * np.cos(omega * t), R * np.sin(omega * t), z, 0.0])
