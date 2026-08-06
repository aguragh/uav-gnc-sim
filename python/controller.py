import numpy as np
from dynamics import m, g, T_max, Tau_max, Theta_max


I_MAX_POS = 5.0
I_MAX_ATT = 5e-4

class PID:
    def __init__(self, kp, ki, kd, i_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_max = i_max
        self.integral = 0.0
        self.prev_measurement = None
        self._trial_integral = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_measurement = None
        self._trial_integral = 0.0

    def update(self, error, measurement, dt):
        self._trial_integral = np.clip(self.integral + error * dt, -self.i_max, self.i_max)
        if self.prev_measurement is None or dt <= 0.0:
            derivative = 0.0
        else:
            derivative = -(measurement - self.prev_measurement) / dt
        self.prev_measurement = measurement
        return self.kp * error + self.ki * self._trial_integral + self.kd * derivative

    def commit(self, accept):
        if accept:
            self.integral = self._trial_integral


def _saturated(val, lo, hi, err):
    return (val >= hi and err > 0) or (val <= lo and err < 0)


def make_pids():
    return [
        PID(kp=2.0,     ki=0.8, kd=2.0,     i_max=I_MAX_POS),  # x
        PID(kp=2.0,     ki=0.8, kd=2.0,     i_max=I_MAX_POS),  # y
        PID(kp=4.0,     ki=1.6, kd=4.0,     i_max=I_MAX_POS),  # z
        PID(kp=3.15e-3, ki=0.0, kd=2.94e-4, i_max=I_MAX_ATT),  # roll
        PID(kp=3.15e-3, ki=0.0, kd=2.94e-4, i_max=I_MAX_ATT),  # pitch
        PID(kp=5.43e-4, ki=0.0, kd=1.52e-4, i_max=I_MAX_ATT),  # yaw
    ]


def _attitude_from_accel(ax, ay, az, psi_des):
    a_des = np.array([ax, ay, az + g])
    mag = np.linalg.norm(a_des)
    if mag < 1e-6:
        return 0.0, 0.0
    z_des = a_des / mag

    y_c = np.array([-np.sin(psi_des), np.cos(psi_des), 0.0])
    x_des = np.cross(y_c, z_des)
    xn = np.linalg.norm(x_des)
    if xn < 1e-6:
        return 0.0, 0.0
    x_des /= xn

    R_des = np.column_stack([x_des, np.cross(z_des, x_des), z_des])
    phi_des = np.arctan2(R_des[2, 1], R_des[2, 2])
    theta_des = np.arcsin(np.clip(-R_des[2, 0], -1.0, 1.0))
    return phi_des, theta_des


def control_step(state, pids, setpoint, dt):
    x, vx, y, vy, z, vz      = state[:6]
    phi, p, theta, q, psi, r = state[6:]
    pid_x, pid_y, pid_z, pid_phi, pid_theta, pid_psi = pids
    x_des, y_des, z_des, psi_des = setpoint

    ex, ey, ez = x_des - x, y_des - y, z_des - z

    ax = pid_x.update(ex, x, dt)
    ay = pid_y.update(ey, y, dt)
    az = pid_z.update(ez, z, dt)

    T_des = m * np.sqrt(ax**2 + ay**2 + (az + g)**2)
    phi_des, theta_des = _attitude_from_accel(ax, ay, az, psi_des)

    T         = np.clip(T_des,   0.0,        T_max)
    phi_tgt   = np.clip(phi_des, -Theta_max, Theta_max)
    theta_tgt = np.clip(theta_des, -Theta_max, Theta_max)

    z_sat  = _saturated(T_des, 0.0, T_max, ez)
    xy_sat = (_saturated(phi_des,   -Theta_max, Theta_max, ey) or
              _saturated(theta_des, -Theta_max, Theta_max, ex))

    pid_z.commit(not z_sat)
    pid_x.commit(not xy_sat)
    pid_y.commit(not xy_sat)

    e_phi   = phi_tgt - phi
    e_theta = theta_tgt - theta
    e_psi   = (psi_des - psi + np.pi) % (2 * np.pi) - np.pi

    tau_phi   = np.clip(pid_phi.update(e_phi, phi, dt),     -Tau_max[0], Tau_max[0])
    tau_theta = np.clip(pid_theta.update(e_theta, theta, dt), -Tau_max[1], Tau_max[1])
    tau_psi   = np.clip(pid_psi.update(e_psi, psi, dt),     -Tau_max[2], Tau_max[2])

    pid_phi.commit(  not _saturated(tau_phi,   -Tau_max[0], Tau_max[0], e_phi))
    pid_theta.commit(not _saturated(tau_theta, -Tau_max[1], Tau_max[1], e_theta))
    pid_psi.commit(  not _saturated(tau_psi,   -Tau_max[2], Tau_max[2], e_psi))

    return T, np.array([tau_phi, tau_theta, tau_psi])
