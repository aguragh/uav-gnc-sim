"""
Validation. Two things:

  1. Comprehensive check of the live-tuned gains (tuner_defaults.json):
     tracking error, as a percent of each path's characteristic scale, on
     preset paths under wind + a mid-flight gust, averaged over several RNG
     seeds -- plus a wind-speed envelope sweep.
  2. A sanity check against an analytically-derived gain set (standard
     second-order pole placement on the linearized double-integrator plant --
     see the README Results section) confirming the tuned gains aren't just
     arbitrary numbers with no principled basis.

Run directly to regenerate the numbers/plots the README cites.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dynamics
import wind
import controller
import paths

DT = 0.005
N_SEEDS = 6
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuner_defaults.json')) as f:
    TUNED = json.load(f)
TUNED_PSI = dict(kp=5.43e-4, kd=1.52e-4)  # yaw gains are hardcoded, not user-tunable

# path fn -> (characteristic scale [m], period [s] or None for a fixed point)
PRESET_PATHS = {
    'hover':        (paths.hover_path,        2.0, None),
    'circle':       (paths.circle_path,       2.5, 2 * np.pi / 0.4),
    'figure_eight': (paths.figure_eight_path, 2.5, 2 * np.pi / 0.4),
}


def analytic_gains():
    """Standard PD pole placement: e_ddot + Kd*e_dot + Kp*e = 0 for a chosen
    (zeta, wn) gives Kp = wn^2, Kd = 2*zeta*wn. Applies directly here because
    thrust-vector inversion + gravity feedforward makes each position axis a
    plain double integrator, and J*theta_ddot = tau makes the attitude loop
    the same math scaled by inertia. Attitude wn is placed 8x above position
    wn, standard cascade practice (inner loop must be much faster than the
    outer loop's own design assumes)."""
    zeta = 1 / np.sqrt(2)  # "maximally flat", textbook default, ~4.3% overshoot
    wn_pos = 4 / (zeta * 3.0)  # 4/(zeta*ts), target 3s settling time
    Kp_pos, Kd_pos = wn_pos ** 2, 2 * zeta * wn_pos
    Ki_pos = Kp_pos * (wn_pos / 10)  # corner freq well below wn, doesn't disturb poles

    wn_att = 8 * wn_pos
    Kp_att, Kd_att = dynamics.Ixx * wn_att ** 2, dynamics.Ixx * 2 * zeta * wn_att
    Kp_psi, Kd_psi = dynamics.Izz * wn_att ** 2, dynamics.Izz * 2 * zeta * wn_att

    gains = dict(Kp_xy=Kp_pos, Ki_xy=Ki_pos, Kd_xy=Kd_pos,
                 Kp_z=Kp_pos,  Ki_z=Ki_pos,  Kd_z=Kd_pos,
                 Kp_att=Kp_att, Kd_att=Kd_att)
    return gains, dict(kp=Kp_psi, kd=Kd_psi), dict(zeta=zeta, wn_pos=wn_pos, wn_att=wn_att)


def make_pids(g, psi_g):
    ip, ia = 5.0, 5e-4
    return [
        controller.PID(kp=g['Kp_xy'],  ki=g['Ki_xy'], kd=g['Kd_xy'],  i_max=ip),
        controller.PID(kp=g['Kp_xy'],  ki=g['Ki_xy'], kd=g['Kd_xy'],  i_max=ip),
        controller.PID(kp=g['Kp_z'],   ki=g['Ki_z'],  kd=g['Kd_z'],   i_max=ip),
        controller.PID(kp=g['Kp_att'], ki=0.0, kd=g['Kd_att'], i_max=ia),
        controller.PID(kp=g['Kp_att'], ki=0.0, kd=g['Kd_att'], i_max=ia),
        controller.PID(kp=psi_g['kp'], ki=0.0, kd=psi_g['kd'], i_max=ia),
    ]


def run(path_fn, gains, psi_g, seed, duration=25.0, V_mean=3.0, sigma_turb=1.0,
        gust_time=12.0, return_xy=False):
    """Fly one path under wind (+ an optional mid-flight gust). Returns RMS
    position error over the run, and optionally the raw (x, y) trajectory."""
    pids = make_pids(gains, psi_g)
    state = np.zeros(12)
    state[[0, 2, 4]] = path_fn(0.0)[:3]
    w = wind.WindModel(V_mean=V_mean, phi_az=np.radians(30), sigma_turb=sigma_turb,
                        rng=np.random.default_rng(seed))
    n = int(duration / DT)
    sq_err, gust_fired = 0.0, False
    xs, ys = (np.zeros(n), np.zeros(n)) if return_xy else (None, None)
    for k in range(n):
        t = k * DT
        sp = path_fn(t)
        if gust_time is not None and not gust_fired and t >= gust_time:
            w.fire_gust()
            gust_fired = True
        v_drone = state[[1, 3, 5]]
        v_wind = w.velocity(DT, altitude=max(state[4], 0.1), v_drone=v_drone)
        F_wind = wind.wind_force(v_wind, v_drone)
        T, tau = controller.control_step(state, pids, sp, DT)
        sq_err += (sp[0] - state[0]) ** 2 + (sp[1] - state[2]) ** 2 + (sp[2] - state[4]) ** 2
        if return_xy:
            xs[k], ys[k] = state[0], state[2]
        state = dynamics.rk4_step(state, T, tau, DT, F_wind=F_wind)
    rms = np.sqrt(sq_err / n)
    return (rms, xs, ys) if return_xy else rms


def cross_track_rms(xs, ys, path_fn, period, n_ref=4000, stride=5):
    """RMS distance from each flown point to the *nearest* point anywhere on
    the reference curve, instead of the point at the same time index. This
    separates genuinely being off the path (bad) from correctly tracing the
    same shape with a time lag (not a control problem, just a consequence of
    no setpoint-velocity feedforward)."""
    t_dense = np.linspace(0, period, n_ref)
    ref = np.array([path_fn(t)[:2] for t in t_dense])
    d = np.empty(len(range(0, len(xs), stride)))
    for i, k in enumerate(range(0, len(xs), stride)):
        d[i] = np.hypot(ref[:, 0] - xs[k], ref[:, 1] - ys[k]).min()
    return np.sqrt(np.mean(d ** 2))


def step_response(gains, psi_g, duration=8.0, final=2.0):
    pids = make_pids(gains, psi_g)
    state = np.zeros(12)
    state[4] = 2.0
    n = int(duration / DT)
    xs, ts = np.zeros(n), np.zeros(n)
    for k in range(n):
        T, tau = controller.control_step(state, pids, np.array([final, 0.0, 2.0, 0.0]), DT)
        state = dynamics.rk4_step(state, T, tau, DT, F_wind=np.zeros(3))
        xs[k], ts[k] = state[0], k * DT
    return ts, xs


def step_metrics(ts, xs, final=2.0, band=0.05):
    overshoot = max(0.0, (xs.max() - final) / final * 100)
    tol = band * final
    settled = next((i + 1 for i in range(len(xs) - 1, -1, -1) if abs(xs[i] - final) > tol), 0)
    settle_t = ts[settled] if settled < len(ts) else 0.0
    rs, re = np.argmax(xs >= 0.1 * final), np.argmax(xs >= 0.9 * final)
    rise_t = ts[re] - ts[rs] if re > rs else float('nan')
    return overshoot, settle_t, rise_t


if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    ANALYTIC, ANALYTIC_PSI, params = analytic_gains()

    print('=== Analytically-derived gains (pole placement) ===')
    print(f"zeta={params['zeta']:.4f}  wn_pos={params['wn_pos']:.4f} rad/s  "
          f"wn_att={params['wn_att']:.4f} rad/s (8x position wn)")
    print(f"Kp_xy=Kp_z={ANALYTIC['Kp_xy']:.4f}  Kd_xy=Kd_z={ANALYTIC['Kd_xy']:.4f}")
    print(f"Kp_att={ANALYTIC['Kp_att']:.6f}  (live-tuned: {TUNED['Kp_att']})")

    print()
    print('=== Preset-path tracking: cross-track RMS error as % of path scale '
          f'(wind + gust, {N_SEEDS} seeds) ===')
    print('    cross-track = actual vs. nearest point anywhere on the reference curve')
    print('    (adherence to the path shape itself, independent of timing/lag)')
    for name, (path_fn, scale, period) in PRESET_PATHS.items():
        for label, (g, psi_g) in [('tuned', (TUNED, TUNED_PSI)), ('analytic', (ANALYTIC, ANALYTIC_PSI))]:
            pct = []
            for s in range(N_SEEDS):
                if period is not None:
                    _, xs, ys = run(path_fn, g, psi_g, seed=s, return_xy=True)
                    pct.append(100 * cross_track_rms(xs, ys, path_fn, period) / scale)
                else:
                    pct.append(100 * run(path_fn, g, psi_g, seed=s) / scale)
            print(f'{name:14s} {label:10s} {np.mean(pct):5.2f}% +/- {np.std(pct):.2f}%')

    print()
    print('=== Step response: tuned vs analytical ===')
    ts_tu, xs_tu = step_response(TUNED, TUNED_PSI)
    ts_an, xs_an = step_response(ANALYTIC, ANALYTIC_PSI)
    os_tu, set_tu, rise_tu = step_metrics(ts_tu, xs_tu)
    os_an, set_an, rise_an = step_metrics(ts_an, xs_an)
    print(f'tuned:      overshoot={os_tu:.1f}%  settling={set_tu:.2f}s  rise={rise_tu:.2f}s')
    print(f'analytical: overshoot={os_an:.1f}%  settling={set_an:.2f}s  rise={rise_an:.2f}s')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts_tu, xs_tu, color='#2266aa', lw=1.5, label='live-tuned')
    ax.plot(ts_an, xs_an, color='#3a9d5c', lw=1.5, label='analytical (pole placement)')
    ax.axhline(2.0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.fill_between(ts_tu, 1.9, 2.1, color='gray', alpha=0.12, label='+/-5% band')
    ax.set(xlabel='t [s]', ylabel='x [m]', title='Position step response: tuned vs analytically-derived gains')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'step_tuned_vs_analytic.png'), dpi=150)
    plt.close(fig)

    print()
    print('=== Wind-speed envelope: hover RMS error vs wind speed, tuned vs analytical ===')
    V_RANGE = np.arange(0, 8.5, 1.0)
    results = {}
    for label, (g, psi_g) in [('tuned', (TUNED, TUNED_PSI)), ('analytical', (ANALYTIC, ANALYTIC_PSI))]:
        rms_m, rms_s = [], []
        for V in V_RANGE:
            rms_runs = [run(paths.hover_path, g, psi_g, seed=s, V_mean=V, sigma_turb=1.5, gust_time=None)
                        for s in range(N_SEEDS)]
            rms_m.append(np.mean(rms_runs))
            rms_s.append(np.std(rms_runs))
        results[label] = (np.array(rms_m), np.array(rms_s))
        print(f'{label:12s} RMS @ V=0: {rms_m[0]:.4f}   RMS @ V=8: {rms_m[-1]:.4f}')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, color in [('tuned', '#2266aa'), ('analytical', '#3a9d5c')]:
        rms_m, rms_s = results[label]
        ax.errorbar(V_RANGE, rms_m, yerr=rms_s, fmt='o-', color=color, capsize=3, label=f'{label} RMS')
    ax.set(xlabel='mean wind speed [m/s]', ylabel='RMS position error [m]',
           title='Hover RMS error vs wind speed: tuned vs analytical gains')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'envelope_tuned_vs_analytic.png'), dpi=150)
    plt.close(fig)

    print()
    print(f'plots written to {PLOTS_DIR}/')
