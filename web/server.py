"""
UAV wind-sim WebSocket server.

Run:  uvicorn server:app --reload
Then: open http://localhost:8000
"""

import sys, os, asyncio, json
_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_WEB_DIR)
sys.path.insert(0, os.path.join(_ROOT, 'python'))

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from paths import (circle_path, figure_eight_path, spiral_path,
                   rose_path, lissajous_path, trefoil_path)
from wind import WindModel, wind_force
from controller import control_step, PID
from dynamics import rk4_step

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

DT          = 0.005   # physics timestep  [s]
STEPS_FRAME = 3       # physics steps per send → 200/3 ≈ 67 Hz to browser
FRAME_SLEEP = 1 / 60  # ~60 fps target

_PATHS = {
    'Hover':        lambda t: np.array([0.0, 0.0, 2.0, 0.0]),
    'Circle':       circle_path,
    'Figure-eight': figure_eight_path,
    'Spiral':       spiral_path,
    'Rose':         rose_path,
    'Lissajous':    lissajous_path,
    'Trefoil':      trefoil_path,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_gains():
    """Load saved gains from tuner, falling back to conservative defaults."""
    defaults = dict(Kp_xy=2.7, Ki_xy=2.15, Kd_xy=2.0,
                    Kp_z=4.0,  Ki_z=1.6,   Kd_z=4.0,
                    Kp_att=3.15e-3, Kd_att=2.94e-4)
    try:
        with open(os.path.join(_ROOT, 'python', 'tuner_defaults.json')) as f:
            return {**defaults, **json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError):
        return defaults


def _make_pids(g):
    ip, ia = 5.0, 5e-4
    return [
        PID(kp=g['Kp_xy'],  ki=g['Ki_xy'],  kd=g['Kd_xy'],  i_max=ip),  # x
        PID(kp=g['Kp_xy'],  ki=g['Ki_xy'],  kd=g['Kd_xy'],  i_max=ip),  # y
        PID(kp=g['Kp_z'],   ki=g['Ki_z'],   kd=g['Kd_z'],   i_max=ip),  # z
        PID(kp=g['Kp_att'], ki=0.0, kd=g['Kd_att'], i_max=ia),           # roll
        PID(kp=g['Kp_att'], ki=0.0, kd=g['Kd_att'], i_max=ia),           # pitch
        PID(kp=5.43e-4,     ki=0.0, kd=1.52e-4,     i_max=ia),           # yaw
    ]


def _make_wind(V_mean=3.0, wind_dir=45.0, sigma_w=1.5):
    return WindModel(V_mean=V_mean, phi_az=np.radians(wind_dir),
                     sigma_turb=sigma_w, rng=np.random.default_rng())


def _init_state(path_fn):
    sp0   = path_fn(0.0)
    state = np.zeros(12)
    state[[0, 2, 4]] = sp0[:3]
    return state


# ── WebSocket sim loop ────────────────────────────────────────────────────────

@app.get('/gains')
async def get_gains():
    return _load_gains()


@app.websocket('/ws')
async def sim_ws(ws: WebSocket):
    await ws.accept()

    gains    = _load_gains()
    path_key = 'Hover'
    path_fn  = _PATHS[path_key]
    pids     = _make_pids(gains)
    wind     = _make_wind()
    state    = _init_state(path_fn)
    t        = 0.0
    T        = 0.0
    sp       = path_fn(0.0)
    paused   = True

    # Non-blocking receive: keep a persistent task, check it each frame
    recv_task = asyncio.create_task(ws.receive_json())

    try:
        while True:
            # ── Drain incoming commands (non-blocking) ────────────────────────
            done, _ = await asyncio.wait({recv_task}, timeout=0.0)
            if recv_task in done:
                try:
                    msg = recv_task.result()
                    cmd = msg.get('cmd', '')

                    if cmd == 'gust':
                        wind.fire_gust()

                    elif cmd == 'set_path':
                        path_key = msg.get('path', 'Circle')
                        path_fn  = _PATHS.get(path_key, circle_path)
                        state    = _init_state(path_fn)
                        pids     = _make_pids(gains)
                        t        = 0.0

                    elif cmd == 'set_wind':
                        wind = _make_wind(
                            V_mean   = float(msg.get('V_mean',   3.0)),
                            wind_dir = float(msg.get('wind_dir', 45.0)),
                            sigma_w  = float(msg.get('sigma_w',  1.5)),
                        )

                    elif cmd == 'set_gains':
                        gains = {k: float(v) for k, v in msg.items() if k != 'cmd'}
                        pids  = _make_pids(gains)

                    elif cmd == 'pause':
                        paused = True

                    elif cmd == 'resume':
                        paused = False

                    elif cmd == 'set_custom_path':
                        ts_a = np.array(msg['times'], dtype=float)
                        xs_a = np.array(msg['xs'],    dtype=float)
                        ys_a = np.array(msg['ys'],    dtype=float)
                        zs_a = np.array(msg['zs'],    dtype=float)
                        T_p  = float(ts_a[-1])
                        def _custom(t,_ts=ts_a,_xs=xs_a,_ys=ys_a,_zs=zs_a,_T=T_p):
                            tm = t % _T
                            return np.array([float(np.interp(tm,_ts,_xs)),
                                             float(np.interp(tm,_ts,_ys)),
                                             float(np.interp(tm,_ts,_zs)), 0.0])
                        path_key = 'Custom'
                        path_fn  = _custom
                        state    = _init_state(path_fn)
                        pids     = _make_pids(gains)
                        t        = 0.0

                    elif cmd == 'reset':
                        state  = _init_state(path_fn)
                        pids   = _make_pids(gains)
                        t      = 0.0
                        paused = False

                except (KeyError, ValueError, TypeError, IndexError, AttributeError):
                    pass  # malformed command payload — ignore, keep listening
                finally:
                    recv_task = asyncio.create_task(ws.receive_json())

            # ── Physics steps ─────────────────────────────────────────────────
            if not paused:
                for _ in range(STEPS_FRAME):
                    sp  = path_fn(t)
                    vd  = np.array([state[1], state[3], state[5]])
                    vw  = wind.velocity(DT, altitude=max(state[4], 0.1), v_drone=vd)
                    Fw  = wind_force(vw, vd)
                    T, tau = control_step(state, pids, sp, DT)
                    state  = rk4_step(state, T, tau, DT, Fw)
                    t     += DT

            # ── Send state packet ─────────────────────────────────────────────
            ex   = float(sp[0] - state[0])
            ey   = float(sp[1] - state[2])
            ez   = float(sp[2] - state[4])
            emag = float(np.sqrt(ex**2 + ey**2 + ez**2))
            # Wind velocity at current altitude (for browser arrow visualisation)
            vw_drone = np.array([state[1], state[3], state[5]])
            vw   = wind.velocity(0.0, altitude=max(state[4], 0.1), v_drone=vw_drone)

            await ws.send_json({
                'x': float(state[0]), 'y': float(state[2]), 'z': float(state[4]),
                'phi': float(state[6]), 'th': float(state[8]), 'psi': float(state[10]),
                'T': float(T),
                't': round(t, 3),
                'ex': ex, 'ey': ey, 'ez': ez, 'emag': emag,
                'sp_x': float(sp[0]), 'sp_y': float(sp[1]), 'sp_z': float(sp[2]),
                'path': path_key,
                # Wind vector [sim x, sim y, sim z] for 3D arrow
                'vwx': float(vw[0]), 'vwy': float(vw[1]), 'vwz': float(vw[2]),
            })

            await asyncio.sleep(FRAME_SLEEP)

    except WebSocketDisconnect:
        recv_task.cancel()
    except Exception:
        recv_task.cancel()
        raise


# ── Static serving ────────────────────────────────────────────────────────────

@app.get('/')
async def index():
    return FileResponse(os.path.join(_WEB_DIR, 'index.html'))
