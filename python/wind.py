import numpy as np


FT_PER_M = 3.28084
ALT_FLOOR_M = 3.0 

class WindModel:
    
    def __init__(self, V_mean, phi_az, sigma_turb, V_gust=8.0, T_gust=1.0, rng=None):
        self.V_mean, self.phi_az, self.sigma_turb = V_mean, phi_az, sigma_turb
        self.V_gust, self.T_gust = V_gust, T_gust
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self._y = np.zeros(3)  
        self._gust_t = None  

    def fire_gust(self):
        self._gust_t = 0.0

    def velocity(self, dt, altitude, v_drone):
        d = np.array([np.cos(self.phi_az), np.sin(self.phi_az), 0.0])
        v_mean = self.V_mean * d
        v_turb = self._turbulence(dt, altitude, v_mean, v_drone)
        v_gust = self._gust(dt) * d
        return v_mean + v_turb + v_gust

    def _turbulence(self, dt, altitude, v_mean, v_drone):
        z_ft = max(altitude, ALT_FLOOR_M) * FT_PER_M
        L_w_ft = z_ft
        L_uv_ft = z_ft / (0.177 + 0.000823 * z_ft) ** 1.2
        L = np.array([L_uv_ft, L_uv_ft, L_w_ft]) / FT_PER_M
        # true relative airspeed, not just mean wind speed -- matches the
        # relative velocity wind_force() computes downstream, since it's the
        # vehicle's speed through the frozen turbulence pattern that sets the
        # filter's time constant, not the wind's speed on its own
        V_air = max(np.linalg.norm(v_mean - v_drone), 1.0)
        alpha = dt * V_air / L
        # exact OU-process transition, not a small-step approximation --
        # Madden (NASA/NTRS 20190000875) documents this exponential form as
        # what verified flight-sim turbulence filters (LaSRS++) actually use,
        # citing Beal 1993, JGCD 16(1):132-138. Stationary variance is exactly
        # sigma_turb^2 for any alpha, so no clip is needed here.
        self._y = np.exp(-alpha) * self._y + self.sigma_turb * np.sqrt(1 - np.exp(-2 * alpha)) * self.rng.standard_normal(3)
        return self._y

    def _gust(self, dt):
        if self._gust_t is None:
            return 0.0
        if self._gust_t > self.T_gust:
            self._gust_t = None
            return 0.0
        gf = 0.5 * self.V_gust * (1 - np.cos(2 * np.pi * self._gust_t / self.T_gust))
        self._gust_t += dt
        return gf

def wind_force(v_wind, v_drone, k_drag=0.01):
    return k_drag * (v_wind - v_drone)
