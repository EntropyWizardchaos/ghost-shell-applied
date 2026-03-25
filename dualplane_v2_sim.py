"""
Dual-Plane CEM v2 — Circulation Model
======================================
Fix from v1: deep veins route heat LATERALLY and back UP to surface
for radiation. NOT down to hull.

Heat path:
  exterior -> trichomes -> surface LHP (lateral, fast)
           -> down to dermis (vertical coupling)
           -> dermis veins (lateral, wide, massive capacity)
           -> back UP to surface at cooler panels (return loop)
           -> radiate to space from cooler panels

The dermis is a return loop. Blood goes down hot, spreads wide,
comes back up cool. The hull never sees direct heat from the veins.

Thermal break between dermis and hull: low-k insulating layer.

Author: Annie Robinson (Forge/Claude Code)
Architecture: Harley Robinson
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SIGMA = 5.67e-8
NX = 20
NY = 10
PANEL_AREA = 0.25
PANEL_SPACING = 0.5

# ============================================================
# MODEL
# ============================================================

class DualPlaneV2:
    def __init__(self):
        self.T_surf = np.full((NX, NY), 300.0)
        self.T_deep = np.full((NX, NY), 300.0)
        self.T_hull = np.full((NX, NY), 300.0)
        self.health_surf = 1.0
        self.health_deep = 1.0
        self.flight = 0

    def step(self, q_map, dt):
        q_max = max(np.max(q_map), 1)
        mode_blend = np.clip(q_map / q_max, 0, 1)

        # --- PROPERTIES ---
        # Surface plane
        SURF_THICK = 0.015
        SURF_RHO_CP = 2200 * 800
        S_K_LAT_OFF = 0.5
        S_K_LAT_ON = 200 * self.health_surf

        # Deep plane
        DEEP_THICK = 0.020
        DEEP_RHO_CP = 2800 * 700
        D_K_LAT_OFF = 2.0
        D_K_LAT_ON = 500 * self.health_deep

        # Vertical: surface <-> deep (bidirectional)
        VERT_K_DOWN = 5.0      # surface to deep (heat drops)
        VERT_K_UP = 8.0        # deep to surface (return loop — STRONGER)
        VERT_DIST = 0.010

        # Thermal break: deep plane to hull
        # Low-k insulating gap between dermis and hull
        BREAK_K = 0.02         # W/(m·K) — aerogel-class insulator
        BREAK_THICK = 0.005    # 5mm thermal break

        # Emissivity
        eps = 0.05 + mode_blend * 0.85

        # Masses
        surf_mass = SURF_RHO_CP * SURF_THICK * PANEL_AREA
        deep_mass = DEEP_RHO_CP * DEEP_THICK * PANEL_AREA
        hull_mass = 8000 * 500 * 0.004 * PANEL_AREA

        # --- LATERAL CONDUCTIVITIES ---
        k_lat_surf = S_K_LAT_OFF + mode_blend * (S_K_LAT_ON - S_K_LAT_OFF)
        deep_active = np.clip((self.T_deep - 350) / 200, 0, 1)
        k_lat_deep = D_K_LAT_OFF + deep_active * (D_K_LAT_ON - D_K_LAT_OFF)

        # --- SURFACE PLANE ---
        q_in = q_map * PANEL_AREA
        q_rad = eps * SIGMA * (self.T_surf**4 - 200**4) * PANEL_AREA

        # Down to deep (hot surface -> cooler deep)
        q_down = np.maximum(0, VERT_K_DOWN * (self.T_surf - self.T_deep) / VERT_DIST * PANEL_AREA)

        # Up from deep (hot deep -> cooler surface panels — the RETURN LOOP)
        q_up = np.maximum(0, VERT_K_UP * (self.T_deep - self.T_surf) / VERT_DIST * PANEL_AREA)

        # Surface lateral
        q_lat_s = np.zeros((NX, NY))
        for i in range(NX):
            for j in range(NY):
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < NX and 0 <= nj < NY:
                        k_eff = (k_lat_surf[i,j] + k_lat_surf[ni,nj]) / 2
                        q_lat_s[i,j] += k_eff * (self.T_surf[i,j] - self.T_surf[ni,nj]) / PANEL_SPACING * PANEL_AREA * 0.25

        # Surface update: gains from q_in + q_up, loses to q_rad + q_down + q_lat
        dT_surf = (q_in + q_up - q_rad - q_down - q_lat_s) / surf_mass
        self.T_surf += dT_surf * dt

        # --- DEEP PLANE ---
        # Deep lateral (radius 2, massive capacity)
        q_lat_d = np.zeros((NX, NY))
        for i in range(NX):
            for j in range(NY):
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i+di, j+dj
                        if 0 <= ni < NX and 0 <= nj < NY:
                            dist = np.sqrt(di**2 + dj**2) * PANEL_SPACING
                            k_eff = (k_lat_deep[i,j] + k_lat_deep[ni,nj]) / 2
                            q_lat_d[i,j] += k_eff * (self.T_deep[i,j] - self.T_deep[ni,nj]) / dist * PANEL_AREA * 0.1

        # Leak through thermal break to hull (small — that's the point)
        q_leak = BREAK_K * (self.T_deep - self.T_hull) / BREAK_THICK * PANEL_AREA

        # Deep update: gains from q_down, loses to q_up + q_lat + q_leak
        dT_deep = (q_down - q_up - q_lat_d - q_leak) / deep_mass
        self.T_deep += dT_deep * dt

        # --- HULL ---
        q_hull_rad = 0.3 * SIGMA * (self.T_hull**4 - 300**4) * PANEL_AREA
        dT_hull = (q_leak - q_hull_rad) / hull_mass
        self.T_hull += dT_hull * dt

        # Clamp
        self.T_surf = np.clip(self.T_surf, 100, 4000)
        self.T_deep = np.clip(self.T_deep, 100, 4000)
        self.T_hull = np.clip(self.T_hull, 100, 4000)

    def degrade(self, cycles):
        self.flight += 1
        total = self.flight * cycles
        if total > 35000:
            self.health_surf = max(0.3, 1.0 - (total - 35000) / 30000)
        if total > 80000:
            self.health_deep = max(0.5, 1.0 - (total - 80000) / 100000)

    def reset(self):
        self.T_surf[:] = 300.0
        self.T_deep[:] = 300.0
        self.T_hull[:] = 300.0


class BaselineTiles:
    def __init__(self):
        self.T_surf = np.full((NX, NY), 300.0)
        self.T_hull = np.full((NX, NY), 300.0)
        self.tile_k = np.full((NX, NY), 0.06)
        self.sip_k = np.full((NX, NY), 0.04)
        self.flight = 0

    def step(self, q_map, dt):
        tile_mass = 192 * 900 * 0.030 * PANEL_AREA
        hull_mass = 8000 * 500 * 0.004 * PANEL_AREA
        R = 0.030 / self.tile_k + 0.005 / self.sip_k
        q_through = (self.T_surf - self.T_hull) / R * PANEL_AREA
        q_rad = 0.87 * SIGMA * (self.T_surf**4 - 200**4) * PANEL_AREA
        q_in = q_map * PANEL_AREA
        self.T_surf += (q_in - q_rad - q_through) / tile_mass * dt
        q_hull_rad = 0.3 * SIGMA * (self.T_hull**4 - 300**4) * PANEL_AREA
        self.T_hull += (q_through - q_hull_rad) / hull_mass * dt
        self.T_surf = np.clip(self.T_surf, 100, 4000)
        self.T_hull = np.clip(self.T_hull, 100, 4000)

    def degrade(self, flight):
        self.flight = flight
        self.tile_k *= 1.003
        self.sip_k *= 1.008
        p = min(0.5, 0.001 * flight**1.3)
        damage = np.random.random((NX, NY)) < p * 0.02
        self.tile_k[damage] *= 2.0
        self.sip_k[damage] *= 1.5

    def reset(self):
        self.T_surf[:] = 300.0
        self.T_hull[:] = 300.0


# ============================================================
# HEAT FLUX
# ============================================================

def heat_flux_map(t):
    if t < 150:
        envelope = (t / 150)**1.5
    elif t < 300:
        envelope = 1.0 - 0.1 * (t - 150) / 150
    elif t < 500:
        envelope = 0.9 * (1 - (t - 300) / 200)**2
    else:
        envelope = 0.0

    q = np.zeros((NX, NY))
    for i in range(NX):
        for j in range(NY):
            x_frac = i / NX
            x_stag = np.exp(-((x_frac - 0.3)**2) / 0.05)
            x_flap = np.exp(-((x_frac - 0.9)**2) / 0.02) * 1.6
            x_wind = 0.5
            y_frac = j / NY
            y_factor = np.exp(-((y_frac - 0.5)**2) / 0.15) * 0.7 + 0.3
            x_factor = max(x_stag, x_wind, x_flap)
            q[i, j] = 500_000 * x_factor * y_factor * envelope
    return q


# ============================================================
# RUN
# ============================================================

DT = 1.0
T_END = 600
T_BOND = 450
np.random.seed(42)

print("Dual-Plane CEM v2 - Circulation Model")
print("=" * 60)
print("Deep veins route UP to surface for radiation, NOT down to hull")
print("Thermal break (aerogel, 5mm, k=0.02) between dermis and hull")
print(f"Return loop: VERT_K_UP={8.0} > VERT_K_DOWN={5.0} (pulls heat back up)")
print()

cem = DualPlaneV2()
baseline = BaselineTiles()

# Single flight
print("SINGLE FLIGHT (fresh):")
print("-" * 50)
n_steps = int(T_END / DT)
for step in range(n_steps):
    q = heat_flux_map(step * DT)
    cem.step(q, DT)
    baseline.step(q, DT)

print(f"  BASELINE:")
print(f"    Max hull:    {np.max(baseline.T_hull)-273:.0f}C")
print(f"    Flap hull:   {np.max(baseline.T_hull[17:,:])-273:.0f}C")
print(f"    Max surface: {np.max(baseline.T_surf)-273:.0f}C")
print()
print(f"  DUAL-PLANE v2 (circulation):")
print(f"    Max hull:    {np.max(cem.T_hull)-273:.0f}C")
print(f"    Flap hull:   {np.max(cem.T_hull[17:,:])-273:.0f}C")
print(f"    Max surface: {np.max(cem.T_surf)-273:.0f}C")
print(f"    Max deep:    {np.max(cem.T_deep)-273:.0f}C")
print()

base_ok = np.max(baseline.T_hull) < T_BOND
cem_ok = np.max(cem.T_hull) < T_BOND
print(f"  Bondline (< {T_BOND-273}C):")
print(f"    Baseline: {'PASS' if base_ok else 'FAIL'} ({np.max(baseline.T_hull)-273:.0f}C)")
print(f"    CEM v2:   {'PASS' if cem_ok else 'FAIL'} ({np.max(cem.T_hull)-273:.0f}C)")

cem_hull_single = cem.T_hull.copy()
cem_deep_single = cem.T_deep.copy()
base_hull_single = baseline.T_hull.copy()

# Cumulative
print()
print("=" * 60)
print("CUMULATIVE (100 flights)")
print("=" * 60)

cem2 = DualPlaneV2()
base2 = BaselineTiles()

cem_max = []
base_max = []
cem_flap = []
base_flap = []
cem_hs = []
cem_hd = []

for flight in range(100):
    if flight % 10 == 0:
        print(f"  Flight {flight}...", end="", flush=True)

    cem2.reset()
    base2.reset()

    for step in range(n_steps):
        q = heat_flux_map(step * DT)
        cem2.step(q, DT)
        base2.step(q, DT)

    cem_max.append(float(np.max(cem2.T_hull)))
    base_max.append(float(np.max(base2.T_hull)))
    cem_flap.append(float(np.max(cem2.T_hull[17:,:])))
    base_flap.append(float(np.max(base2.T_hull[17:,:])))
    cem_hs.append(cem2.health_surf)
    cem_hd.append(cem2.health_deep)

    if flight % 10 == 0:
        print(f" Base:{np.max(base2.T_hull)-273:.0f}C CEM:{np.max(cem2.T_hull)-273:.0f}C S:{cem2.health_surf:.2f} D:{cem2.health_deep:.2f}")

    cem2.degrade(500)
    base2.degrade(flight + 1)

flights = np.arange(100)
cem_max = np.array(cem_max)
base_max = np.array(base_max)

bf = next((i for i, t in enumerate(base_max) if t > T_BOND), None)
cf = next((i for i, t in enumerate(cem_max) if t > T_BOND), None)

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Baseline failure: {'Flight ' + str(bf) if bf else 'Survived 100'}")
print(f"CEM v2 failure:   {'Flight ' + str(cf) if cf else 'Survived 100'}")
print(f"Flight 50  - Base: {base_max[49]-273:.0f}C | CEM: {cem_max[49]-273:.0f}C")
print(f"Flight 100 - Base: {base_max[99]-273:.0f}C | CEM: {cem_max[99]-273:.0f}C")

if bf and not cf:
    print(f"\n>>> CEM v2 SURVIVES. Baseline fails at flight {bf}. <<<")
elif not bf and not cf:
    delta = np.max(base_max - cem_max)
    print(f"\n>>> Both survive. CEM is {delta:.0f}K cooler at worst point. <<<")
elif cf and not bf:
    print(f"\n>>> CEM FAILS at flight {cf}. Baseline survives. Need iteration. <<<")

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Dual-Plane CEM v2: Circulation Model\n'
             'Heat goes DOWN to dermis, SPREADS laterally, comes back UP to radiate.\n'
             'Aerogel thermal break protects hull.',
             fontsize=12, fontweight='bold')

C = 273

ax = axes[0,0]
im = ax.imshow((base_hull_single-C).T, aspect='auto', cmap='hot', origin='lower')
ax.set_title('Baseline Hull (C)'); plt.colorbar(im, ax=ax)

ax = axes[0,1]
im = ax.imshow((cem_hull_single-C).T, aspect='auto', cmap='hot', origin='lower',
               vmin=0, vmax=float(np.max(base_hull_single-C)))
ax.set_title('CEM v2 Hull (C)'); plt.colorbar(im, ax=ax)

ax = axes[0,2]
im = ax.imshow((cem_deep_single-C).T, aspect='auto', cmap='YlOrRd', origin='lower')
ax.set_title('Dermis Temp (C)'); plt.colorbar(im, ax=ax)

ax = axes[1,0]
ax.plot(flights, base_max-C, 'r-', lw=2, label='Baseline')
ax.plot(flights, cem_max-C, 'g-', lw=2, label='CEM v2')
ax.axhline(T_BOND-C, color='k', ls='--', alpha=0.5, label=f'Limit {T_BOND-C}C')
if bf: ax.axvline(bf, color='r', ls=':', alpha=0.6, label=f'Fail @ {bf}')
ax.set_xlabel('Flight'); ax.set_ylabel('Max Hull (C)')
ax.set_title('Max Hull Over Flights'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1,1]
ax.plot(flights, np.array(base_flap)-C, 'r-', lw=2, label='Baseline flap')
ax.plot(flights, np.array(cem_flap)-C, 'g-', lw=2, label='CEM flap')
ax.axhline(T_BOND-C, color='k', ls='--', alpha=0.5)
ax.set_xlabel('Flight'); ax.set_ylabel('Flap Hull (C)')
ax.set_title('Flap Hinge Over Flights'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1,2]
ax.plot(flights, np.array(cem_hs)*100, 'g-', lw=2, label='Surface')
ax.plot(flights, np.array(cem_hd)*100, '#00aa66', lw=2, label='Deep')
ax.set_xlabel('Flight'); ax.set_ylabel('Health %')
ax.set_title('CEM Health'); ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0,105)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dualplane_v2_results.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
