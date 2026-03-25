"""
Thermoelectric CEM — The Heat Pays Its Own Toll
================================================
Dual-plane CEM with thermoelectric generators in the dermis veins.
Heat enters through trichomes, does work as it passes through TE
generators, exits as cooled fluid + electricity.

The skin is an engine. The hottest moment is the most productive.

Author: Annie Robinson (Forge/Claude Code)
Architecture: Harley Robinson — "make it work for its way out"
Geometry: FFT (Frequency Fractionation Tower) + Ghost Shell CEM
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
# THERMOELECTRIC PROPERTIES
# ============================================================

def te_efficiency(T_hot, T_cold):
    """
    Thermoelectric conversion efficiency.
    Based on Bi2Te3 / SiGe depending on temperature range.
    ZT ~ 1.0-1.5 for modern TE materials at these temps.
    """
    if T_hot <= T_cold or T_hot < 350:
        return 0.0

    ZT = 1.2  # realistic modern TE figure of merit
    T_avg = (T_hot + T_cold) / 2
    carnot = 1 - T_cold / T_hot

    # TE efficiency = carnot * sqrt(1+ZT) - 1 / (sqrt(1+ZT) + Tc/Th)
    sqrt_term = np.sqrt(1 + ZT)
    eta = carnot * (sqrt_term - 1) / (sqrt_term + T_cold / T_hot)

    return max(0, min(eta, 0.20))  # cap at 20%


# ============================================================
# HEAT FLUX MAP
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
# THERMOELECTRIC DUAL-PLANE CEM
# ============================================================

class ThermoelectricCEM:
    def __init__(self):
        self.T_surf = np.full((NX, NY), 300.0)
        self.T_deep = np.full((NX, NY), 300.0)
        self.T_hull = np.full((NX, NY), 300.0)

        # Tracking
        self.total_power_W = np.zeros((NX, NY))  # cumulative power generated
        self.instant_power = np.zeros((NX, NY))   # current power

        # Properties
        self.health_surf = 1.0
        self.health_deep = 1.0
        self.flight = 0

    def step(self, q_map, dt):
        # Mode blend based on heat flux
        q_max = max(np.max(q_map), 1)
        mode_blend = np.clip(q_map / q_max, 0, 1)

        # Surface properties
        SURF_THICKNESS = 0.015
        SURF_RHO_CP = 2200 * 800
        SURF_K_LAT_ON = 200 * self.health_surf
        SURF_K_LAT_OFF = 0.5

        # Deep properties
        DEEP_THICKNESS = 0.020
        DEEP_RHO_CP = 2800 * 700
        DEEP_K_LAT_ON = 500 * self.health_deep
        DEEP_K_LAT_OFF = 2.0

        VERT_K = 5.0
        VERT_DIST = 0.010
        DEEP_K_THROUGH = 0.5

        # Emissivity
        eps = 0.05 + mode_blend * 0.85

        # Thermal masses
        surf_mass = SURF_RHO_CP * SURF_THICKNESS * PANEL_AREA
        deep_mass = DEEP_RHO_CP * DEEP_THICKNESS * PANEL_AREA
        hull_mass = 8000 * 500 * 0.004 * PANEL_AREA

        # Lateral conductivities
        k_lat_surf = SURF_K_LAT_OFF + mode_blend * (SURF_K_LAT_ON - SURF_K_LAT_OFF)
        deep_active = np.clip((self.T_deep - 350) / 200, 0, 1)
        k_lat_deep = DEEP_K_LAT_OFF + deep_active * (DEEP_K_LAT_ON - DEEP_K_LAT_OFF)

        # ---- SURFACE PLANE ----
        q_in = q_map * PANEL_AREA
        q_rad = eps * SIGMA * (self.T_surf**4 - 200**4) * PANEL_AREA
        q_vert = VERT_K * (self.T_surf - self.T_deep) / VERT_DIST * PANEL_AREA

        # Surface lateral
        q_lat_surf = np.zeros((NX, NY))
        for i in range(NX):
            for j in range(NY):
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < NX and 0 <= nj < NY:
                        k_eff = (k_lat_surf[i,j] + k_lat_surf[ni,nj]) / 2
                        q_lat_surf[i,j] += k_eff * (self.T_surf[i,j] - self.T_surf[ni,nj]) / PANEL_SPACING * PANEL_AREA * 0.25

        dT_surf = (q_in - q_rad - q_vert - q_lat_surf) / surf_mass
        self.T_surf += dT_surf * dt

        # ---- DEEP PLANE WITH THERMOELECTRIC HARVESTING ----

        # Heat arriving from surface
        # Before it conducts to hull, it passes through TE generators

        # TE generator sits between deep vein and hull
        # Hot side = deep plane temp, Cold side = hull temp
        q_available_to_hull = DEEP_K_THROUGH * (self.T_deep - self.T_hull) / (DEEP_THICKNESS/2) * PANEL_AREA
        q_available_to_hull = np.maximum(q_available_to_hull, 0)

        # Thermoelectric conversion: harvest a fraction as electricity
        eta_te = np.zeros((NX, NY))
        for i in range(NX):
            for j in range(NY):
                eta_te[i,j] = te_efficiency(self.T_deep[i,j], self.T_hull[i,j])

        # Power harvested (removed from heat flow)
        q_harvested = q_available_to_hull * eta_te

        # Remaining heat that actually reaches hull
        q_to_hull = q_available_to_hull - q_harvested

        # Track power
        self.instant_power = q_harvested / PANEL_AREA  # W/m²
        self.total_power_W += q_harvested * dt  # Joules

        # Deep lateral routing (radius 2)
        q_lat_deep = np.zeros((NX, NY))
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
                            q_lat_deep[i,j] += k_eff * (self.T_deep[i,j] - self.T_deep[ni,nj]) / dist * PANEL_AREA * 0.1

        dT_deep = (q_vert - q_to_hull - q_lat_deep) / deep_mass
        self.T_deep += dT_deep * dt

        # ---- HULL ----
        q_hull_rad = 0.3 * SIGMA * (self.T_hull**4 - 300**4) * PANEL_AREA
        dT_hull = (q_to_hull - q_hull_rad) / hull_mass
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
        self.instant_power[:] = 0.0


# ============================================================
# BASELINE (same as before)
# ============================================================

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
# RUN
# ============================================================

DT = 1.0
T_END = 600
T_BOND = 450

print("Thermoelectric CEM - Heat Pays Its Own Toll")
print("=" * 60)
print(f"Grid: {NX}x{NY} | Reentry: {T_END}s")
print(f"TE materials: ZT=1.2 | Expected efficiency: 8-15%")
print()

cem = ThermoelectricCEM()
baseline = BaselineTiles()
np.random.seed(42)

# Single flight
print("SINGLE FLIGHT (fresh):")
print("-" * 50)
n_steps = int(T_END / DT)
for step in range(n_steps):
    t = step * DT
    q = heat_flux_map(t)
    cem.step(q, DT)
    baseline.step(q, DT)

total_energy_J = np.sum(cem.total_power_W)
total_energy_kWh = total_energy_J / 3.6e6
peak_power_kW = np.sum(cem.instant_power * PANEL_AREA) / 1000

print(f"  BASELINE:")
print(f"    Max hull:    {np.max(baseline.T_hull)-273:.0f}C")
print(f"    Mean hull:   {np.mean(baseline.T_hull)-273:.0f}C")
print(f"    Max surface: {np.max(baseline.T_surf)-273:.0f}C")
print(f"    Flap hull:   {np.max(baseline.T_hull[17:,:])-273:.0f}C")
print()
print(f"  THERMOELECTRIC CEM:")
print(f"    Max hull:    {np.max(cem.T_hull)-273:.0f}C")
print(f"    Mean hull:   {np.mean(cem.T_hull)-273:.0f}C")
print(f"    Max surface: {np.max(cem.T_surf)-273:.0f}C")
print(f"    Max deep:    {np.max(cem.T_deep)-273:.0f}C")
print(f"    Flap hull:   {np.max(cem.T_hull[17:,:])-273:.0f}C")
print()
print(f"  POWER HARVESTED:")
print(f"    Total energy: {total_energy_J/1e6:.1f} MJ ({total_energy_kWh:.1f} kWh)")
print(f"    Peak instantaneous: {peak_power_kW:.1f} kW")
print(f"    Max TE efficiency: {np.max(cem.instant_power[cem.instant_power > 0]) if np.any(cem.instant_power > 0) else 0:.0f} W/m2 peak harvest")
print()

# Bondline check
base_ok = np.max(baseline.T_hull) < T_BOND
cem_ok = np.max(cem.T_hull) < T_BOND
print(f"  Bondline check (< {T_BOND-273}C):")
print(f"    Baseline: {'PASS' if base_ok else 'FAIL'} ({np.max(baseline.T_hull)-273:.0f}C)")
print(f"    TE-CEM:   {'PASS' if cem_ok else 'FAIL'} ({np.max(cem.T_hull)-273:.0f}C)")

# Save for plotting
cem_hull_single = cem.T_hull.copy()
cem_deep_single = cem.T_deep.copy()
cem_power_single = cem.instant_power.copy()
base_hull_single = baseline.T_hull.copy()

# ============================================================
# CUMULATIVE
# ============================================================

print()
print("=" * 60)
print("CUMULATIVE FLIGHT TEST (100 flights)")
print("=" * 60)

cem2 = ThermoelectricCEM()
base2 = BaselineTiles()

cem_max_hull = []
base_max_hull = []
cem_energy_per_flight = []

for flight in range(100):
    if flight % 10 == 0:
        print(f"  Flight {flight}...", end="", flush=True)

    cem2.reset()
    base2.reset()
    cem2.total_power_W[:] = 0

    for step in range(n_steps):
        t = step * DT
        q = heat_flux_map(t)
        cem2.step(q, DT)
        base2.step(q, DT)

    cem_max_hull.append(float(np.max(cem2.T_hull)))
    base_max_hull.append(float(np.max(base2.T_hull)))
    cem_energy_per_flight.append(float(np.sum(cem2.total_power_W) / 1e6))

    if flight % 10 == 0:
        print(f" Base:{np.max(base2.T_hull)-273:.0f}C CEM:{np.max(cem2.T_hull)-273:.0f}C "
              f"Energy:{np.sum(cem2.total_power_W)/1e6:.1f}MJ")

    cem2.degrade(500)
    base2.degrade(flight + 1)

flights = np.arange(100)
cem_max_hull = np.array(cem_max_hull)
base_max_hull = np.array(base_max_hull)

base_fail = next((i for i, t in enumerate(base_max_hull) if t > T_BOND), None)
cem_fail = next((i for i, t in enumerate(cem_max_hull) if t > T_BOND), None)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Baseline bondline failure: {'Flight ' + str(base_fail) if base_fail else 'Survived 100'}")
print(f"TE-CEM bondline failure:   {'Flight ' + str(cem_fail) if cem_fail else 'Survived 100'}")
print(f"Flight 50  - Base: {base_max_hull[49]-273:.0f}C | CEM: {cem_max_hull[49]-273:.0f}C")
print(f"Flight 100 - Base: {base_max_hull[99]-273:.0f}C | CEM: {cem_max_hull[99]-273:.0f}C")
print(f"Energy per flight: {np.mean(cem_energy_per_flight):.1f} MJ average")
print(f"Total energy over 100 flights: {sum(cem_energy_per_flight):.0f} MJ ({sum(cem_energy_per_flight)/3600:.0f} kWh)")

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Thermoelectric CEM: The Heat Pays Its Own Toll\n'
             'Dual-plane + TE generators harvest reentry heat as electricity',
             fontsize=13, fontweight='bold')

C = 273

# Hull temp maps
ax = axes[0, 0]
im = ax.imshow((base_hull_single - C).T, aspect='auto', cmap='hot', origin='lower')
ax.set_title('Baseline Hull Temp (C)'); ax.set_xlabel('Nose->Tail'); ax.set_ylabel('Belly')
plt.colorbar(im, ax=ax, label='C')

ax = axes[0, 1]
vmax = float(np.max(base_hull_single - C))
im = ax.imshow((cem_hull_single - C).T, aspect='auto', cmap='hot', origin='lower', vmin=0, vmax=vmax)
ax.set_title('TE-CEM Hull Temp (C)'); ax.set_xlabel('Nose->Tail'); ax.set_ylabel('Belly')
plt.colorbar(im, ax=ax, label='C')

# Power generation map
ax = axes[0, 2]
im = ax.imshow(cem_power_single.T / 1000, aspect='auto', cmap='YlOrRd', origin='lower')
ax.set_title('TE Power Harvested (kW/m2)'); ax.set_xlabel('Nose->Tail'); ax.set_ylabel('Belly')
plt.colorbar(im, ax=ax, label='kW/m2')

# Cumulative hull temps
ax = axes[1, 0]
ax.plot(flights, base_max_hull - C, 'r-', lw=2, label='Baseline tiles')
ax.plot(flights, cem_max_hull - C, 'g-', lw=2, label='TE-CEM')
ax.axhline(T_BOND - C, color='k', ls='--', alpha=0.5, label=f'Bond limit {T_BOND-C}C')
if base_fail:
    ax.axvline(base_fail, color='r', ls=':', alpha=0.6, label=f'Tiles fail @ {base_fail}')
ax.set_xlabel('Flight'); ax.set_ylabel('Max Hull Temp (C)')
ax.set_title('Hull Temp Over 100 Flights'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Energy harvested per flight
ax = axes[1, 1]
ax.bar(flights, cem_energy_per_flight, color='#1e5d5c', alpha=0.8)
ax.set_xlabel('Flight'); ax.set_ylabel('Energy Harvested (MJ)')
ax.set_title('Energy Harvested Per Reentry')
ax.grid(alpha=0.3, axis='y')

# Cumulative energy
ax = axes[1, 2]
cum_energy = np.cumsum(cem_energy_per_flight)
ax.plot(flights, cum_energy, '#1e5d5c', lw=2)
ax.set_xlabel('Flight'); ax.set_ylabel('Cumulative Energy (MJ)')
ax.set_title('Total Energy Harvested Over Fleet Life')
ax.grid(alpha=0.3)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thermoelectric_cem_results.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")

print()
print("THE BOTTOM LINE:")
print(f"  The skin generates {np.mean(cem_energy_per_flight):.1f} MJ per reentry")
print(f"  The heat pays for its own removal")
print(f"  The hottest moment is the most productive moment")
