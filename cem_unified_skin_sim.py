"""
CEM Unified Skin — One Skin, Two Modes
=======================================
The merge: CEM trichome panels ARE the thermal protection system.
Not tiles + underlayer. One continuous adaptive skin.

Reentry mode: route heat from hot zones to cool zones laterally
Orbital mode: route solar heat to radiators, protect fuel

2D thermal model: grid of CEM panels with lateral heat routing.
Each panel can PROTECT, VENT, or DUMP independently.

This is the Ghost Shell's Electrodermus applied to Starship as
the primary TPS — not a retrofit, a replacement.

Author: Annie Robinson (Forge/Claude Code)
Origin: Harley Robinson, Ghost Shell CEM + Starship Reentry seeds
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# CEM PANEL PROPERTIES
# ============================================================

class CEMPanel:
    """Single CEM trichome panel — the building block."""

    def __init__(self):
        # Physical
        self.area = 0.25          # m² per panel (500x500mm)
        self.thickness = 0.035    # m total (trichome layer + LHP + backing)
        self.mass_per_m2 = 3.5    # kg/m² (trichome + LHP + structure + actuators)

        # Thermal properties (effective, mode-dependent)
        self.k_protect = 0.03     # W/(m·K) — closed trichomes, better than MLI
        self.k_vent = 0.5         # W/(m·K) — partial routing
        self.k_dump = 5.0         # W/(m·K) — full routing, heat highway

        # Lateral conductivity (through LHP corkscrew to neighbors)
        self.k_lateral_off = 0.1  # W/(m·K) — LHP idle
        self.k_lateral_on = 200   # W/(m·K) — LHP active, corkscrew pumping

        # Trichome properties
        self.emissivity_closed = 0.05   # low — reflects/insulates
        self.emissivity_open = 0.90     # high — radiates hard
        self.max_dump_W = 2500 * 0.25   # W per panel at full dump (2500 W/m² * area)

        # Durability
        self.cycle_life = 50000         # trichome open/close cycles before degradation
        self.cycles_used = 0
        self.health = 1.0               # 1.0 = perfect, 0.0 = dead

        # State
        self.T = 300.0            # K current temperature
        self.mode = 'PROTECT'     # PROTECT, VENT, DUMP

    def degrade(self, cycles_this_flight):
        """Degrade panel based on thermal cycling."""
        self.cycles_used += cycles_this_flight
        # Gradual degradation — not cliff-edge like ceramic
        if self.cycles_used > self.cycle_life * 0.7:
            # Past 70% of life, start losing efficiency
            wear = (self.cycles_used - self.cycle_life * 0.7) / (self.cycle_life * 0.3)
            self.health = max(0.1, 1.0 - wear * 0.5)  # never goes below 10%

    def effective_k_through(self):
        """Through-thickness conductivity based on mode and health."""
        if self.mode == 'PROTECT':
            return self.k_protect * self.health
        elif self.mode == 'VENT':
            return self.k_vent * self.health
        else:  # DUMP
            return self.k_dump * self.health

    def effective_k_lateral(self):
        """Lateral conductivity to neighbors based on mode and health."""
        if self.mode == 'PROTECT':
            return self.k_lateral_off
        elif self.mode == 'VENT':
            blend = 0.5
            return (self.k_lateral_off * (1-blend) + self.k_lateral_on * blend) * self.health
        else:  # DUMP
            return self.k_lateral_on * self.health

    def effective_emissivity(self):
        """Surface emissivity based on mode."""
        if self.mode == 'PROTECT':
            return self.emissivity_closed
        elif self.mode == 'VENT':
            return (self.emissivity_closed + self.emissivity_open) / 2
        else:
            return self.emissivity_open

    def choose_mode(self, q_incident):
        """Autonomously choose mode based on local heat flux."""
        if q_incident < 1000:       # < 1 kW/m² — shadow or cool zone
            self.mode = 'PROTECT'
        elif q_incident < 50000:    # < 50 kW/m² — moderate heating
            self.mode = 'VENT'
        else:                        # > 50 kW/m² — heavy reentry heating
            self.mode = 'DUMP'


# ============================================================
# STARSHIP SKIN GRID
# ============================================================

class StarshipSkin:
    """
    2D grid of CEM panels wrapping Starship belly.
    Models lateral heat routing between panels.
    """

    def __init__(self, nx=20, ny=10):
        """
        nx: panels along length (nose to tail)
        ny: panels around circumference (belly wrap)
        """
        self.nx = nx
        self.ny = ny
        self.panels = [[CEMPanel() for _ in range(ny)] for _ in range(nx)]
        self.hull_T = np.full((nx, ny), 300.0)  # hull temp behind each panel
        self.hull_k = 16.0    # steel conductivity
        self.hull_L = 0.004   # hull thickness
        self.hull_rho_cp = 8000 * 500  # rho*cp for steel

    def heat_flux_map(self, phase='reentry'):
        """
        Heat flux distribution across the belly.
        Reentry: peak at nose, falls toward tail. Peak at centerline.
        Orbital: solar flux on sun-facing side only.
        """
        q = np.zeros((self.nx, self.ny))

        if phase == 'reentry':
            for i in range(self.nx):
                for j in range(self.ny):
                    # Nose-to-tail: peaks at ~30% from nose (stagnation region)
                    x_frac = i / self.nx
                    x_factor = np.exp(-((x_frac - 0.3)**2) / 0.08) * 0.8 + 0.2

                    # Centerline: peaks at belly center
                    y_frac = j / self.ny
                    y_factor = np.exp(-((y_frac - 0.5)**2) / 0.15) * 0.7 + 0.3

                    q[i, j] = 200_000 * x_factor * y_factor  # W/m²

        elif phase == 'orbital':
            for i in range(self.nx):
                for j in range(self.ny):
                    # Sun hits one side
                    y_frac = j / self.ny
                    if y_frac < 0.5:  # sun-facing half
                        q[i, j] = 1361 * 0.35 * np.sin(np.pi * y_frac / 0.5)
                    else:
                        q[i, j] = 0  # shadow side

        return q

    def step(self, q_map, dt=0.1):
        """One timestep of the 2D thermal model."""
        SIGMA = 5.67e-8
        panel_L = 0.035  # panel thickness
        panel_spacing = 0.5  # m between panel centers

        for i in range(self.nx):
            for j in range(self.ny):
                p = self.panels[i][j]

                # Panel chooses its own mode
                p.choose_mode(q_map[i, j])

                # Through-thickness: heat from surface to hull
                k_thru = p.effective_k_through()
                q_through = k_thru * (p.T - self.hull_T[i, j]) / panel_L * p.area

                # Radiation from surface
                eps = p.effective_emissivity()
                q_rad = eps * SIGMA * p.T**4 * p.area

                # Lateral heat routing to neighbors
                k_lat = p.effective_k_lateral()
                q_lateral = 0.0
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        neighbor = self.panels[ni][nj]
                        # Heat flows from hot to cold through LHP
                        q_lat = k_lat * (p.T - neighbor.T) / panel_spacing * p.area * 0.25
                        q_lateral += q_lat

                # Panel temperature update
                # panel thermal mass: rho*cp*V
                panel_rho_cp = 2500 * 700  # effective
                panel_vol = p.area * panel_L
                thermal_mass = panel_rho_cp * panel_vol

                dT_panel = (q_map[i,j] * p.area - q_rad - q_through - q_lateral) / thermal_mass
                p.T += dT_panel * dt
                p.T = np.clip(p.T, 100, 4000)

                # Hull temperature update
                hull_vol = p.area * self.hull_L
                hull_thermal_mass = self.hull_rho_cp * hull_vol
                q_hull_rad = 0.3 * SIGMA * self.hull_T[i,j]**4 * p.area
                dT_hull = (q_through - q_hull_rad) / hull_thermal_mass
                self.hull_T[i,j] += dT_hull * dt
                self.hull_T[i,j] = np.clip(self.hull_T[i,j], 100, 4000)

    def degrade_all(self, cycles):
        """Degrade all panels after a flight."""
        for row in self.panels:
            for p in row:
                p.degrade(cycles)

    def max_hull_temp(self):
        return float(np.max(self.hull_T))

    def mean_hull_temp(self):
        return float(np.mean(self.hull_T))

    def max_surface_temp(self):
        return max(p.T for row in self.panels for p in row)

    def mean_health(self):
        return np.mean([[p.health for p in row] for row in self.panels])

    def reset_temps(self):
        """Cool down between flights."""
        for row in self.panels:
            for p in row:
                p.T = 300.0
        self.hull_T[:] = 300.0


# ============================================================
# BASELINE: TRADITIONAL TILES (simplified 2D)
# ============================================================

class BaselineSkin:
    """Traditional ceramic tiles — passive, no lateral routing."""

    def __init__(self, nx=20, ny=10):
        self.nx = nx
        self.ny = ny
        self.tile_L = np.full((nx, ny), 0.025)    # tile thickness
        self.tile_k = np.full((nx, ny), 1.5)       # tile conductivity
        self.insul_L = np.full((nx, ny), 0.015)    # insulation thickness
        self.insul_k = np.full((nx, ny), 0.05)     # insulation conductivity
        self.hull_T = np.full((nx, ny), 300.0)
        self.surf_T = np.full((nx, ny), 300.0)
        self.hull_rho_cp = 8000 * 500
        self.hull_L = 0.004
        self.panel_area = 0.25

    def step(self, q_map, dt=0.1):
        SIGMA = 5.67e-8
        for i in range(self.nx):
            for j in range(self.ny):
                # Through-thickness conduction (tile + insulation in series)
                R_tile = self.tile_L[i,j] / self.tile_k[i,j]
                R_insul = self.insul_L[i,j] / self.insul_k[i,j]
                R_total = R_tile + R_insul

                q_rad = 0.85 * SIGMA * self.surf_T[i,j]**4 * self.panel_area
                q_in = q_map[i,j] * self.panel_area
                q_through = (self.surf_T[i,j] - self.hull_T[i,j]) / R_total * self.panel_area

                # Surface temp
                tile_rho_cp = 2200 * 900
                tile_vol = self.panel_area * self.tile_L[i,j]
                surf_mass = tile_rho_cp * tile_vol
                dT_surf = (q_in - q_rad - q_through) / surf_mass
                self.surf_T[i,j] += dT_surf * dt
                self.surf_T[i,j] = np.clip(self.surf_T[i,j], 100, 4000)

                # Hull temp
                hull_vol = self.panel_area * self.hull_L
                hull_mass = self.hull_rho_cp * hull_vol
                q_hull_rad = 0.3 * SIGMA * self.hull_T[i,j]**4 * self.panel_area
                dT_hull = (q_through - q_hull_rad) / hull_mass
                self.hull_T[i,j] += dT_hull * dt
                self.hull_T[i,j] = np.clip(self.hull_T[i,j], 100, 4000)

    def degrade(self, flight):
        """Cumulative tile + insulation degradation."""
        # Tile erosion
        self.tile_L *= 0.997          # 0.3% thickness loss per flight
        self.tile_k *= 1.003          # micro-cracks increase conductivity

        # Insulation degradation from gap torching
        self.insul_L *= 0.998
        self.insul_k *= 1.01          # 1% per flight

        # Random tile loss (probabilistic)
        p_loss = min(0.95, 0.001 * flight**1.3)
        for i in range(self.nx):
            for j in range(self.ny):
                if np.random.random() < p_loss * 0.01:  # per-panel probability
                    self.tile_L[i,j] *= 0.3  # catastrophic thinning
                    self.insul_k[i,j] *= 3   # gap torch damage to insulation below

    def max_hull_temp(self):
        return float(np.max(self.hull_T))

    def mean_hull_temp(self):
        return float(np.mean(self.hull_T))

    def reset_temps(self):
        self.hull_T[:] = 300.0
        self.surf_T[:] = 300.0


# ============================================================
# RUN SIMULATION
# ============================================================

MAX_FLIGHTS = 100
REENTRY_DURATION = 400   # seconds
DT = 0.5                 # timestep
T_CRIT = 1200            # K

print("CEM Unified Skin — One Skin, Two Modes")
print("=" * 60)
print(f"Grid: 20x10 panels (200 panels, 50m²)")
print(f"Reentry: {REENTRY_DURATION}s at up to 200 kW/m²")
print(f"Steel critical: {T_CRIT}K")
print(f"Simulating {MAX_FLIGHTS} flights...")
print()

cem_skin = StarshipSkin(nx=20, ny=10)
baseline_skin = BaselineSkin(nx=20, ny=10)

cem_max_hull = []
cem_mean_hull = []
cem_health = []
base_max_hull = []
base_mean_hull = []
cem_failed = None
base_failed = None

np.random.seed(42)

for flight in range(MAX_FLIGHTS):
    if flight % 10 == 0:
        print(f"  Flight {flight}...", end="", flush=True)

    # Get heat flux map for reentry
    q_reentry = cem_skin.heat_flux_map('reentry')

    # Run reentry for both
    n_steps = int(REENTRY_DURATION / DT)
    for step in range(n_steps):
        cem_skin.step(q_reentry, DT)
        baseline_skin.step(q_reentry, DT)

    # Record peak temps
    cem_max_hull.append(cem_skin.max_hull_temp())
    cem_mean_hull.append(cem_skin.mean_hull_temp())
    cem_health.append(cem_skin.mean_health())
    base_max_hull.append(baseline_skin.max_hull_temp())
    base_mean_hull.append(baseline_skin.mean_hull_temp())

    if flight % 10 == 0:
        print(f" CEM:{cem_skin.max_hull_temp():.0f}K Base:{baseline_skin.max_hull_temp():.0f}K health:{cem_skin.mean_health():.2f}")

    # Check failure
    if baseline_skin.max_hull_temp() >= T_CRIT and base_failed is None:
        base_failed = flight
        print(f"\n  *** BASELINE FAILURE at flight {flight}! ***")

    if cem_skin.max_hull_temp() >= T_CRIT and cem_failed is None:
        cem_failed = flight
        print(f"\n  *** CEM FAILURE at flight {flight}! ***")

    # Degrade and cool
    cem_skin.degrade_all(cycles=500)  # ~500 thermal cycles per reentry (heating transients)
    baseline_skin.degrade(flight)
    cem_skin.reset_temps()
    baseline_skin.reset_temps()

# ============================================================
# RESULTS
# ============================================================

flights = np.arange(MAX_FLIGHTS)
cem_max_hull = np.array(cem_max_hull)
base_max_hull = np.array(base_max_hull)

print()
print("=" * 60)
print("RESULTS — CEM Unified Skin vs Ceramic Tiles")
print("=" * 60)
print()
print(f"Baseline first failure: {'Flight ' + str(base_failed) if base_failed else f'Survived {MAX_FLIGHTS} flights'}")
print(f"CEM first failure:      {'Flight ' + str(cem_failed) if cem_failed else f'Survived {MAX_FLIGHTS} flights'}")
print()
print(f"Flight 50  — Baseline max hull: {base_max_hull[49]-273:.0f}°C | CEM: {cem_max_hull[49]-273:.0f}°C")
print(f"Flight 100 — Baseline max hull: {base_max_hull[99]-273:.0f}°C | CEM: {cem_max_hull[99]-273:.0f}°C")
print(f"Max temp advantage: {np.max(base_max_hull - cem_max_hull):.0f}K cooler with CEM")
print(f"CEM panel health at flight 100: {cem_health[-1]:.1%}")
print()

# Vehicle life estimate
if base_failed:
    print(f"Baseline vehicle life: ~{base_failed} flights")
else:
    # Extrapolate
    if len(base_max_hull) > 20:
        slope = (base_max_hull[-1] - base_max_hull[-20]) / 20
        if slope > 0:
            flights_to_crit = (T_CRIT - base_max_hull[-1]) / slope + MAX_FLIGHTS
            print(f"Baseline projected failure: ~flight {flights_to_crit:.0f}")

if cem_failed:
    print(f"CEM vehicle life: ~{cem_failed} flights")
else:
    print(f"CEM vehicle life: >{MAX_FLIGHTS} flights (health: {cem_health[-1]:.1%})")

print()
print("MASS COMPARISON:")
print(f"  CEM skin (50m²): {50 * 3.5:.0f} kg")
print(f"  Ceramic tiles (50m²): {50 * 8:.0f} kg")  # ceramic TPS ~8 kg/m²
print(f"  Mass savings: {50*8 - 50*3.5:.0f} kg lighter with CEM")
print()
print("OPERATIONAL SAVINGS:")
print(f"  No tile inspection/replacement between flights")
print(f"  No gap filler replacement")
print(f"  Self-monitoring (each panel reports its own health)")
print(f"  Graceful degradation instead of catastrophic tile loss")

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('CEM Unified Skin vs Ceramic Tiles — 100 Flight Simulation\n'
             'One adaptive skin replaces passive tiles. Same hardware for reentry + orbital.',
             fontsize=13, fontweight='bold')

# Max hull temp per flight
ax = axes[0,0]
ax.plot(flights, base_max_hull-273, 'r-', lw=2, label='Ceramic tiles', alpha=0.8)
ax.plot(flights, cem_max_hull-273, 'g-', lw=2, label='CEM skin', alpha=0.8)
ax.axhline(T_CRIT-273, color='k', ls='--', alpha=0.4, label=f'Steel critical {T_CRIT-273}°C')
if base_failed:
    ax.axvline(base_failed, color='r', ls=':', alpha=0.6, label=f'Tiles fail @ {base_failed}')
ax.set_xlabel('Flight Number'); ax.set_ylabel('Max Hull Temperature (°C)')
ax.set_title('Peak Hull Temperature Per Flight'); ax.legend(); ax.grid(alpha=0.3)

# Mean hull temp
ax = axes[0,1]
ax.plot(flights, np.array(cem_mean_hull)-273, 'g-', lw=2, label='CEM mean hull')
ax.plot(flights, np.array(base_mean_hull)-273, 'r-', lw=2, label='Tiles mean hull')
ax.set_xlabel('Flight Number'); ax.set_ylabel('Mean Hull Temperature (°C)')
ax.set_title('Average Hull Temperature'); ax.legend(); ax.grid(alpha=0.3)

# CEM health
ax = axes[1,0]
ax.plot(flights, np.array(cem_health)*100, 'g-', lw=2)
ax.set_xlabel('Flight Number'); ax.set_ylabel('Panel Health (%)')
ax.set_title('CEM Panel Health Over Flights'); ax.grid(alpha=0.3)
ax.set_ylim(0, 105)

# Heat map of hull temps at final flight (CEM)
ax = axes[1,1]
# Reconstruct final temps by running one more reentry
q_final = cem_skin.heat_flux_map('reentry')
for _ in range(int(REENTRY_DURATION/DT)):
    cem_skin.step(q_final, DT)
im = ax.imshow(cem_skin.hull_T.T - 273, aspect='auto', cmap='hot', origin='lower')
ax.set_xlabel('Nose → Tail'); ax.set_ylabel('Belly Circumference')
ax.set_title(f'Hull Temperature Map — Flight {MAX_FLIGHTS} Reentry (°C)')
plt.colorbar(im, ax=ax, label='°C')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cem_unified_skin_results.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")
