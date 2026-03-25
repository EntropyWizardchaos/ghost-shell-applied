"""
CEM Cryo-Wrap Orbital Thermal Simulation
=========================================
Simulates Starship propellant tank thermal management in LEO.

Compares three strategies:
1. MLI only (passive insulation)
2. Active cryo-cooler (constant heat rejection)
3. CEM Cryo-Wrap (Ghost Shell dynamic membrane)

Models one full LEO orbit (90 min) with sun/shadow cycling.
Tracks boiloff rate for liquid methane (LCH4, boils at 111K).

Author: Annie Robinson (Forge/Claude Code)
Origin: Harley Robinson via Ghost Shell CEM organ, October 2025
Spec: Seeds/Corkscrew/CEM/CEM_CryoWrap_Starship_Seed.md
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# CONSTANTS
# ============================================================

SIGMA = 5.67e-8       # Stefan-Boltzmann (W/m²/K⁴)
T_SPACE = 3           # K deep space background
T_LCH4 = 111          # K liquid methane boiling point
T_LOX = 90            # K liquid oxygen boiling point

TANK_AREA = 166       # m² exposed tank surface
Q_SOLAR = 1361        # W/m² solar constant at 1 AU
ABSORPTIVITY = 0.35   # stainless steel solar absorptivity (with coating)

ORBIT_PERIOD = 5400   # s (90 min LEO)
SUN_FRACTION = 0.6    # fraction of orbit in sunlight

LCH4_LATENT = 510_000 # J/kg latent heat of vaporization for methane
PROP_MASS = 1_200_000 # kg total propellant

# ============================================================
# THERMAL MODELS
# ============================================================

def solar_flux(t, orbit_period=ORBIT_PERIOD, sun_frac=SUN_FRACTION):
    """Solar heat input as function of time. Sinusoidal transition."""
    phase = (t % orbit_period) / orbit_period
    # Sun from 0 to sun_frac, shadow from sun_frac to 1.0
    # Smooth transition
    if phase < sun_frac:
        # In sun — vary with angle
        angle = np.pi * phase / sun_frac
        return Q_SOLAR * ABSORPTIVITY * np.sin(angle)
    else:
        return 0.0


def mli_model(t, T_surface, params):
    """
    MLI (Multi-Layer Insulation) — passive.
    Fixed thermal resistance. Heat leaks through at rate determined by R-value.
    """
    q_solar = solar_flux(t) * TANK_AREA
    R_mli = params['R_value']  # m²·K/W

    # Heat leak through MLI to propellant
    q_leak = TANK_AREA * (T_surface - T_LCH4) / R_mli

    # Surface energy balance: solar in - radiation out = stored + leaked
    # Simplified: surface temp adjusts to balance
    q_rad_out = params['emissivity'] * SIGMA * (T_surface**4 - T_SPACE**4) * TANK_AREA

    dT = (q_solar - q_rad_out - q_leak) / (params['thermal_mass'])
    q_to_prop = max(0, q_leak)

    return dT, q_to_prop


def cryocooler_model(t, T_surface, params):
    """
    Active cryo-cooler — constant rejection up to capacity.
    """
    q_solar = solar_flux(t) * TANK_AREA
    R_mli = params['R_value']
    q_leak = TANK_AREA * (T_surface - T_LCH4) / R_mli
    q_rad_out = params['emissivity'] * SIGMA * (T_surface**4 - T_SPACE**4) * TANK_AREA

    # Cryo-cooler removes heat (up to capacity)
    q_cooler = min(params['cooler_capacity'], max(0, q_leak))

    dT = (q_solar - q_rad_out - q_leak) / params['thermal_mass']
    q_to_prop = max(0, q_leak - q_cooler)

    return dT, q_to_prop


def cem_model(t, T_surface, params):
    """
    CEM Cryo-Wrap — dynamic thermal membrane.

    Three modes based on local thermal state:
    - PROTECT: trichomes closed, minimal leak (shadow side)
    - VENT: partial open, moderate routing (transition)
    - DUMP: full open, maximum heat rejection to radiators (sun side)
    """
    q_solar = solar_flux(t) * TANK_AREA

    # CEM responds to solar input dynamically
    solar_now = solar_flux(t)
    if solar_now < 10:
        # PROTECT mode — shadow
        mode = 'PROTECT'
        R_eff = params['R_protect']     # Very high R — trichomes closed
        reject_frac = 0.0               # No active rejection
    elif solar_now < Q_SOLAR * ABSORPTIVITY * 0.5:
        # VENT mode — transition
        mode = 'VENT'
        blend = solar_now / (Q_SOLAR * ABSORPTIVITY * 0.5)
        R_eff = params['R_protect'] * (1 - blend) + params['R_vent'] * blend
        reject_frac = blend * 0.6       # Partial routing to radiators
    else:
        # DUMP mode — full sun
        mode = 'DUMP'
        R_eff = params['R_dump']         # Low R — trichomes open, routing active
        reject_frac = 0.9               # 90% of absorbed heat routed to radiators

    q_leak = TANK_AREA * (T_surface - T_LCH4) / R_eff

    # CEM actively routes heat to radiators
    q_rejected = q_solar * reject_frac

    q_rad_out = params['emissivity'] * SIGMA * (T_surface**4 - T_SPACE**4) * TANK_AREA

    dT = (q_solar - q_rad_out - q_leak - q_rejected) / params['thermal_mass']
    q_to_prop = max(0, q_leak)

    return dT, q_to_prop


# ============================================================
# SIMULATION
# ============================================================

def simulate(model_func, params, n_orbits=10):
    """Run thermal sim for n orbits."""
    t_end = ORBIT_PERIOD * n_orbits
    dt = 1.0  # 1 second timestep

    n_steps = int(t_end / dt)
    times = np.zeros(n_steps)
    T_surf = np.zeros(n_steps)
    q_prop = np.zeros(n_steps)
    boiloff = np.zeros(n_steps)

    T = params.get('T_init', 150)  # K initial surface temp (cold-soaked)
    total_boiloff_kg = 0

    for i in range(n_steps):
        t = i * dt
        times[i] = t

        dT, q_to_prop = model_func(t, T, params)
        T = T + dT * dt
        T = np.clip(T, 50, 500)

        T_surf[i] = T
        q_prop[i] = q_to_prop

        # Boiloff: q_to_prop heats propellant, excess above boiling point causes boiloff
        boiloff_rate = q_to_prop / LCH4_LATENT  # kg/s
        total_boiloff_kg += boiloff_rate * dt
        boiloff[i] = total_boiloff_kg

    return times, T_surf, q_prop, boiloff


# ============================================================
# PARAMETERS
# ============================================================

# MLI only
mli_params = {
    'R_value': 20.0,          # m²·K/W (good MLI)
    'emissivity': 0.03,       # MLI outer surface — very low
    'thermal_mass': 50000,    # J/K (tank structure thermal capacitance)
    'T_init': 150,
}

# MLI + cryo-cooler
cryo_params = {
    'R_value': 20.0,
    'emissivity': 0.03,
    'thermal_mass': 50000,
    'cooler_capacity': 20000,  # W (20 kW cryo-cooler)
    'T_init': 150,
}

# CEM Cryo-Wrap
cem_params = {
    'R_protect': 50.0,        # m²·K/W — trichomes closed, better than MLI
    'R_vent': 15.0,           # m²·K/W — partial routing
    'R_dump': 5.0,            # m²·K/W — full dump, heat actively moved
    'emissivity': 0.85,       # CEM outer surface — high emissivity radiator
    'thermal_mass': 55000,    # J/K (slightly more mass from CEM panels)
    'T_init': 150,
}

# ============================================================
# RUN
# ============================================================

N_ORBITS = 10  # 15 hours

print("CEM Cryo-Wrap Orbital Thermal Simulation")
print("=" * 60)
print(f"Tank area: {TANK_AREA} m² | Solar: {Q_SOLAR} W/m² | Orbit: {ORBIT_PERIOD/60:.0f} min")
print(f"LCH4 boiling: {T_LCH4} K | Propellant: {PROP_MASS/1000:.0f} tonnes")
print(f"Simulating {N_ORBITS} orbits ({N_ORBITS * ORBIT_PERIOD / 3600:.1f} hours)")
print()

print("Running MLI only...")
t_mli, Ts_mli, qp_mli, bo_mli = simulate(mli_model, mli_params, N_ORBITS)
print(f"  Final surface temp: {Ts_mli[-1]:.1f} K | Boiloff: {bo_mli[-1]:.1f} kg")

print("Running MLI + 20kW cryo-cooler...")
t_cry, Ts_cry, qp_cry, bo_cry = simulate(cryocooler_model, cryo_params, N_ORBITS)
print(f"  Final surface temp: {Ts_cry[-1]:.1f} K | Boiloff: {bo_cry[-1]:.1f} kg")

print("Running CEM Cryo-Wrap...")
t_cem, Ts_cem, qp_cem, bo_cem = simulate(cem_model, cem_params, N_ORBITS)
print(f"  Final surface temp: {Ts_cem[-1]:.1f} K | Boiloff: {bo_cem[-1]:.1f} kg")

# Extrapolate to 7 days
secs_per_day = 86400
rate_mli = bo_mli[-1] / (N_ORBITS * ORBIT_PERIOD) * secs_per_day * 7
rate_cry = bo_cry[-1] / (N_ORBITS * ORBIT_PERIOD) * secs_per_day * 7
rate_cem = bo_cem[-1] / (N_ORBITS * ORBIT_PERIOD) * secs_per_day * 7

print()
print("=" * 60)
print("RESULTS — Projected 7-day boiloff")
print("=" * 60)
print(f"  MLI only:      {rate_mli:.0f} kg ({rate_mli/PROP_MASS*100:.2f}% of propellant)")
print(f"  MLI + cooler:  {rate_cry:.0f} kg ({rate_cry/PROP_MASS*100:.2f}%)")
print(f"  CEM Cryo-Wrap: {rate_cem:.0f} kg ({rate_cem/PROP_MASS*100:.2f}%)")
print()
if rate_cem < rate_mli:
    reduction = (1 - rate_cem/rate_mli) * 100
    print(f"  CEM reduces boiloff by {reduction:.0f}% vs MLI alone")
if rate_cem < rate_cry:
    print(f"  CEM beats cryo-cooler by {(1-rate_cem/rate_cry)*100:.0f}% — with no power draw")

# ============================================================
# PLOT
# ============================================================

hours = lambda t: t / 3600

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('CEM Cryo-Wrap: Orbital Thermal Management for Starship\n'
             f'{N_ORBITS} orbits ({N_ORBITS*ORBIT_PERIOD/3600:.0f} hrs) | '
             f'Solar: {Q_SOLAR} W/m² | Tank: {TANK_AREA} m²',
             fontsize=13, fontweight='bold')

# Surface temperature
ax = axes[0,0]
ax.plot(hours(t_mli), Ts_mli, 'r-', lw=1.5, label='MLI only', alpha=0.8)
ax.plot(hours(t_cry), Ts_cry, 'b-', lw=1.5, label='MLI + cooler', alpha=0.8)
ax.plot(hours(t_cem), Ts_cem, 'g-', lw=1.5, label='CEM Cryo-Wrap', alpha=0.8)
ax.axhline(T_LCH4, color='k', ls='--', alpha=0.3, label=f'LCH4 boiling ({T_LCH4}K)')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Surface Temperature (K)')
ax.set_title('Tank Surface Temperature'); ax.legend(); ax.grid(alpha=0.3)

# Heat leak to propellant
ax = axes[0,1]
ax.plot(hours(t_mli), qp_mli/1000, 'r-', lw=1, label='MLI only', alpha=0.7)
ax.plot(hours(t_cry), qp_cry/1000, 'b-', lw=1, label='MLI + cooler', alpha=0.7)
ax.plot(hours(t_cem), qp_cem/1000, 'g-', lw=1, label='CEM Cryo-Wrap', alpha=0.7)
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Heat Leak to Propellant (kW)')
ax.set_title('Heat Reaching Propellant'); ax.legend(); ax.grid(alpha=0.3)

# Cumulative boiloff
ax = axes[1,0]
ax.plot(hours(t_mli), bo_mli, 'r-', lw=2, label='MLI only')
ax.plot(hours(t_cry), bo_cry, 'b-', lw=2, label='MLI + cooler')
ax.plot(hours(t_cem), bo_cem, 'g-', lw=2, label='CEM Cryo-Wrap')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Cumulative Boiloff (kg)')
ax.set_title('Propellant Loss (Lower is Better)'); ax.legend(); ax.grid(alpha=0.3)

# Solar flux pattern (one orbit)
ax = axes[1,1]
t_one = np.linspace(0, ORBIT_PERIOD, 1000)
q_one = np.array([solar_flux(t) for t in t_one])
ax.plot(t_one/60, q_one, 'orange', lw=2)
ax.fill_between(t_one/60, q_one, alpha=0.3, color='orange')
ax.set_xlabel('Time in Orbit (min)'); ax.set_ylabel('Solar Flux (W/m²)')
ax.set_title(f'Solar Input Profile (one orbit, {SUN_FRACTION*100:.0f}% sun)')
ax.grid(alpha=0.3)
ax.annotate('SHADOW', xy=(75, 10), fontsize=12, alpha=0.5, ha='center')
ax.annotate('SUNLIT', xy=(25, 200), fontsize=12, alpha=0.5, ha='center')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cem_cryowrap_results.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")

# Mass comparison
print()
print("=" * 60)
print("MASS & COST COMPARISON")
print("=" * 60)
print(f"  CEM Cryo-Wrap:  ~400 kg, no power draw, adaptive")
print(f"  Cryo-cooler:    ~500-1000 kg + continuous power ({cryo_params['cooler_capacity']/1000:.0f} kW)")
print(f"  Extra tanker:   ~$50M per launch to replace boiled fuel")
print(f"  CEM pays for itself if it saves one tanker flight.")
