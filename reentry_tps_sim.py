"""
Starship Reentry TPS Simulation — Ghost Under-Skin v0
======================================================
Compares baseline tile stack vs Ghost Shell retrofit.

Key insight: Insulation soak time is ~900s. Standard reentry peak heating
is 300-600s. The insulation handles SHORT exposures fine. The Ghost Shell
retrofit matters for:
1. Degraded insulation (gap torching erodes insulation over flights)
2. Extended heating (Mars EDL is longer than LEO reentry)
3. Repeated flights (cumulative damage model)

This sim tests DEGRADED insulation scenarios — the real failure mode.

Author: Annie Robinson (Forge/Claude Code)
Origin: Harley Robinson, Seeds/Concepts/Starship Reentry.txt
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# MATERIALS
# ============================================================

tile =       {'name': 'Ceramic Tile',      'k': 1.5,  'rho': 2200, 'cp': 900,  'L': 0.025}
insulation = {'name': 'Insulation',         'k': 0.05, 'rho': 200,  'cp': 1000, 'L': 0.015}
damaged_ins ={'name': 'Damaged Insulation', 'k': 0.15, 'rho': 180,  'cp': 900,  'L': 0.010}  # thinner, degraded
hull =       {'name': 'Steel Hull',         'k': 16.0, 'rho': 8000, 'cp': 500,  'L': 0.004}
gap_insert = {'name': 'Gap Insert',         'k': 0.8,  'rho': 1800, 'cp': 800,  'L': 0.003}
under_mesh = {'name': 'Under-Mesh (SiC)',   'k': 40.0, 'rho': 2500, 'cp': 700,  'L': 0.003}
rib =        {'name': 'Carbon Rib',         'k': 80.0, 'rho': 1800, 'cp': 750,  'L': 0.005}

# ============================================================
# STACKS
# ============================================================

# Fresh ship
baseline_intact = [tile, insulation, hull]
ghost_intact    = [tile, gap_insert, under_mesh, rib, insulation, hull]

# Tile missing, fresh insulation
baseline_missing = [insulation, hull]
ghost_missing    = [under_mesh, rib, insulation, hull]

# Tile missing, DEGRADED insulation (after multiple flights with gap torching)
baseline_degraded = [damaged_ins, hull]
ghost_degraded    = [under_mesh, rib, damaged_ins, hull]

# Worst case: tile missing, insulation burned through (direct to hull)
baseline_worst = [hull]
ghost_worst    = [under_mesh, rib, hull]

# ============================================================
# SOLVER
# ============================================================

def run_sim(layers, q_flux, T0, t_end):
    dx = 0.0004  # 0.4mm nodes

    k_arr, rho_arr, cp_arr = [], [], []
    x_pos = 0.0
    for layer in layers:
        n = max(3, round(layer['L'] / dx))
        ldx = layer['L'] / n
        for i in range(n):
            k_arr.append(layer['k'])
            rho_arr.append(layer['rho'])
            cp_arr.append(layer['cp'])
        x_pos += layer['L']

    k_arr = np.array(k_arr, dtype=np.float64)
    rho_arr = np.array(rho_arr, dtype=np.float64)
    cp_arr = np.array(cp_arr, dtype=np.float64)
    N = len(k_arr)
    dx_n = x_pos / N

    alpha = k_arr / (rho_arr * cp_arr)
    dt = 0.25 * dx_n**2 / np.max(alpha)

    n_steps = int(t_end / dt) + 1
    save_every = max(1, n_steps // 2000)

    T = np.full(N, T0)
    times, hull_temps, surf_temps = [], [], []
    sigma = 5.67e-8

    for step in range(n_steps):
        T_new = T.copy()

        # Interior
        for i in range(1, N-1):
            kl = 2*k_arr[i-1]*k_arr[i]/(k_arr[i-1]+k_arr[i])
            kr = 2*k_arr[i]*k_arr[i+1]/(k_arr[i]+k_arr[i+1])
            T_new[i] = T[i] + dt * (kl*(T[i-1]-T[i]) + kr*(T[i+1]-T[i])) / (rho_arr[i]*cp_arr[i]*dx_n**2)

        # Surface: flux in - radiation out + conduction to next node
        q_rad = 0.85 * sigma * T[0]**4
        kr = 2*k_arr[0]*k_arr[1]/(k_arr[0]+k_arr[1])
        T_new[0] = T[0] + dt * ((q_flux - q_rad)/dx_n + kr*(T[1]-T[0])/dx_n**2) / (rho_arr[0]*cp_arr[0])

        # Back face: conduction in - radiation out
        kl = 2*k_arr[-2]*k_arr[-1]/(k_arr[-2]+k_arr[-1])
        q_back = 0.3 * sigma * T[-1]**4
        T_new[-1] = T[-1] + dt * (kl*(T[-2]-T[-1])/dx_n**2 - q_back/dx_n) / (rho_arr[-1]*cp_arr[-1])

        T = np.clip(T_new, 100, 6000)

        if step % save_every == 0:
            times.append(step * dt)
            hull_temps.append(float(T[-1]))
            surf_temps.append(float(T[0]))

    return np.array(times), np.array(hull_temps), np.array(surf_temps)

# ============================================================
# RUN ALL SCENARIOS
# ============================================================

Q = 200_000   # W/m²
T0 = 300      # K
T_CRIT = 1200 # K — 304L steel loses 75% yield strength here

print("Starship Reentry TPS — Ghost Shell Applied")
print("=" * 60)
print(f"Heat flux: {Q/1000:.0f} kW/m² | Init: {T0}K | Steel critical: {T_CRIT}K")
print()

scenarios = [
    ("1. Intact tiles (300s)",     baseline_intact,   ghost_intact,   300),
    ("2. Tile missing, fresh (600s)", baseline_missing, ghost_missing, 600),
    ("3. Tile missing, degraded insulation (300s)", baseline_degraded, ghost_degraded, 300),
    ("4. Tile + insulation gone (60s)", baseline_worst, ghost_worst, 60),
]

results = {}
for name, b_layers, g_layers, t_end in scenarios:
    print(f"\n{name}")
    print("-" * 40)

    tb, hb, sb = run_sim(b_layers, Q, T0, t_end)
    tg, hg, sg = run_sim(g_layers, Q, T0, t_end)

    # Time to critical
    tcb = next((tb[i] for i,t in enumerate(hb) if t >= T_CRIT), None)
    tcg = next((tg[i] for i,t in enumerate(hg) if t >= T_CRIT), None)

    print(f"  Baseline — Hull: {hb[-1]-273:.0f}°C  Surface: {sb[-1]-273:.0f}°C  Crit: {f'{tcb:.1f}s' if tcb else 'survived'}")
    print(f"  Ghost    — Hull: {hg[-1]-273:.0f}°C  Surface: {sg[-1]-273:.0f}°C  Crit: {f'{tcg:.1f}s' if tcg else 'survived'}")

    if tcb and not tcg:
        print(f"  >>> GHOST SURVIVES. Baseline fails at {tcb:.1f}s <<<")
    elif tcb and tcg:
        print(f"  >>> Ghost buys {tcg-tcb:.1f}s extra <<<")
    elif not tcb and not tcg:
        hull_delta = hb[-1] - hg[-1]
        print(f"  >>> Both survive. Ghost is {hull_delta:.0f}K cooler <<<")

    results[name] = {
        'tb': tb, 'hb': hb, 'sb': sb,
        'tg': tg, 'hg': hg, 'sg': sg,
        'tcb': tcb, 'tcg': tcg,
    }

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Starship Reentry TPS: Baseline vs Ghost Shell Retrofit\n'
             f'Heat flux: {Q/1000:.0f} kW/m² | Steel critical: {T_CRIT-273}°C',
             fontsize=13, fontweight='bold')

C = 273.15
titles = list(results.keys())

for idx, (name, r) in enumerate(results.items()):
    ax = axes[idx//2, idx%2]
    ax.plot(r['tb'], r['hb']-C, 'r-', lw=2, label='Baseline hull')
    ax.plot(r['tg'], r['hg']-C, 'g-', lw=2, label='Ghost hull')
    ax.plot(r['tb'], r['sb']-C, 'r--', lw=1, alpha=0.5, label='Baseline surface')
    ax.plot(r['tg'], r['sg']-C, 'g--', lw=1, alpha=0.5, label='Ghost surface')
    ax.axhline(T_CRIT-C, color='k', ls='--', alpha=0.4, label=f'Steel critical {T_CRIT-273}°C')
    if r['tcb']:
        ax.axvline(r['tcb'], color='r', ls=':', alpha=0.6)
    if r['tcg']:
        ax.axvline(r['tcg'], color='g', ls=':', alpha=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reentry_tps_results.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")
