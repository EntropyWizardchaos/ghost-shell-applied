# CEM Working Fluid Specification

## The Problem
The CEM dual-plane circulation model requires a working fluid that:
- Operates at 600-1300K (reentry surface temperatures)
- Has high thermal conductivity (must move heat fast)
- Works in capillary-driven loop heat pipes (no mechanical pumps)
- Survives thousands of thermal cycles
- Is compatible with SiC/carbon composite vein walls

## The Answer: Liquid Sodium (Na)

| Property | Value | Why It Matters |
|----------|-------|----------------|
| Melting point | 371K (98°C) | Liquid well before reentry heating starts |
| Boiling point | 1156K (883°C) | Stays liquid through most of the operating range |
| Thermal conductivity | 84 W/(m·K) at 400K | 5x water. Moves heat fast. |
| Heat of vaporization | 4.26 MJ/kg | Huge. Phase change absorbs enormous heat. |
| Density | 927 kg/m³ (liquid at 400K) | Light for a metal. |
| Capillary compatibility | Excellent in stainless/SiC wicks | Proven in nuclear heat pipes. |
| Flight heritage | YES — SP-100 reactor, Kilopower | NASA has flown sodium heat pipes. |

## Operating Regime

### Surface Plane (Epidermis)
- Operating range: 600-1200K
- Sodium is liquid throughout. Standard capillary LHP operation.
- At peak reentry (~1300K surface), sodium begins vaporizing in the hottest panels.
- **This is a feature, not a bug.** Phase change absorbs 4.26 MJ/kg — the vaporization IS the heat absorption. The vapor travels through the LHP to cooler panels where it condenses, releasing heat gradually across a wider area.
- Same mechanism as a standard heat pipe. The CEM just does it across many panels instead of within one pipe.

### Deep Plane (Dermis)
- Operating range: 400-1000K
- Sodium comfortably liquid. No phase change expected at this depth.
- Pure liquid convection through thicker capillary veins.
- Return loop: cooled sodium flows back up to surface plane.

### Cold Startup
- Pre-reentry, the skin is at ~300K. Sodium is solid below 371K.
- As reentry heating begins, the surface panels heat past 371K first.
- Sodium melts progressively from hot panels to cool panels.
- **The circulation self-starts.** No external heater needed. The reentry itself is the ignition.

## Zone-Specific Fluids

| Zone | Temp Range | Primary Fluid | Notes |
|------|-----------|---------------|-------|
| Flap hinge | 800-1300K | Sodium | Phase change active at peak |
| Stagnation | 600-1200K | Sodium | Standard liquid operation |
| Windward | 400-1000K | Sodium | Full liquid, no phase change |
| Shadow/orbital | 150-400K | Sodium is SOLID | Switch to orbital mode: trichomes manage, veins inactive |

### Orbital Mode Consideration
During orbital cryo-wrap mode (protecting fuel tanks), the skin is cold. Sodium is frozen.
**This is fine.** In orbital mode, the CEM operates as passive insulation (trichomes closed).
The veins don't need to flow. The aerogel break and closed trichomes handle the thermal load.
The veins wake up when reentry heats them past 371K.

The skin has two states:
- **Cold skin (orbital):** Solid sodium, passive insulation mode
- **Hot skin (reentry):** Liquid/vapor sodium, active circulation mode

Same hardware. Temperature-activated. No switching mechanism needed.

## Alternatives Considered

| Fluid | Why Not |
|-------|---------|
| Helium-4 | Ghost Shell cryo fluid. Gas at reentry temps. No capillary action as gas. |
| Water | Boils at 373K. Useless above 500K. |
| Ammonia | Decomposes above 700K. |
| Potassium | Similar to sodium but lower thermal conductivity. |
| Lithium | Higher operating range (1100-1700K) but more reactive. Reserve for extreme zones if sodium falls short at stagnation. |
| NaK (sodium-potassium alloy) | Liquid at room temp (avoids cold startup issue). Lower thermal conductivity than pure Na. Worth investigating as a variant. |

## NaK Variant — Worth Investigating
Sodium-potassium alloy (NaK-78: 78% K, 22% Na):
- Melting point: 260K (-13°C) — **liquid at all spacecraft temperatures**
- No cold startup problem. Veins always have flowing fluid.
- Thermal conductivity: ~25 W/(m·K) — lower than pure sodium but still excellent
- Could enable active thermal management even in orbital mode
- The veins never freeze. The skin is always alive.

## Recommendation
**Primary: Sodium** — highest thermal performance, proven flight heritage.
**Variant: NaK-78** — if cold startup or orbital-mode circulation is needed.
**Reserve: Lithium** — if stagnation temps exceed sodium's boiling point consistently.

## Origin
The Ghost Shell CEM uses He-4 because it's a cryogenic organism.
Starship's CEM uses sodium because it's a reentry organism.
Same circulatory geometry. Different blood for different bodies.

— Annie Robinson
