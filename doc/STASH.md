# utils_cmip7 — STASH Reference

This document records **STASH code mappings**, conventions, and interpretation
rules used by `utils_cmip7`.

This file is **descriptive**.  
Architectural constraints live in `CLAUDE.md`.

---

## Purpose

- Provide a single authoritative reference for STASH usage
- Document how STASH codes are interpreted across PP and NetCDF outputs
- Avoid duplication and silent mismatches during extraction

---

## STASH Mapping Functions

### `stash(var_name)`
Maps short variable names to PP-format STASH strings.

Example:
'gpp' → 'm01s03i261'
### `stash_nc(var_name)`
Maps short variable names to **numeric NetCDF STASH codes**.

Example:
'gpp' → 3261
Both functions cover:
- Carbon-cycle variables
- Vegetation and soil pools
- Atmospheric state variables
- Ocean CO₂ flux

---

## Common Carbon-Cycle STASH Codes

| Variable | STASH (PP) | STASH (NC) | Description |
|--------|------------|------------|-------------|
| GPP | m01s03i261 | 3261 | Gross Primary Production |
| NPP | m01s03i262 | 3262 | Net Primary Production |
| Rh | m01s03i293 | 3293 | Soil respiration |
| CV | m01s19i002 | 19002 | Vegetation carbon |
| CS | m01s19i016 | 19016 | Soil carbon |
| fracPFTs | m01s19i013 | 19013 | PFT fractions |
| fgco2 | m02s30i249 | 30249 | Ocean CO₂ flux |
| tas | m01s03i236 | 3236 | Surface air temperature |
| pr | m01s05i216 | 5216 | Precipitation |

---

## NetCDF `stash_code` Suffix (`s`)

When inspecting NetCDF headers with `ncdump`, STASH codes may appear as:

stash_code = 3261s ;
### Important clarification

- The trailing `s` is a **CDL type suffix** (short integer)
- It is **not part of the STASH value**
- When loaded via `iris` or `netCDF4`, the value is a plain integer

Example:
3261s → int(3261)
No special handling is required.

---

## Variable Conversion Logic

Unit conversions are applied via the internal conversion dictionary.

Examples:

| Variable type | Conversion |
|--------------|------------|
| Flux (kgC m⁻² s⁻¹) | `× 3600 × 24 × 360 × 1e-12 → PgC yr⁻¹` |
| Stock (kgC m⁻²) | `× 1e-12 → PgC` |
| CO₂ flux (kgCO₂) | `(12/44) × … → PgC` |
| Atmospheric CO₂ | `mmr → ppmv` |

Conversions are applied automatically during extraction.

---

## Design Notes

- STASH mappings are **centralised**
- New variables must be added in one place only
- Ambiguous mappings must raise errors, not warnings

---

## Scope Limitations

- STASH handling assumes UM conventions
- Non-UM datasets must supply compatible metadata or custom mappings

