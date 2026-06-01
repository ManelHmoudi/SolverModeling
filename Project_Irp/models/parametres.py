"""
Load instance data from JSON and build sets_ / params_ dicts for the IRP model.
"""

import json
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "instance_15_clients.json")

with open(DATA_PATH, "r") as f:
    data = json.load(f)

sets_raw   = data["sets"]
params_raw = data["parameters"]


# =============================================================================
# SETS
# =============================================================================

N       = sets_raw["N"]
clients = sets_raw["clients"]
O       = sets_raw["O"]
T       = sets_raw["T"]
M       = sets_raw["M"]
A       = [(i, j) for i in N for j in N if i != j]

sets_ = {
    "N":       N,
    "A":       A,
    "T":       T,
    "M":       M,
    "O":       O,
    "clients": clients,
}


# =============================================================================
# PARAMETERS
# =============================================================================

# ── Client demand parameters ──────────────────────────────────────────────────
# q_lt[l,t]         : quantity demanded by client l in period t
# K_lt[l,t]         : vehicle(s) assigned to client l in period t
# requires_cold[l,t]: vehicle type(s) that must serve client l in period t
q_lt = {
    (int(k.split(",")[0]), int(k.split(",")[1])): v
    for k, v in params_raw["q_lt"].items()
}

K_lt = {
    (int(k.split(",")[0]), int(k.split(",")[1])): v
    for k, v in params_raw["K_lt"].items()
}

requires_cold = {
    (int(k.split(",")[0]), int(k.split(",")[1])): v
    for k, v in params_raw["requires_cold"].items()
}

# ── Vehicle parameters ────────────────────────────────────────────────────────
# v[k]         : speed of vehicle k (km/h)
# v_ms[k]      : speed of vehicle k (m/s)
# v2[k]        : squared speed (m/s)² — used in CMEM aerodynamic drag term
# Q[k]         : maximum load capacity of vehicle k (units)
# frigo_trucks : set of refrigerated vehicle ids
v            = {int(k): val for k, val in params_raw["v"].items()}
v_ms         = {k: v[k] / 3.6 for k in M}
v2           = {k: v_ms[k] ** 2 for k in M}
Q            = {int(k): val for k, val in params_raw["Q"].items()}
frigo_trucks = set(params_raw["frigo_trucks"])

# ── Network distance and routing cost parameters ──────────────────────────────
# d[i,j]      : distance between nodes i and j (km)
# d_m[i,j]    : distance between nodes i and j (metres) — for CMEM formula
# c_route[i,j]: base per-unit routing cost on arc (i,j)
d       = {(i, j): abs(i - j) * 10 for (i, j) in A}
d_m     = {(i, j): d[i, j] * 1000  for (i, j) in A}
c_route = {(i, j): params_raw["c_route_value"] for (i, j) in A}

# ── Refrigeration and holding cost parameters ─────────────────────────────────
# p5      : refrigeration surcharge per unit distance for cold-chain vehicles
# e_stock : energy consumption per unit stored per period
# alpha_r : carbon cost coefficient for refrigeration energy
# h_O     : total holding cost per unit per period (space + refrigeration energy)
p5      = params_raw["p5"]
e_stock = params_raw["e_stock"]
alpha_r = params_raw["alpha_r"]
h_O     = params_raw["h_O_space"] + alpha_r * e_stock

# c_ijk[i,j,k]: per-unit arc cost for vehicle k on arc (i,j)
c_ijk = {
    (i, j, k): (
        c_route[i, j] + p5 / v[k]
        if k in frigo_trucks
        else c_route[i, j]
    )
    for (i, j) in A for k in M
}

# ── Time-window and service time parameters ───────────────────────────────────
# ET[l,t]  : earliest allowed arrival time at client l in period t (hours)
# LT[l,t]  : latest allowed arrival time at client l in period t (hours)
# tau_min  : global minimum depot return time across all periods (hours)
# tau_max  : global maximum depot return time across all periods (hours)
# s[i]     : service time at node i (hours)
ET      = {(l, t): params_raw["ET_value"] for l in clients for t in T}
LT      = {(l, t): params_raw["LT_value"] for l in clients for t in T}
tau_min = params_raw["tau_min"]
tau_max = params_raw["tau_max"]
s       = {i: params_raw["s_value"] for i in N}

# ── Depot inventory parameters ────────────────────────────────────────────────
# I_O_init_{frigo,nonfrigo}  : initial depot stock level at t=0 (units)
# I_O_max_{frigo,nonfrigo}   : maximum depot storage capacity (units)
# I_O_min_{frigo,nonfrigo}   : minimum safety stock level at depot (units)
I_O_init_frigo    = params_raw["I_O_init_frigo"]
I_O_init_nonfrigo = params_raw["I_O_init_nonfrigo"]
I_O_max_frigo     = params_raw["I_O_max_frigo"]
I_O_max_nonfrigo  = params_raw["I_O_max_nonfrigo"]
I_O_min_frigo     = params_raw["I_O_min_frigo"]
I_O_min_nonfrigo  = params_raw["I_O_min_nonfrigo"]

# ── Replenishment parameters ──────────────────────────────────────────────────
# R_frigo[t]    : total quantity of refrigerated product received at depot in t
# R_nonfrigo[t] : total quantity of non-refrigerated product received at depot in t
# R[t]          : total replenishment at depot in period t
R_frigo    = {}
R_nonfrigo = {}
for t in T:
    rf, rnf = 0, 0
    for l in clients:
        truck = requires_cold[l, t][0]
        if truck in frigo_trucks:
            rf  += q_lt[l, t]
        else:
            rnf += q_lt[l, t]
    R_frigo[t]    = rf
    R_nonfrigo[t] = rnf

R = {t: R_frigo[t] + R_nonfrigo[t] for t in T}

# ── Environmental / CO2 emission parameters ───────────────────────────────────
# g              : gravitational constant (m/s²)
# Cr             : rolling resistance coefficient (dimensionless)
# Cd             : aerodynamic drag coefficient (dimensionless)
# A_f            : vehicle frontal area (m²)
# rho            : air density (kg/m³)
# a              : vehicle acceleration (m/s²)
# w              : vehicle curb weight (kg)
# alpha_co2[i,j] : rolling-resistance emission factor on arc (i,j) = g * Cr
# beta_co2       : aerodynamic drag emission factor = 0.5 * Cd * A_f * rho
# e_co2          : CO2 emission factor (kg CO2 / litre fuel)
# kg_per_unit    : weight of one product unit (kg)
# fuel_to_joules : energy content of fuel (J / litre)
g    = params_raw["g"]
Cr   = params_raw["Cr"]
Cd   = params_raw["Cd"]
A_f  = params_raw["A_f"]
rho  = params_raw["rho"]
a    = params_raw["a"]
w    = params_raw["w"]

alpha_co2      = {(i, j): g * Cr for (i, j) in A}
beta_co2       = 0.5 * Cd * A_f * rho
e_co2          = params_raw["e_co2"]
kg_per_unit    = params_raw["kg_per_unit"]
fuel_to_joules = params_raw["fuel_to_joules"]

# ── Financial / working capital parameters ────────────────────────────────────
# DSO          : days sales outstanding (customer receivables cycle, days)
# DPO          : days payable outstanding (supplier payables cycle, days)
# DIO          : days inventory outstanding (inventory turnover cycle, days)
# P_sale[l]    : unit selling price for client l
# P_purchase[k]: unit purchase price for vehicle k's product type
DSO        = params_raw["DSO"]
DPO        = params_raw["DPO"]
DIO        = params_raw["DIO"]
P_sale     = {int(k): val for k, val in params_raw["P_sale"].items()}
P_purchase = {int(k): val for k, val in params_raw["P_purchase"].items()}

# ── Objective bounds and solver constants ─────────────────────────────────────
# c1    : penalty weight for early arrival at client (soft time window)
# c2    : penalty weight for late arrival at client (soft time window)
# C_max : upper bound on total logistics cost (f1 budget constraint)
# E_max : upper bound on total CO2 emissions (f2 budget constraint)
# T_max : upper bound on total travel time (f3 budget constraint)
# B     : upper bound on working capital requirement / BFR (f4 budget constraint)
# BIG_M : big-M constant for linearising indicator constraints
c1    = params_raw["c1"]
c2    = params_raw["c2"]
C_max = params_raw["C_max"]
E_max = params_raw["E_max"]
T_max = params_raw["T_max"]
B     = params_raw["B"]
BIG_M = params_raw["BIG_M"]

params_ = {
    "q_lt":              q_lt,
    "K_lt":              K_lt,
    "requires_cold":     requires_cold,
    "v":                 v,
    "v2":                v2,
    "Q":                 Q,
    "frigo_trucks":      frigo_trucks,
    "d":                 d,
    "d_m":               d_m,
    "c_ijk":             c_ijk,
    "ET":                ET,
    "LT":                LT,
    "tau_min":           tau_min,
    "tau_max":           tau_max,
    "s":                 s,
    "I_O_init_frigo":    I_O_init_frigo,
    "I_O_init_nonfrigo": I_O_init_nonfrigo,
    "I_O_max_frigo":     I_O_max_frigo,
    "I_O_max_nonfrigo":  I_O_max_nonfrigo,
    "I_O_min_frigo":     I_O_min_frigo,
    "I_O_min_nonfrigo":  I_O_min_nonfrigo,
    "h_O":               h_O,
    "R":                 R,
    "R_frigo":           R_frigo,
    "R_nonfrigo":        R_nonfrigo,
    "alpha_co2":         alpha_co2,
    "beta_co2":          beta_co2,
    "w":                 w,
    "kg_per_unit":       kg_per_unit,
    "e_co2":             e_co2,
    "fuel_to_joules":    fuel_to_joules,
    "P_sale":            P_sale,
    "P_purchase":        P_purchase,
    "DIO":               DIO,
    "DSO":               DSO,
    "DPO":               DPO,
    "c1":                c1,
    "c2":                c2,
    "C_max":             C_max,
    "E_max":             E_max,
    "T_max":             T_max,
    "B":                 B,
    "BIG_M":             BIG_M,
}
