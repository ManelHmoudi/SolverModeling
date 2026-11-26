"""Load an instance JSON into the sets_ / params_ dicts of the IRP model.

Strict loader: every parameter listed in REQUIRED_KEYS must be present in
the JSON. There are no silent defaults; a missing key raises a KeyError that
names it. Units are given in docs/model_formulation.md §4.
"""

import json
import math
import os

BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "instance_15_clients.json")

PRODUCT_CLASSES = ("frigo", "nonfrigo")

REQUIRED_KEYS = (
    # demand and fleet
    "q_lt", "cold", "v", "Q", "frigo_trucks",
    # time
    "s_value", "tau_max", "ET", "LT", "c1", "c2",
    # transport cost
    "F_veh", "c_km", "p5", "c_load",
    # depot inventory and supply
    "I_O_init_frigo", "I_O_init_nonfrigo", "I_O_min_frigo", "I_O_min_nonfrigo",
    "I_O_max_frigo", "I_O_max_nonfrigo", "h_cold", "h_ambient",
    "F_order",
    # finance
    "P_sale", "P_purchase", "L_S", "L_P",
    # CMEM
    "g", "Cr", "Cd", "A_f", "rho", "a", "w", "kg_per_unit", "e_co2",
    "fuel_to_joules",
)


def _pair(key):
    l, t = key.split(",")
    return int(l), int(t)


def load_instance(data_path=None):
    with open(data_path or DEFAULT_DATA_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return instance_from_dict(data, data_path)


def instance_from_dict(data, data_path="<dict>"):
    """Same as load_instance, from an already-parsed instance dict."""
    sets_raw, raw = data["sets"], data["parameters"]

    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise KeyError(f"{data_path}: missing parameters {missing} "
                       f"(run: python -m scripts.generate_synthetic_instances --migrate)")

    N, clients, O = sets_raw["N"], sets_raw["clients"], sets_raw["O"]
    T, M = sets_raw["T"], sets_raw["M"]
    A = [(i, j) for i in N for j in N if i != j]
    sets_ = {"N": N, "A": A, "T": T, "M": M, "O": O, "clients": clients}

    coords = {int(k): tuple(v) for k, v in sets_raw["coordinates"].items()}
    d = {(i, j): math.dist(coords[i], coords[j]) for (i, j) in A}          # km

    spd = {int(k): float(v) for k, v in raw["v"].items()}                  # km/h
    frigo_trucks = set(raw["frigo_trucks"])
    q_lt = {_pair(k): v for k, v in raw["q_lt"].items()}
    requires_cold = {_pair(k): bool(v) for k, v in raw["cold"].items()}

    tau_max = float(raw["tau_max"])
    s = {i: float(raw["s_value"]) for i in N}       # s[O] = loading time at the depot

    params_ = {
        # demand, fleet, geography
        "q_lt":          q_lt,
        "requires_cold": requires_cold,
        "v":             spd,
        "v_ms":          {k: spd[k] / 3.6 for k in M},                     # m/s
        "Q":             {int(k): v for k, v in raw["Q"].items()},
        "frigo_trucks":  frigo_trucks,
        "d":             d,
        "d_m":           {a: d[a] * 1000.0 for a in A},                    # m
        # time
        "s":             s,
        "tau_max":       tau_max,
        "ET":            {(l, t): float(raw["ET"][str(l)]) for l in clients for t in T},   # per-client window
        "LT":            {(l, t): float(raw["LT"][str(l)]) for l in clients for t in T},
        "c1":            float(raw["c1"]),
        "c2":            float(raw["c2"]),
        # transport cost: €/route, €/vehicle-km (+ refrigeration €/h ÷ km/h), €/(unit·km)
        "F_veh":         float(raw["F_veh"]),
        "c_km_k":        {k: raw["c_km"] + (raw["p5"] / spd[k] if k in frigo_trucks else 0.0) for k in M},
        "c_load":        float(raw["c_load"]),
        # depot inventory (units) and holding (€/unit/period)
        "h_frigo":       float(raw["h_cold"]),
        "h_nonfrigo":    float(raw["h_ambient"]),
        "F_order":       float(raw["F_order"]),
        # finance
        "P_sale":        {l: float(raw["P_sale"][str(l)]) for l in clients},
        "P_purchase":    float(raw["P_purchase"]),
        "L_S":           int(raw["L_S"]),
        "L_P":           int(raw["L_P"]),
        # linearised CMEM (Bektas & Laporte 2011)
        "alpha_co2":     raw["a"] + raw["g"] * raw["Cr"],                    # m/s²
        "beta_co2":      0.5 * raw["Cd"] * raw["A_f"] * raw["rho"],          # kg/m
        "w":             float(raw["w"]),                                    # kg
        "kg_per_unit":   float(raw["kg_per_unit"]),
        "e_co2":         float(raw["e_co2"]),                                # kg CO2 / L
        "fuel_to_joules": float(raw["fuel_to_joules"]),                      # J / L
    }
    for p in PRODUCT_CLASSES:
        for key in ("I_O_init", "I_O_min", "I_O_max"):
            params_[f"{key}_{p}"] = float(raw[f"{key}_{p}"])
        # Largest useful order: refill from I_min to I_max and cover one period
        # of shipments on the class's fleet. Not a data parameter -- a valid
        # big-M for r <= M·z (C8).
        fleet_cap = sum(params_["Q"][k] for k in M if (k in frigo_trucks) == (p == "frigo"))
        params_[f"R_bigM_{p}"] = params_[f"I_O_max_{p}"] - params_[f"I_O_min_{p}"] + fleet_cap
    return sets_, params_
