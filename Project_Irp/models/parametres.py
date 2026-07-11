"""Load instance data from JSON and build sets_ / params_ dicts for the IRP model."""

import json
import math
import os
import warnings

BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "instance_15_clients.json")


def load_instance(data_path=None):
    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    with open(data_path, "r") as f:
        data = json.load(f)

    sets_raw   = data["sets"]
    params_raw = data["parameters"]

    N       = sets_raw["N"]
    clients = sets_raw["clients"]
    O       = sets_raw["O"]
    T       = sets_raw["T"]
    M       = sets_raw["M"]
    A       = [(i, j) for i in N for j in N if i != j]

    sets_ = {"N": N, "A": A, "T": T, "M": M, "O": O, "clients": clients}

    q_lt = {
        (int(k.split(",")[0]), int(k.split(",")[1])): v
        for k, v in params_raw["q_lt"].items()
    }

    requires_cold = {
        (int(k.split(",")[0]), int(k.split(",")[1])): v[0] in params_raw["frigo_trucks"]
        for k, v in params_raw["requires_cold"].items()
    }

    non_frigo_trucks = [k for k in M if k not in params_raw["frigo_trucks"]]
    K_lt = {
        (l, t): list(params_raw["frigo_trucks"]) if requires_cold[l, t] else non_frigo_trucks
        for (l, t) in requires_cold
    }

    spd          = {int(k): val for k, val in params_raw["v"].items()}
    v_ms         = {k: spd[k] / 3.6 for k in M}
    v2           = {k: v_ms[k] ** 2 for k in M}
    Q            = {int(k): val for k, val in params_raw["Q"].items()}
    frigo_trucks = set(params_raw["frigo_trucks"])

    if "coordinates" in sets_raw:
        coords = {int(k): tuple(v) for k, v in sets_raw["coordinates"].items()}
        d = {
            (i, j): math.sqrt((coords[i][0] - coords[j][0]) ** 2
                               + (coords[i][1] - coords[j][1]) ** 2)
            for (i, j) in A
        }
    else:
        d = {(i, j): abs(i - j) * 10 for (i, j) in A}

    d_m     = {(i, j): d[i, j] * 1000 for (i, j) in A}
    c_route = {(i, j): params_raw["c_route_value"] for (i, j) in A}

    p5      = params_raw["p5"]
    e_stock = params_raw["e_stock"]
    alpha_r = params_raw["alpha_r"]
    h_O     = params_raw["h_O_space"] + alpha_r * e_stock

    c_ijk = {
        (i, j, k): (
            c_route[i, j] + p5 / spd[k] if k in frigo_trucks else c_route[i, j]
        )
        for (i, j) in A for k in M
    }

    ET      = {(l, t): params_raw["ET_value"] for l in clients for t in T}
    LT      = {(l, t): params_raw["LT_value"] for l in clients for t in T}
    tau_min = params_raw["tau_min"]
    tau_max = params_raw["tau_max"]
    s       = {i: params_raw["s_value"] for i in N}

    # Demand replenishment — computed before inventory params so defaults can reference them.
    R_frigo    = {t: sum(q_lt[l, t] for l in clients if     requires_cold[l, t]) for t in T}
    R_nonfrigo = {t: sum(q_lt[l, t] for l in clients if not requires_cold[l, t]) for t in T}
    R          = {t: R_frigo[t] + R_nonfrigo[t] for t in T}

    # Inventory bounds — fall back to demand-derived defaults so new instances
    # without these JSON keys still work correctly. A missing/typo'd key warns
    # (same policy as the P_sale backfill below) so a silent substitution is
    # never mistaken for an intentionally-configured value.
    def _get_or_warn(key, default):
        if key in params_raw:
            return params_raw[key]
        warnings.warn(
            f"[parametres] '{key}' missing from instance JSON — using computed "
            f"default {default}. Add it to the JSON to remove this warning.",
            RuntimeWarning, stacklevel=3,
        )
        return default

    _n_T         = max(len(T), 1)
    _avg_R_f     = sum(R_frigo.values())    / _n_T
    _avg_R_nf    = sum(R_nonfrigo.values()) / _n_T

    I_O_init_frigo    = _get_or_warn("I_O_init_frigo",    round(_avg_R_f  * 2.0))
    I_O_init_nonfrigo = _get_or_warn("I_O_init_nonfrigo", round(_avg_R_nf * 2.0))
    I_O_max_frigo     = _get_or_warn("I_O_max_frigo",     round(_avg_R_f  * 4.0))
    I_O_max_nonfrigo  = _get_or_warn("I_O_max_nonfrigo",  round(_avg_R_nf * 4.0))
    I_O_min_frigo     = _get_or_warn("I_O_min_frigo",     (max(1, round(_avg_R_f  * 0.25)) if _avg_R_f  > 0 else 0))
    I_O_min_nonfrigo  = _get_or_warn("I_O_min_nonfrigo",  (max(1, round(_avg_R_nf * 0.25)) if _avg_R_nf > 0 else 0))

    g    = params_raw["g"]
    Cr   = params_raw["Cr"]
    Cd   = params_raw["Cd"]
    A_f  = params_raw["A_f"]
    rho  = params_raw["rho"]
    w    = params_raw["w"]

    alpha_co2      = {(i, j): g * Cr for (i, j) in A}
    beta_co2       = 0.5 * Cd * A_f * rho
    e_co2          = params_raw["e_co2"]
    kg_per_unit    = params_raw["kg_per_unit"]
    fuel_to_joules = params_raw["fuel_to_joules"]

    DSO        = params_raw["DSO"]
    DPO        = params_raw["DPO"]
    DIO        = params_raw["DIO"]
    P_sale     = {int(k): val for k, val in params_raw["P_sale"].items()}
    P_purchase = {int(k): val for k, val in params_raw["P_purchase"].items()}

    # Guarantee every client has a P_sale entry so evaluator.py never crashes on KeyError.
    # Missing entries get the average sale price with a printed warning.
    _missing_clients = [l for l in clients if l not in P_sale]
    if _missing_clients:
        _avg_price = sum(P_sale.values()) / len(P_sale) if P_sale else 1.0
        warnings.warn(
            f"[parametres] P_sale missing for clients {_missing_clients}. "
            f"Using average price {_avg_price:.4f}. Add them to the JSON to remove this warning.",
            RuntimeWarning, stacklevel=2,
        )
        for _l in _missing_clients:
            P_sale[_l] = _avg_price

    c1    = params_raw["c1"]
    c2    = params_raw["c2"]
    C_max = _get_or_warn("C_max", 99999)
    E_max = _get_or_warn("E_max", 99999)
    T_max = _get_or_warn("T_max", 99999)
    B     = _get_or_warn("B",     99999)

    # BIG_M: the minimum valid value is tau_max + max_arc_time + service_time.
    # Auto-compute from network geometry so new instances never need to set it manually.
    # If the JSON supplies a larger value, use that (never go below the valid minimum).
    _s_val     = params_raw["s_value"]
    _max_d     = max(d.values()) if d else 0.0
    _min_v     = min(spd.values()) if spd else 1.0
    _auto_BIG_M = tau_max + _s_val + _max_d / _min_v + 1.0
    _json_BIG_M = params_raw.get("BIG_M")
    BIG_M = max(float(_json_BIG_M), _auto_BIG_M) if _json_BIG_M is not None else _auto_BIG_M

    # Non-mandatory deliveries below this threshold are skipped in the decoder.
    # Configurable per instance; defaults to 5 (empirically suitable for typical demand scale).
    min_delivery_threshold = int(params_raw.get("min_delivery_threshold", 5))

    params_ = {
        "q_lt":              q_lt,
        "K_lt":              K_lt,
        "requires_cold":     requires_cold,
        "v":                 spd,
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
        "min_delivery_threshold": min_delivery_threshold,
    }

    return sets_, params_
