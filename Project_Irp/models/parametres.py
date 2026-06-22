"""Load instance data from JSON and build sets_ / params_ dicts for the IRP model."""

import json
import math
import os

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

    I_O_init_frigo    = params_raw["I_O_init_frigo"]
    I_O_init_nonfrigo = params_raw["I_O_init_nonfrigo"]
    I_O_max_frigo     = params_raw["I_O_max_frigo"]
    I_O_max_nonfrigo  = params_raw["I_O_max_nonfrigo"]
    I_O_min_frigo     = params_raw["I_O_min_frigo"]
    I_O_min_nonfrigo  = params_raw["I_O_min_nonfrigo"]

    R_frigo    = {t: sum(q_lt[l, t] for l in clients if     requires_cold[l, t]) for t in T}
    R_nonfrigo = {t: sum(q_lt[l, t] for l in clients if not requires_cold[l, t]) for t in T}
    R          = {t: R_frigo[t] + R_nonfrigo[t] for t in T}

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
    }

    return sets_, params_
