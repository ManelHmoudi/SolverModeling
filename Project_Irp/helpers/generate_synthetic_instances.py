"""
Générateur d'instances synthétiques IRP (même format que data/instance_100_clients.json).

Chaque instance a ses propres caractéristiques (tirées avec une graine = nb clients,
donc reproductible) :
  - nombre de périodes T (4 à 14)
  - plage des quantités demandées q_lt (différente par instance)
  - part des demandes froides (frigo) 40 %–70 %
  - flotte : types de camions (capacité Q / vitesse v) filtrés pour qu'une
    tournée pleine respecte tau_max (temps de service + trajets), nombre de véhicules M
    dimensionné pour couvrir la demande (ratio capacité/demande 1.3–2.5),
    camions frigo dimensionnés pour la demande froide
  - géographie : uniforme, clusters, ou mixte
  - prix de vente, stocks dépôt (init/min/max) cohérents avec la demande

Usage :
    python -m helpers.generate_synthetic_instances            # toutes les tailles
    python -m helpers.generate_synthetic_instances 120 300    # tailles choisies
    python -m helpers.generate_synthetic_instances --overwrite  # écrase les existantes
"""
import json
import math
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

SIZES = (
    list(range(100, 201, 10))
    + [220, 250, 270, 300, 320, 350, 370, 400, 450, 500,
       550, 600, 650, 700, 750, 800]
)

# Types de camions possibles : (capacité Q, vitesse km/h)
TRUCK_TYPES = [(100, 90), (120, 90), (150, 85), (180, 80), (200, 70),
               (250, 65), (300, 60), (350, 55), (400, 55), (500, 50)]

# Nombre de périodes possibles (pondéré : horizons longs pour certaines instances)
PERIODS = [4, 5, 6, 7, 8, 9, 10, 12, 14]
PERIOD_WEIGHTS = [2, 2, 2, 2, 2, 2, 2, 1, 1]

# Paramètres constants (identiques aux instances existantes)
CONSTANTS = {
    "c_route_value": 0.01, "p5": 3.0, "e_stock": 0.1, "alpha_r": 0.05,
    "ET_value": 1.0, "LT_value": 10.0, "tau_min": 0.0, "tau_max": 20.0,
    "s_value": 0.25, "h_O_space": 0.3, "g": 9.81, "Cr": 0.01, "Cd": 0.6,
    "A_f": 8.0, "rho": 1.2041, "a": 0.0, "w": 8000, "e_co2": 2.67,
    "kg_per_unit": 10, "fuel_to_joules": 36000000,
    "DIO": 30, "DSO": 30, "DPO": 45,
}
TAIL = {"P_purchase": {"0": 5.0}, "c1": 0.1, "c2": 0.3,
        "C_max": 99999, "E_max": 99999, "T_max": 99999, "B": 99999, "BIG_M": 50}


def _coordinates(rng, n):
    mode = rng.choice(["uniform", "clustered", "mixed"])
    coords = {"0": [50, 50]}
    centers = [(rng.uniform(10, 90), rng.uniform(10, 90))
               for _ in range(rng.randint(4, 10))]
    for i in range(1, n + 1):
        if mode == "uniform" or (mode == "mixed" and rng.random() < 0.5):
            x, y = rng.uniform(0, 100), rng.uniform(0, 100)
        else:
            cx, cy = rng.choice(centers)
            x = min(100, max(0, rng.gauss(cx, 8)))
            y = min(100, max(0, rng.gauss(cy, 8)))
        coords[str(i)] = [round(x, 2), round(y, 2)]
    return coords, mode


def _allowed_truck_types(n, avg_q, tau_max, s_value, budget=0.65):
    """Types de camions dont une tournée PLEINE tient dans budget*tau_max.

    Estimation : aller-retour dépôt (~80 km) + nb_arrêts * (service + trajet
    entre voisins). Le trajet entre voisins dépend de la densité de clients
    (grille 100x100, ~moitié des clients par type de produit).
    Sans ce filtre, un camion de 500 avec q≈8 visite ~60 clients → >20 h.
    """
    leg_km = 0.6 * 100 / math.sqrt(max(n * 0.5, 1))
    def est(Q, v):
        return 80 / v + (Q / avg_q) * (s_value + leg_km / v)
    ok = [tt for tt in TRUCK_TYPES if est(*tt) <= budget * tau_max]
    if len(ok) < 2:
        ok = sorted(TRUCK_TYPES, key=lambda tt: est(*tt))[:2]
    return ok


def _fleet(rng, demand, cap_ratio, n_types, allowed):
    """Ajoute des camions (types tirés au hasard) jusqu'à couvrir demand*cap_ratio."""
    types = rng.sample(allowed, min(n_types, len(allowed)))
    fleet = []
    while sum(q for q, _ in fleet) < demand * cap_ratio:
        fleet.append(rng.choice(types))
    return fleet


def generate(n):
    rng = random.Random(n * 7919 + 13)

    n_T = rng.choices(PERIODS, weights=PERIOD_WEIGHTS)[0]
    q_lo = rng.randint(3, 8)
    q_hi = q_lo + rng.randint(4, 10)
    cold_share = rng.uniform(0.40, 0.70)
    cap_ratio = rng.uniform(1.3, 2.5)
    n_types = rng.randint(2, 4)
    p_lo = round(rng.uniform(6.0, 7.5), 1)
    p_hi = round(rng.uniform(11.0, 13.5), 1)
    stock_init = rng.uniform(1.5, 2.5)
    stock_max = rng.uniform(3.0, 4.5)
    stock_min = rng.uniform(0.15, 0.30)

    clients = list(range(1, n + 1))
    T = list(range(1, n_T + 1))
    coords, geo = _coordinates(rng, n)

    q_lt, cold = {}, {}
    for l in clients:
        client_cold_bias = rng.random() < cold_share   # un client est plutôt froid ou non
        for t in T:
            q_lt[(l, t)] = rng.randint(q_lo, q_hi)
            p = 0.85 if client_cold_bias else 0.15
            cold[(l, t)] = rng.random() < p

    # Demande max par période (froid / non froid) → dimensionnement flotte
    dem_f = max(sum(q_lt[l, t] for l in clients if cold[l, t]) for t in T)
    dem_nf = max(sum(q_lt[l, t] for l in clients if not cold[l, t]) for t in T)

    allowed = _allowed_truck_types(n, (q_lo + q_hi) / 2,
                                   CONSTANTS["tau_max"], CONSTANTS["s_value"])
    frigo_fleet = _fleet(rng, dem_f, cap_ratio, n_types, allowed)
    nonfrigo_fleet = _fleet(rng, dem_nf, cap_ratio, n_types, allowed)
    fleet = [(q, v, True) for q, v in frigo_fleet] + [(q, v, False) for q, v in nonfrigo_fleet]
    rng.shuffle(fleet)

    M = list(range(1, len(fleet) + 1))
    frigo_trucks = [k for k, (_, _, f) in zip(M, fleet) if f]
    nonfrigo_trucks = [k for k in M if k not in frigo_trucks]

    # requires_cold[l,t] = [id d'un camion compatible] (convention du loader)
    requires_cold = {
        f"{l},{t}": [rng.choice(frigo_trucks if cold[l, t] else nonfrigo_trucks)]
        for l in clients for t in T
    }

    avg_f = sum(q_lt[l, t] for l in clients for t in T if cold[l, t]) / n_T
    avg_nf = sum(q_lt[l, t] for l in clients for t in T if not cold[l, t]) / n_T

    params = {
        "q_lt": {f"{l},{t}": q_lt[l, t] for l in clients for t in T},
        "requires_cold": requires_cold,
        "v": {str(k): fleet[k - 1][1] for k in M},
        "Q": {str(k): fleet[k - 1][0] for k in M},
        "frigo_trucks": frigo_trucks,
        **CONSTANTS,
        "I_O_init_frigo": round(avg_f * stock_init),
        "I_O_init_nonfrigo": round(avg_nf * stock_init),
        "I_O_max_frigo": round(avg_f * stock_max),
        "I_O_max_nonfrigo": round(avg_nf * stock_max),
        "I_O_min_frigo": max(1, round(avg_f * stock_min)),
        "I_O_min_nonfrigo": max(1, round(avg_nf * stock_min)),
        "P_sale": {str(l): round(rng.uniform(p_lo, p_hi), 1) for l in clients},
        **TAIL,
    }
    inst = {
        "sets": {"N": [0] + clients, "clients": clients, "O": 0,
                 "T": T, "M": M, "coordinates": coords},
        "parameters": params,
    }
    summary = dict(n=n, T=n_T, M=len(M), frigo=len(frigo_trucks),
                   Qs=sorted({q for q, _, _ in fleet}),
                   q=f"{q_lo}-{q_hi}", cold=round(cold_share, 2),
                   cap_ratio=round(cap_ratio, 2), geo=geo)
    return inst, summary


def main(argv):
    overwrite = "--overwrite" in argv
    sizes = [int(a) for a in argv if a.isdigit()] or SIZES
    for n in sizes:
        path = os.path.join(DATA_DIR, f"instance_{n}_clients.json")
        if os.path.exists(path) and not overwrite:
            print(f"skip   {n:4d} (existe déjà, --overwrite pour remplacer)")
            continue
        inst, s = generate(n)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(inst, f)
        print(f"écrit  {n:4d}  T={s['T']} M={s['M']:3d} (frigo {s['frigo']:2d}) "
              f"Q={s['Qs']} q={s['q']:<6} froid={s['cold']} cap/dem={s['cap_ratio']} geo={s['geo']}")


if __name__ == "__main__":
    main(sys.argv[1:])
