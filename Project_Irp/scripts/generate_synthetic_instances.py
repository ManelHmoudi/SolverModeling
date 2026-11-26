"""
Générateur d'instances synthétiques IRP (format du modèle final, voir
docs/model_formulation.md §4 pour la liste et les unités des paramètres).

Chaque instance a ses propres caractéristiques (graine = nb clients, donc
reproductible) :
  - nombre de périodes T (4 à 14)
  - plage des quantités demandées q_lt
  - part des demandes froides 40 %–70 %
  - flotte : types de camions (Q, v) filtrés pour qu'une tournée pleine
    respecte tau_max ; nombre de véhicules dimensionné sur la demande
  - géographie : uniforme, clusters ou mixte
  - prix de vente, stocks dépôt

Usage :
    python -m scripts.generate_synthetic_instances              # toutes les tailles
    python -m scripts.generate_synthetic_instances 120 300      # tailles choisies
    python -m scripts.generate_synthetic_instances --overwrite  # écrase les existantes
    python -m scripts.generate_synthetic_instances --replicates 5 100 200
        # 5 instances indépendantes par taille -> data/replicates/instance_{n}_clients_r{r}.json
    python -m scripts.generate_synthetic_instances --migrate
        # convertit TOUTES les instances existantes au format final, sans
        # toucher à la demande, la flotte ni la géographie
"""
import json
import math
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPLICATES_DIR = os.path.join(DATA_DIR, "replicates")

SIZES = (
    list(range(100, 201, 10))
    + [220, 250, 270, 300, 320, 350, 370, 400, 450, 500,
       550, 600, 650, 700, 750, 800]
)

# Types de camions possibles : (capacité Q, vitesse km/h)
TRUCK_TYPES = [(100, 90), (120, 90), (150, 85), (180, 80), (200, 70),
               (250, 65), (300, 60), (350, 55), (400, 55), (500, 50)]

PERIODS = [4, 5, 6, 7, 8, 9, 10, 12, 14]
PERIOD_WEIGHTS = [2, 2, 2, 2, 2, 2, 2, 1, 1]

# Paramètres communs à toutes les instances (unités : docs/model_formulation.md §4)
CONSTANTS = {
    # temps
    "s_value": 0.25,        # h, service chez un client = chargement au dépôt
    "tau_max": 10.0,        # h, durée maximale d'une tournée (journée d'un chauffeur)
    "c1": 20.0,             # €/h d'arrivée anticipée (avant l'ouverture de la réception, sans attente)
    "c2": 40.0,             # €/h de retard (livraison hors horaires : pénalité de service)
    # transport
    "F_veh": 20.0,          # €/tournée lancée
    "c_km": 0.8,            # €/véhicule-km (retour à vide compris)
    "p5": 3.0,              # €/h, groupe frigorifique
    "c_load": 0.002,        # €/(unité·km)
    # stocks
    "h_cold": 0.305,        # €/(unité·période), dépôt, produits froids
    "h_ambient": 0.3,       # €/(unité·période), dépôt, produits ambiants
    "F_order": 50.0,        # €/commande fournisseur
    # finance
    "P_purchase": 5.0,      # €/unité
    "L_S": 3,               # délai client, en périodes
    "L_P": 2,               # délai fournisseur, en périodes
    # CMEM linéarisé : valeurs véhicule du PRP (Bektas & Laporte 2011 ; Demir, Bektas & Laporte 2012)
    "g": 9.81, "Cr": 0.01, "Cd": 0.7, "A_f": 3.912, "rho": 1.2041, "a": 0.0,
    "w": 6350, "kg_per_unit": 10, "e_co2": 2.67, "fuel_to_joules": 36000000,
}

# Clés des anciens formats, retirées par --migrate
OBSOLETE_KEYS = ("c_route_value", "DIO", "DSO", "DPO", "period_days", "C_max", "E_max",
                 "T_max", "B", "BIG_M", "R_max_frigo", "R_max_nonfrigo",
                 "requires_cold", "h_O_space", "alpha_r", "e_stock",
                 "min_delivery_threshold", "tau_min", "MOQ_frigo", "MOQ_nonfrigo",
                 "ET_value", "LT_value", "U_client", "h_client", "eta_tf", "eta_engine",
                 "k_e", "N_e", "V_e")


# ── Paramètres dérivés ────────────────────────────────────────────────────────

def time_windows(n_inst, clients, tau_max):
    """Fenêtre horaire propre à chaque client, comme les horaires de réception
    d'une pharmacie ou d'un hôpital : matin, après-midi ou journée (en heures
    depuis le départ du dépôt ; tirage reproductible, graine = taille d'instance)."""
    rng = random.Random(n_inst * 31337 + 7)
    kinds = {"matin": (0.05, 0.45), "apres_midi": (0.35, 0.80), "journee": (0.05, 0.80)}
    et, lt = {}, {}
    for l in clients:
        lo, hi = kinds[rng.choice(list(kinds))]
        et[str(l)], lt[str(l)] = round(lo * tau_max, 2), round(hi * tau_max, 2)
    return et, lt


def depot_params(q_lt, cold, n_T, init=2.0, smax=4.0, smin=0.25):
    """Stocks dépôt (initial, min, max) par classe."""
    avg = {"frigo": 0.0, "nonfrigo": 0.0}
    for key, q in q_lt.items():
        avg["frigo" if cold[key] else "nonfrigo"] += q / n_T
    out = {}
    for c in ("frigo", "nonfrigo"):
        out[f"I_O_init_{c}"] = round(avg[c] * init)
        out[f"I_O_max_{c}"] = round(avg[c] * smax)
        out[f"I_O_min_{c}"] = max(1, round(avg[c] * smin)) if avg[c] > 0 else 0
    return out


def migrate(inst):
    """N'importe quel ancien format -> format final. Demande, flotte et
    géographie inchangées ; paramètres complétés et nettoyés."""
    sets, p = inst["sets"], inst["parameters"]
    clients, T = sets["clients"], sets["T"]

    if "coordinates" not in sets:
        # ancien repli d = |i - j| × 10 km : points alignés, distances identiques
        sets["coordinates"] = {str(i): [10 * i, 0] for i in sets["N"]}
    if "requires_cold" in p:
        frigo = set(p["frigo_trucks"])
        p["cold"] = {k: v[0] in frigo for k, v in p["requires_cold"].items()}
    if "h_O_space" in p:
        p.setdefault("h_cold", p["h_O_space"] + p.get("alpha_r", 0) * p.get("e_stock", 0))
        p.setdefault("h_ambient", p["h_O_space"])
    if isinstance(p.get("P_purchase"), dict):
        p["P_purchase"] = float(next(iter(p["P_purchase"].values())))
    if "P_sale" in p:
        avg = sum(p["P_sale"].values()) / len(p["P_sale"])
        for l in clients:
            p["P_sale"].setdefault(str(l), round(avg, 2))

    depot = depot_params(p["q_lt"], p["cold"], len(T))
    for k, v in depot.items():
        p.setdefault(k, v)
    for k in OBSOLETE_KEYS:
        p.pop(k, None)
    for k, v in CONSTANTS.items():
        if k in ("s_value", "kg_per_unit"):
            p.setdefault(k, v)          # caractéristiques propres à l'instance : gardées
        else:
            p[k] = v
    p["ET"], p["LT"] = time_windows(len(clients), clients, p["tau_max"])
    return resize_fleet(inst)


def resize_fleet(inst, slack=0.7, max_rounds=8):
    """Flotte dimensionnée sur le TEMPS (tau_max) et pas seulement sur la
    capacité : pour chaque classe et chaque période, la durée totale des
    tournées du plan « livraisons obligatoires seulement » doit tenir dans
    slack × tau_max × (nombre de camions de la classe). On ajoute des camions
    du type le plus fréquent de la classe jusqu'à ce que ce soit vrai.
    Demande et géographie inchangées."""
    import numpy as np
    sys.path.insert(0, BASE_DIR)
    from models.parameters import instance_from_dict
    from solvers.nsga3.decoder import n_genes
    from solvers.nsga3.problem import decode_full
    from solvers.nsga3.evaluator import route_duration

    p, sets = inst["parameters"], inst["sets"]
    for _ in range(max_rounds):
        sets_, params_ = instance_from_dict(inst)
        nC, nT = len(sets_["clients"]), len(sets_["T"])
        x = np.zeros(n_genes(sets_))
        x[nC * nT:nC * nT + nC] = 0.5
        rr = decode_full(x, sets_, params_)
        added = False
        for is_cold in (True, False):
            trucks = [k for k in sets["M"] if (k in p["frigo_trucks"]) == is_cold]
            if not trucks:
                continue
            busiest = max(sum(route_duration(i["path"], k, params_) for k, i in r.items()
                              if (k in p["frigo_trucks"]) == is_cold) for r in rr["routes_data"].values())
            need = int(math.ceil(busiest / (slack * p["tau_max"])))
            types = [(p["Q"][str(k)], p["v"][str(k)]) for k in trucks]
            q, v = max(set(types), key=types.count)
            for _ in range(max(0, need - len(trucks))):
                k = max(sets["M"]) + 1
                sets["M"].append(k)
                p["Q"][str(k)], p["v"][str(k)] = q, v
                if is_cold:
                    p["frigo_trucks"].append(k)
                added = True
        if not added:
            break
    return inst


# ── Génération ────────────────────────────────────────────────────────────────

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
    """Types de camions dont une tournée PLEINE tient dans budget*tau_max
    (aller-retour dépôt ~80 km + arrêts × (service + trajet entre voisins))."""
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


def generate(n, replicate=0):
    """replicate=0 reproduit l'instance historique ; r>=1 donne une instance
    indépendante de même taille (autre graine)."""
    rng = random.Random(n * 7919 + 13 + replicate * 104729)

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
            q_lt[f"{l},{t}"] = rng.randint(q_lo, q_hi)
            cold[f"{l},{t}"] = rng.random() < (0.85 if client_cold_bias else 0.15)

    dem_f = max(sum(q_lt[f"{l},{t}"] for l in clients if cold[f"{l},{t}"]) for t in T)
    dem_nf = max(sum(q_lt[f"{l},{t}"] for l in clients if not cold[f"{l},{t}"]) for t in T)

    allowed = _allowed_truck_types(n, (q_lo + q_hi) / 2, CONSTANTS["tau_max"], CONSTANTS["s_value"])
    frigo_fleet = _fleet(rng, dem_f, cap_ratio, n_types, allowed)
    nonfrigo_fleet = _fleet(rng, dem_nf, cap_ratio, n_types, allowed)
    fleet = [(q, v, True) for q, v in frigo_fleet] + [(q, v, False) for q, v in nonfrigo_fleet]
    rng.shuffle(fleet)

    M = list(range(1, len(fleet) + 1))
    frigo_trucks = [k for k, (_, _, f) in zip(M, fleet) if f]
    nonfrigo_trucks = [k for k in M if k not in frigo_trucks]
    for key in cold:            # tirages conservés pour garder la suite aléatoire historique
        rng.choice(frigo_trucks if cold[key] else nonfrigo_trucks)

    params = {
        "q_lt": q_lt,
        "cold": cold,
        "v": {str(k): fleet[k - 1][1] for k in M},
        "Q": {str(k): fleet[k - 1][0] for k in M},
        "frigo_trucks": frigo_trucks,
        **CONSTANTS,
        **depot_params(q_lt, cold, n_T, stock_init, stock_max, stock_min),
        "P_sale": {str(l): round(rng.uniform(p_lo, p_hi), 1) for l in clients},
    }
    params["ET"], params["LT"] = time_windows(n, clients, params["tau_max"])
    inst = resize_fleet({
        "sets": {"N": [0] + clients, "clients": clients, "O": 0,
                 "T": T, "M": M, "coordinates": coords},
        "parameters": params,
    })
    summary = dict(n=n, T=n_T, M=len(M), frigo=len(frigo_trucks),
                   Qs=sorted({q for q, _, _ in fleet}),
                   q=f"{q_lo}-{q_hi}", cold=round(cold_share, 2),
                   cap_ratio=round(cap_ratio, 2), geo=geo)
    return inst, summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _migrate_all():
    for folder in (DATA_DIR, REPLICATES_DIR):
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not (fname.startswith("instance_") and fname.endswith(".json")):
                continue
            path = os.path.join(folder, fname)
            with open(path, encoding="utf-8") as fh:
                inst = json.load(fh)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(migrate(inst), fh)
            print(f"migré  {os.path.relpath(path, BASE_DIR)}")


def _write(path, inst):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(inst, fh)


def main(argv):
    overwrite = "--overwrite" in argv
    if "--migrate" in argv:
        _migrate_all()
        return
    if "--replicates" in argv:
        i = argv.index("--replicates")
        n_rep = int(argv[i + 1])
        rest = argv[:i] + argv[i + 2:]
        sizes = [int(a) for a in rest if a.isdigit()] or SIZES
        os.makedirs(REPLICATES_DIR, exist_ok=True)
        for n in sizes:
            for r in range(1, n_rep + 1):
                path = os.path.join(REPLICATES_DIR, f"instance_{n}_clients_r{r}.json")
                if os.path.exists(path) and not overwrite:
                    print(f"skip   {os.path.basename(path)}")
                    continue
                inst, s = generate(n, replicate=r)
                _write(path, inst)
                print(f"écrit  {os.path.basename(path)}  T={s['T']} M={s['M']} geo={s['geo']}")
        return
    sizes = [int(a) for a in argv if a.isdigit()] or SIZES
    for n in sizes:
        path = os.path.join(DATA_DIR, f"instance_{n}_clients.json")
        if os.path.exists(path) and not overwrite:
            print(f"skip   {n:4d} (existe déjà, --overwrite pour remplacer)")
            continue
        inst, s = generate(n)
        _write(path, inst)
        print(f"écrit  {n:4d}  T={s['T']} M={s['M']:3d} (frigo {s['frigo']:2d}) "
              f"Q={s['Qs']} q={s['q']:<6} froid={s['cold']} cap/dem={s['cap_ratio']} geo={s['geo']}")


if __name__ == "__main__":
    main(sys.argv[1:])
