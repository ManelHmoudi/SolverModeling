from docplex.mp.model import Model

mdl = Model(name="IRP_ManyObjective")

# ─────────────────────────────────────────
# ENSEMBLES
# ─────────────────────────────────────────
N       = [0, 1, 2, 3, 4]   # 0=Origin, 4=Destination, 1-3=clients
clients = [1, 2, 3]
O       = 0
D       = 4
T       = [1, 2]             # 2 périodes
M       = [1, 2]             # 0=non frigo, 1=frigorifique

A = [(i, j) for i in N for j in N if i != j]

# ─────────────────────────────────────────
# 3.1 PARAMÈTRES CLIENTS
# ─────────────────────────────────────────
# phi_l : identifiant client l
phi = {1: "client_1", 2: "client_2", 3: "client_3"}

# q_lt : demande du client l à période t
q_lt = {
    (1, 1): 20, (1, 2): 25,
    (2, 1): 30, (2, 2): 35,
    (3, 1): 15, (3, 2): 20
}

# q'_lt : quantité effectivement transportée (small first, large later)
q_prime_lt = {(l, t): q_lt[(l, t)] for l in clients for t in T}

# ─────────────────────────────────────────
# 3.2 PARAMÈTRES TRANSPORT
# ─────────────────────────────────────────
# v_k : vitesse constante du camion de type k
v = {1: 80, 2: 60}

# d_ij : distance entre nœud i et j
d = {(i, j): abs(i - j) * 10 for (i, j) in A}

# c_ijk : coût de transport du camion k sur l'arc (i,j)
c_ijk = {
    (i, j, k): d[(i, j)] * (1.2 if k == 1 else 1.0)
    for (i, j) in A for k in M
}

# c1 : coût de stockage temporaire (arrivée trop tôt)
c1 = 2.0

# c2 : coût unitaire de pénalité (arrivée trop tard)
c2 = 5.0

# ─────────────────────────────────────────
# 3.3 PARAMÈTRES TEMPORELS
# ─────────────────────────────────────────
# tau_lt : moment d'arrivée de la marchandise au client l à période t
tau_lt = {
    (1, 1): 9.0,  (1, 2): 10.0,
    (2, 1): 11.0, (2, 2): 13.0,
    (3, 1): 8.5,  (3, 2): 9.5
}

# [ET_lt, LT_lt] : fenêtre de temps acceptée par le client l à période t
ET = {(l, t): 8  for l in clients for t in T}
LT = {(l, t): 12 for l in clients for t in T}

# tau_min, tau_max : temps total min et max autorisé (O → D)
tau_min = 6.0
tau_max = 24.0

# ─────────────────────────────────────────
# 3.4 PARAMÈTRES STOCK
# ─────────────────────────────────────────
# I_init : niveau de stock initial au nœud i à la période t
I_init = {(i, t): 50 for i in N for t in T}

# I_max_i : capacité maximale de stockage au nœud i
I_max = {i: 200 for i in N}

# I_min_i : stock de sécurité au nœud i
I_min = {i: 10 for i in N}

# h_i : coût unitaire de stockage au nœud i
h = {i: 0.5 for i in N}

# ─────────────────────────────────────────
# 3.5 PARAMÈTRES CAPACITÉS
# ─────────────────────────────────────────
# Q_k : capacité max du camion de type k
Q = {1: 100, 2: 80}

# C_max : budget max de coût économique total
C_max = 8000

# ─────────────────────────────────────────
# 3.6 PARAMÈTRES FINANCIERS
# ─────────────────────────────────────────
# B : budget financier total
B = 10000

# BFR : besoin en fonds de roulement
BFR = 3000

# DSO : délai de paiement client (jours)
DSO = 30

# DPO : délai de paiement fournisseur (jours)
DPO = 45

# V_i : valeur du produit au nœud i
V_val = {i: 100 for i in N}

# P_vente, P_achat : prix de vente et d'achat unitaires
P_vente = 8.0
P_achat = 5.0

# ─────────────────────────────────────────
# 3.7 PARAMÈTRES ENVIRONNEMENTAUX CO₂
# ─────────────────────────────────────────
# alpha_ij : paramètre énergie lié au poids (Bektas & Laporte 2011)
alpha_co2 = {(i, j): 0.01 for (i, j) in A}

# beta : paramètre énergie lié à la vitesse
beta_co2 = 0.002

# e : facteur de conversion CO₂ (kg CO₂/unité énergie)
e_co2 = 2.68

# E_max : limite max de coût carbone total autorisé
E_max = 5000

# ─────────────────────────────────────────
# 3.8 PARAMÈTRES ÉNERGÉTIQUES RÉFRIGÉRATION (k=1 uniquement)
# ─────────────────────────────────────────
# p5 : coût de réfrigération par unité de temps durant transport (yuan/h)
p5 = 3.0

# e_stock : consommation énergétique par unité stockée
e_stock = 0.1

# alpha_r : coefficient énergie (stock → consommation)
alpha_r = 0.05

# ─────────────────────────────────────────
# VARIABLES DE DÉCISION
# ─────────────────────────────────────────
# x[i,j,t,k] — binaire : véhicule k emprunte arc (i,j) à période t
x = {
    (i, j, t, k): mdl.binary_var(name=f"x_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# f[i,j,t,k] — continue >= 0 : charge transportée
f = {
    (i, j, t, k): mdl.continuous_var(lb=0, name=f"f_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

print(f"Modèle    : {mdl.name}")
print(f"Variables x : {len(x)}")
print(f"Variables f : {len(f)}")
print(f"Total       : {mdl.number_of_variables}")

# ─────────────────────────────────────────
# BLOC 2 — CONTRAINTES
# ─────────────────────────────────────────

BIG_M = 99999  # constante Big-M

# ─────────────────────────────────────────
# 5.1 STRUCTURE RÉSEAU ET FLUX
# ─────────────────────────────────────────

# (c1) Départ unique : chaque camion k quitte O une seule fois par période t
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[O, j, t, k] for j in N if j != O) == 1,
            ctname=f"c1_k{k}_t{t}"
        )

# (c2) Arrivée unique : chaque camion k atteint D une seule fois par période t
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[i, D, t, k] for i in N if i != D) == 1,
            ctname=f"c2_k{k}_t{t}"
        )

# (c3) Conservation de flux
for k in M:
    for t in T:
        for j in N:
            entrant = mdl.sum(x[i, j, t, k] for i in N if i != j)
            sortant = mdl.sum(x[j, i, t, k] for i in N if i != j)
            if j == O:
                mdl.add_constraint(entrant - sortant == -1, ctname=f"c3_O_k{k}_t{t}")
            elif j == D:
                mdl.add_constraint(entrant - sortant == 1,  ctname=f"c3_D_k{k}_t{t}")
            else:
                mdl.add_constraint(entrant - sortant == 0,  ctname=f"c3_{j}_k{k}_t{t}")

# ─────────────────────────────────────────
# 5.2 CAPACITÉ ET LOGISTIQUE
# ─────────────────────────────────────────

# (c4) Capacité camion : f_ijt^k <= Q_k * x_ijt^k
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                ctname=f"c4_{i}{j}_k{k}_t{t}"
            )

# (c5a) Bornes de stock : I_min <= I_init <= I_max
for i in N:
    for t in T:
        assert I_min[i] <= I_init[i, t] <= I_max[i], \
            f"Stock initial hors bornes : nœud {i}, période {t}"

# (c5b) Bilan de stock via flux entrants
# I_it = I_i,t-1 + flux_entrant - q_it
# Simplifié : flux entrant au client i à période t = q'_lt
I_calc = {}
for i in clients:
    for t in T:
        t_prev = t - 1
        stock_prev = I_init[i, t_prev] if t_prev >= 1 else I_init[i, 1]
        flux_entrant = q_prime_lt[i, t]   # connu avant résolution
        I_calc[i, t] = stock_prev + flux_entrant - q_lt[i, t]

# (c6) Demande satisfaite : sum_t(q'_lt) = sum_t(q_lt)
for l in clients:
    mdl.add_constraint(
        mdl.sum(q_prime_lt[l, t] for t in T) == mdl.sum(q_lt[l, t] for t in T),
        ctname=f"c6_l{l}"
    )

# (c7) Équilibre des charges au nœud j
# entrant - sortant = q'_jt
for j in clients:
    for t in T:
        entrant = mdl.sum(f[i, j, t, k] for i in N if i != j for k in M)
        sortant = mdl.sum(f[j, l, t, k] for l in N if l != j for k in M)
        mdl.add_constraint(
            entrant - sortant == q_prime_lt[j, t],
            ctname=f"c7_{j}_t{t}"
        )

# ─────────────────────────────────────────
# 5.3 CONTRAINTES TEMPORELLES
# ─────────────────────────────────────────

# Temps de service s_i (déchargement chez client i, en heures)
s = {i: 0.5 for i in N}

# Variable tau[i,t] : moment d'arrivée au nœud i à période t
tau = {
    (i, t): mdl.continuous_var(lb=0, name=f"tau_{i}_{t}")
    for i in N for t in T
}

# (c8) Temps sur l'arc : activation uniquement si x=1
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= d[i, j] / v[k] - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c8_{i}{j}_k{k}_t{t}"
            )

# (c9) Arrivée au nœud suivant
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= tau[i, t] + s[i] + d[i, j] / v[k] - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c9_{i}{j}_k{k}_t{t}"
            )

# (c10) Fenêtre globale O → D
for t in T:
    mdl.add_constraint(tau[D, t] >= tau_min, ctname=f"c10_min_t{t}")
    mdl.add_constraint(tau[D, t] <= tau_max, ctname=f"c10_max_t{t}")

# (c11) Fenêtre client : ET_lt <= tau_lt <= LT_lt
for l in clients:
    for t in T:
        mdl.add_constraint(tau[l, t] >= ET[l, t], ctname=f"c11_ET_l{l}_t{t}")
        mdl.add_constraint(tau[l, t] <= LT[l, t], ctname=f"c11_LT_l{l}_t{t}")

# ─────────────────────────────────────────
# 5.4 CONTRAINTES ÉCONOMIQUES ET ENVIRONNEMENTALES
# ─────────────────────────────────────────

# (c12), (c13), (c14) — connectées après définition des objectifs f1, f2, f4
# Décommentées dans le Bloc 3 :
# mdl.add_constraint(f1_expr <= C_max, ctname="c12_budget_eco")
# mdl.add_constraint(f2_expr <= E_max, ctname="c13_limite_carbone")
# mdl.add_constraint(f4_expr <= B,     ctname="c14_budget_financier")

print(f"Contraintes ajoutées : {mdl.number_of_constraints}")