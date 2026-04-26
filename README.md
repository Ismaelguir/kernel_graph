Ce projet construit une base de données de graphes financiers à partir d’un univers fixe d’actions (≈60) et applique des méthodes à noyaux sur graphes pour prédire un rendement futur de portefeuille équipondéré. Le projet couvre : récupération des prix, construction de graphes de dépendance (corrélations avec un seuil appliqué), apprentissage (KRR / SVR à noyau pré-calculé), baselines, étude sur le seuil `tau`, résultats et figures.

À chaque date `t`, on associe un graphe `G_t` construit à partir des corrélations empiriques sur une fenêtre glissante de `CORR_WINDOW=60` jours : une arête non orientée `(i,j)` est conservée si `|corr(i,j)|>=tau`. Les fichiers d’arêtes stockent aussi un poids `w` (corrélation), mais dans la première itération les noyaux utilisent uniquement la structure binaire. La cible `y_t` est le rendement futur (log, cumulé) du portefeuille équipondéré sur `FWD_HORIZON=20` jours.
=======
# Kernel graph finance — pipeline final

Pipeline nettoyé : une seule convention, toutes les sorties dans `results_final`.

## Principe

- `tau` est un hyperparamètre **interne** : pour chaque `(kernel, model)`, on évalue les candidats `tau` sur validation et on garde le meilleur.
- Le split reste **temporel** (`train / val / test`) — pas de k-fold aléatoire.
# Kernel graph finance — pipeline final

Pipeline unique et nettoyé : entraînement + benchmark + agrégation, avec sorties centralisées dans `results_final`.

## Règles du pipeline

- `tau` est un hyperparamètre **interne** (sélection sur `val_mse`) pour chaque couple `(kernel, model)`.
- Split **temporel fixe** : train / val / test.
- Résultats unifiés sous `results_final/fixed/<tag>/...`.

## Structure du code

```
src/
    core/       # config, utils, splits, metrics, plots
    data/       # graph_io, build_dataset, matrice_cor
    kernels/    # wl, shortest_path
    models/     # krr, svr
    pipeline/   # train_eval, baseline_train_eval, benchmark_inference, summarize_results
    checks/     # check_psd, check_complete_graphs
    legacy/     # scripts obsolètes (RuntimeError explicite)
```

## Données attendues

```
data/processed/tau_xx/labels.csv
data/processed/tau_xx/tickers.json
data/graphs/tau_xx/YYYY-MM-DD.csv
```

Régénération (si nécessaire) :

```bash
python -m src.data.build_dataset --tau 0.40
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Lancement

```bash
chmod +x run_final_pipeline.sh
./run_final_pipeline.sh
```

Variables optionnelles :

- `RESULTS_DIR` (défaut: `results_final`)
- `TAG` (défaut: `internal_tau`)
- `TAU_TAGS` (défaut: auto-détection de tous les `tau_*`)
- `TRAIN_END`, `VAL_END`

Exemple :

```bash
TAG=exp1 TAU_TAGS=tau_0.25,tau_0.30,tau_0.40 ./run_final_pipeline.sh
```

## Modèles exécutés

| kernel | model |
|--------|-------|
| wl | krr |
| wl | svr |
| sp | krr |
| sp | svr |
| baseline | ridge |
| baseline | mean |

## Sorties

Par run :

```
results_final/fixed/<tag>/<kernel>/<model>/run_XXXX/
    config.json
    metrics.json
    timings.json
    predictions.csv
    figures/
```

Agrégations :

```
results_final/fixed/<tag>/table.csv + table.md
results_final/fixed/all_taus.csv + all_taus.md
results_final/fixed/best_by_val.csv + best_by_val.md
```

Benchmark inférence :

```
results_final/fixed/<tag>/inference_<kernel>_<model>.json
```

Les temps d'inférence (`mean_infer_sec`, `median_infer_sec`, `p95_infer_sec`) sont repris dans les tables d'agrégation.

## Vérification PSD (optionnel)

```bash
python -m src.checks.check_psd --tag tau_0.40 --kernel wl \
    --labels_path data/processed/tau_0.40/labels.csv \
    --graphs_dir data/graphs/tau_0.40
```
```
