Ce projet construit une base de données de graphes financiers à partir d’un univers fixe d’actions (≈60) et applique des méthodes à noyaux sur graphes pour prédire un rendement futur de portefeuille équipondéré. Le projet couvre : récupération des prix, construction de graphes de dépendance (corrélations avec un seuil appliqué), apprentissage (KRR / SVR à noyau pré-calculé), baselines, étude sur le seuil `tau`, résultats et figures.

À chaque date `t`, on associe un graphe `G_t` construit à partir des corrélations empiriques sur une fenêtre glissante de `CORR_WINDOW=60` jours : une arête non orientée `(i,j)` est conservée si `|corr(i,j)|>=tau`. Les fichiers d’arêtes stockent aussi un poids `w` (corrélation), mais dans la première itération les noyaux utilisent uniquement la structure binaire. La cible `y_t` est le rendement futur (log, cumulé) du portefeuille équipondéré sur `FWD_HORIZON=20` jours.

Le dataset est généré par valeur de `tau` :
- `data/processed/tau_0.40/labels.csv` : colonnes `date,y,num_edges`
- `data/graphs/tau_0.40/YYYY-MM-DD.csv` : edge-list `i,j,w`
- `data/processed/tau_0.40/tickers.json` : mapping `index_to_ticker` + paramètres

Pour  l'installation :
Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Les données sont déjà fournies dans ce dépôt GitHub, mais j’ai conservé les fonctions permettant de les régénérer automatiquement.

Exemple pour `tau=0.40` :

```bash
python -m src.build_dataset --tau 0.40
```

Sorties : `data/graphs/tau_0.40/`, `data/processed/tau_0.40/labels.csv`, `data/processed/tau_0.40/tickers.json`.

Pour l'entrainement :

Exemple `tau_0.40` :

```bash
python -m src.train_eval --tag tau_0.40 --kernel wl --model krr --labels_path data/processed/tau_0.40/labels.csv --graphs_dir data/graphs/tau_0.40
python -m src.train_eval --tag tau_0.40 --kernel wl --model svr --labels_path data/processed/tau_0.40/labels.csv --graphs_dir data/graphs/tau_0.40
python -m src.train_eval --tag tau_0.40 --kernel sp --model krr --labels_path data/processed/tau_0.40/labels.csv --graphs_dir data/graphs/tau_0.40
python -m src.train_eval --tag tau_0.40 --kernel sp --model svr --labels_path data/processed/tau_0.40/labels.csv --graphs_dir data/graphs/tau_0.40
```

Les runs sont écrits dans : `results/fixed/<tag>/<kernel>/<model>/run_XXXX/`.

Contenu d’un run :
- `config.json` : configuration (split, hyperparams, chemins).
- `metrics.json` : métriques (ici MSE/MAE/R²) sur échantillons val/test + meilleurs hyperparamètres.
- `timings.json` : temps de chargement, temps de calcul de la matrice de Gram, temps d'entrainement du modèle, temps total.
- `predictions.csv` : `date,split,y,yhat`.
- `figures/` : figures automatiques (prédictions vs vrai, résidus, etc.).

J'ai ajouté deux baseline :

Le premier est un baseline de moyenne : `yhat = mean(y_train)`.

```bash
python -m src.mean_baseline --tag tau_0.40 --labels_path data/processed/tau_0.40/labels.csv
```

Dans le second, on vectorise nos graphes : on remplace le graphe par quelques données scalaires sur la fenêtre passée (rendement et volatilité du portefeuille équipondéré, etc.), puis on ajuste une régression linéaire ridge pour prédire le rendement futur.

```bash
python -m src.baseline_train_eval --tag tau_0.40 --labels_path data/processed/tau_0.40/labels.csv --raw_prices_path data/raw/adj_close_2014-01-01_2024-12-31.csv
```

Agrégation des derniers runs par `(kernel, model)` et par tag :

```bash
python -m src.summarize_results
```

Analyse multi-`tau` et génération de figures (ex. `R^2` vs `tau`, `gram_sec` vs `tau`) :

```bash
python -m src.analyze_taus
```

Sorties attendues :
- `results/fixed/all_taus.csv`, `results/fixed/all_taus.md`
- `results/fixed/figures/*_vs_tau.png`


Et enfin quelques ajouts "quality of life" :

Vérification du caractère symétrique défini positif (validité numérique des matrices de Gram sur train) :

```bash
python -m src.check_psd --tag tau_0.40 --kernel wl --labels_path data/processed/tau_0.40/labels.csv --graphs_dir data/graphs/tau_0.40
python -m src.check_psd --tag tau_0.40 --kernel sp --labels_path data/processed/tau_0.40/labels.csv --graphs_dir data/graphs/tau_0.40
```

Densité des graphes (à quel point les graphes sont proches du complet) :

```bash
python -m src.check_complete_graphs --labels_path data/processed/tau_0.40/labels.csv --tickers_path data/processed/tau_0.40/tickers.json
```
