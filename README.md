![Walmart](Walmart-Logo.png)

# Walmart — Prédiction des ventes hebdomadaires

Projet réalisé dans le cadre du bloc 3 de la certification CDSD (Jedha).

---

## Contexte

Walmart veut mieux comprendre comment les indicateurs économiques (chômage, prix du carburant, CPI, température) influencent les ventes hebdomadaires de ses magasins. L'objectif : construire un modèle de régression pour estimer ces ventes et identifier les facteurs les plus déterminants.

---

## Ce que j'ai fait

- EDA : distributions des variables, impact des jours fériés sur les ventes, corrélations
- Nettoyage : suppression des outliers (±3σ), création de features temporelles depuis la colonne Date
- **Régression linéaire** (baseline) : entraînement, évaluation (R², MAE), interprétation des coefficients
- **Ridge et Lasso** : réduction de l'overfitting par régularisation, comparaison des performances

---

## Stack

- Python — Pandas, Scikit-learn, Matplotlib, Seaborn

---

## Données

Dataset issu d'une compétition Kaggle, adapté par Jedha. Contient les ventes hebdomadaires de plusieurs magasins Walmart avec des variables économiques et météo associées.

---

## Structure

```
Walmart/
├── data/
│   └── raw/
│       └── Walmart_Store_sales.csv
├── docs/
│   └── 01-Walmart_sales.ipynb
├── notebooks/
│   └── walmart.ipynb
├── reports/
│   └── figures/
│       ├── 01_weekly_sales_distribution.png
│       ├── 02_temperature_distribution.png
│       ├── 03_fuel_price_distribution.png
│       ├── 04_cpi_distribution.png
│       ├── 05_unemployment_distribution.png
│       ├── 06_sales_holiday_vs_normal.png
│       ├── 07_correlation_matrix.png
│       ├── 08_lr_coefficients.png
│       ├── 09_lr_pred_vs_real.png
│       ├── 10_ridge_pred_vs_real.png
│       ├── 11_lasso_pred_vs_real.png
│       └── 12_models_r2_comparison.png
└── README.md
```

---

Julien CHARLIER — [(Github : Atomik31)](https://github.com/Atomik31)
