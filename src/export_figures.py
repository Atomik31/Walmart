"""
Export des figures Walmart vers reports/figures/
Usage : /opt/anaconda3/bin/python src/export_figures.py
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "Walmart_Store_sales.csv"

# ── Chargement ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)

def save(fig, name):
    path = OUTPUT_DIR / f"{name}.png"
    fig.write_image(str(path))
    print(f"  {path.name}")

print("Export des figures Walmart...")

# 01 — Distribution Weekly_Sales
fig = px.histogram(df['Weekly_Sales'].dropna(), nbins=40,
                   title='Distribution des ventes hebdomadaires',
                   labels={'value': 'Weekly Sales ($)', 'count': 'Nombre'},
                   color_discrete_sequence=['steelblue'])
save(fig, "01_weekly_sales_distribution")

# 02 — Distribution Temperature
fig = px.histogram(df['Temperature'].dropna(), nbins=30,
                   title='Distribution Temperature',
                   labels={'value': 'Temperature', 'count': 'Nombre'},
                   color_discrete_sequence=['tomato'])
save(fig, "02_temperature_distribution")

# 03 — Distribution Fuel_Price
fig = px.histogram(df['Fuel_Price'].dropna(), nbins=30,
                   title='Distribution Fuel_Price',
                   labels={'value': 'Fuel Price ($)', 'count': 'Nombre'},
                   color_discrete_sequence=['orange'])
save(fig, "03_fuel_price_distribution")

# 04 — Distribution CPI
fig = px.histogram(df['CPI'].dropna(), nbins=30,
                   title='Distribution CPI',
                   labels={'value': 'CPI', 'count': 'Nombre'},
                   color_discrete_sequence=['green'])
save(fig, "04_cpi_distribution")

# 05 — Distribution Unemployment
fig = px.histogram(df['Unemployment'].dropna(), nbins=30,
                   title='Distribution Unemployment',
                   labels={'value': 'Unemployment (%)', 'count': 'Nombre'},
                   color_discrete_sequence=['purple'])
save(fig, "05_unemployment_distribution")

# 06 — Ventes : jours fériés vs normaux
df_tmp = df.dropna(subset=['Weekly_Sales', 'Holiday_Flag'])
df_tmp['Holiday'] = df_tmp['Holiday_Flag'].map({0: 'Jour normal', 1: 'Jour férié'})
fig = px.box(df_tmp, x='Holiday', y='Weekly_Sales',
             title='Ventes hebdomadaires — Jour férié vs normal',
             labels={'Weekly_Sales': 'Weekly Sales ($)', 'Holiday': ''},
             color='Holiday', color_discrete_map={'Jour normal': 'steelblue', 'Jour férié': 'tomato'})
save(fig, "06_sales_holiday_vs_normal")

# 07 — Matrice de corrélation
num_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
corr = df[num_cols].corr().round(2)
fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1,
                title='Matrice de corrélation', width=600, height=500)
save(fig, "07_correlation_matrix")

# ── Preprocessing pour les modèles ─────────────────────────────────────────────
df = df.dropna(subset=['Weekly_Sales'])
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df = df.dropna(subset=['Date'])
df['Year']      = df['Date'].dt.year
df['Month']     = df['Date'].dt.month
df['Day']       = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
    df = df.dropna(subset=[col])
    m, s = df[col].mean(), df[col].std()
    df = df[(df[col] >= m - 3*s) & (df[col] <= m + 3*s)]

df = df.dropna(subset=['Holiday_Flag', 'Store'])

cat_features = ['Store', 'Holiday_Flag']
num_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                'Year', 'Month', 'Day', 'DayOfWeek']

X = df[cat_features + num_features]
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features)
])

# Entraînement des 3 modèles
lr_pipe = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
lr_pipe.fit(X_train, y_train)

ridge_pipe = Pipeline([('preprocessor', preprocessor), ('model', Ridge())])
gs_ridge = GridSearchCV(ridge_pipe, {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]}, cv=5, scoring='r2')
gs_ridge.fit(X_train, y_train)

lasso_pipe = Pipeline([('preprocessor', preprocessor), ('model', Lasso(max_iter=100000))])
gs_lasso = GridSearchCV(lasso_pipe, {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]}, cv=5, scoring='r2')
gs_lasso.fit(X_train, y_train)

# 08 — Coefficients LinearRegression
feature_names = (num_features +
                 list(lr_pipe.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(cat_features)))
coefs = pd.Series(lr_pipe.named_steps['model'].coef_, index=feature_names)
coefs_top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15).reset_index()
coefs_top.columns = ['Feature', 'Coefficient']
fig = px.bar(coefs_top, x='Coefficient', y='Feature', orientation='h',
             title='Top 15 coefficients — Régression linéaire',
             color='Coefficient', color_continuous_scale='RdBu', color_continuous_midpoint=0,
             height=500)
fig.update_layout(yaxis={'categoryorder': 'total ascending'})
save(fig, "08_lr_coefficients")

# 09 — LR : Réel vs Prédit
y_pred_lr = lr_pipe.predict(X_test)
fig = px.scatter(x=y_test, y=y_pred_lr,
                 labels={'x': 'Valeurs réelles', 'y': 'Prédictions'},
                 title=f'Régression linéaire — Réel vs Prédit (R²={r2_score(y_test, y_pred_lr):.3f})',
                 opacity=0.6)
fig.add_shape(type='line', x0=y_test.min(), x1=y_test.max(),
              y0=y_test.min(), y1=y_test.max(), line=dict(color='red', dash='dash'))
save(fig, "09_lr_pred_vs_real")

# 10 — Ridge : Réel vs Prédit
y_pred_ridge = gs_ridge.predict(X_test)
fig = px.scatter(x=y_test, y=y_pred_ridge,
                 labels={'x': 'Valeurs réelles', 'y': 'Prédictions'},
                 title=f'Ridge — Réel vs Prédit (R²={r2_score(y_test, y_pred_ridge):.3f}, alpha={gs_ridge.best_params_["model__alpha"]})',
                 opacity=0.6, color_discrete_sequence=['seagreen'])
fig.add_shape(type='line', x0=y_test.min(), x1=y_test.max(),
              y0=y_test.min(), y1=y_test.max(), line=dict(color='red', dash='dash'))
save(fig, "10_ridge_pred_vs_real")

# 11 — Lasso : Réel vs Prédit
y_pred_lasso = gs_lasso.predict(X_test)
fig = px.scatter(x=y_test, y=y_pred_lasso,
                 labels={'x': 'Valeurs réelles', 'y': 'Prédictions'},
                 title=f'Lasso — Réel vs Prédit (R²={r2_score(y_test, y_pred_lasso):.3f}, alpha={gs_lasso.best_params_["model__alpha"]})',
                 opacity=0.6, color_discrete_sequence=['darkorange'])
fig.add_shape(type='line', x0=y_test.min(), x1=y_test.max(),
              y0=y_test.min(), y1=y_test.max(), line=dict(color='red', dash='dash'))
save(fig, "11_lasso_pred_vs_real")

# 12 — Comparaison R² des 3 modèles
results = pd.DataFrame({
    'Modèle': ['LinearRegression', 'Ridge', 'Lasso'],
    'R² Test': [
        round(r2_score(y_test, y_pred_lr), 4),
        round(r2_score(y_test, y_pred_ridge), 4),
        round(r2_score(y_test, y_pred_lasso), 4)
    ],
    'MAE Test': [
        round(mean_absolute_error(y_test, y_pred_lr)),
        round(mean_absolute_error(y_test, y_pred_ridge)),
        round(mean_absolute_error(y_test, y_pred_lasso))
    ]
})

fig = px.bar(results, x='Modèle', y='R² Test',
             title='Comparaison R² Test — 3 modèles',
             color='Modèle', text='R² Test',
             color_discrete_sequence=['steelblue', 'seagreen', 'darkorange'])
fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig.update_layout(yaxis_range=[0.9, 1.0])
save(fig, "12_models_r2_comparison")

print(f"\nDone — 12 figures exportées dans {OUTPUT_DIR}")
