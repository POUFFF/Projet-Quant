# SMA Momentum Backtester

Application de backtesting d'une stratégie momentum par croisement de moyennes mobiles simples (SMA), construite avec Python et Streamlit.

## Fonctionnalités

- Données de marché réelles via `yfinance` (Yahoo Finance)
- Stratégie golden cross / death cross paramétrable
- Métriques de performance : rendement total, ratio de Sharpe, max drawdown, nombre de trades
- Comparaison automatique vs Buy & Hold
- Graphiques interactifs (prix, portefeuille, drawdown) avec Plotly
- Interface web déployable sur Streamlit Cloud

## Installation locale

```bash
git clone https://github.com/TON_USERNAME/sma-backtester.git
cd sma-backtester
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement sur Streamlit Cloud (gratuit)

1. Push ce repo sur GitHub
2. Va sur [share.streamlit.io](https://share.streamlit.io)
3. Connecte ton compte GitHub
4. Sélectionne ce repo → `app.py` → Deploy

## Stratégie

**Achat (golden cross)** : quand SMA rapide > SMA lente  
**Vente (death cross)** : quand SMA rapide < SMA lente  

Le capital est entièrement investi à chaque signal d'achat et liquidé à chaque signal de vente.

## Métriques

| Métrique | Description |
|---|---|
| Rendement total | `(valeur finale - capital initial) / capital initial` |
| Ratio de Sharpe | Rendement annualisé ajusté au risque `(μ × 252) / (σ × √252)` |
| Max drawdown | Perte maximale depuis un sommet historique |
| Nb. de trades | Nombre de signaux d'achat + vente générés |

## Stack technique

- **Python 3.11+**
- **Streamlit** — interface web
- **yfinance** — données de marché
- **Pandas / NumPy** — calculs financiers
- **Plotly** — visualisations interactives

## Améliorations possibles

- Ajouter d'autres stratégies (RSI, Bollinger Bands, MACD)
- Optimisation des paramètres par grid search
- Intégration de frais de transaction
- Backtesting multi-actifs / portefeuille
- Analyse de sensibilité des paramètres
