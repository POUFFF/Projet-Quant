"""Génération des signaux de trading, séparée de l'exécution.

Idée centrale (le "pattern stratégie") : chaque stratégie ne fait qu'UNE chose
— transformer des prix en une colonne `signal` (1 = on veut être investi,
0 = on veut être en cash). Le moteur d'exécution (backtest.py) ne connaît que
cette colonne : il ne sait pas si le signal vient d'un croisement de moyennes
ou d'un RSI. Résultat : ajouter une stratégie ne touche jamais au moteur.

Aucune dépendance à Streamlit ni à Plotly.
"""
import numpy as np
import pandas as pd


# Registre des stratégies : sert à piloter l'interface et les validations
# sans coder en dur "sma"/"rsi" partout dans l'application.
STRATEGIES = {
    "sma": {
        "label": "Momentum — Croisement de moyennes mobiles (SMA)",
        "family": "Momentum",
    },
    "rsi": {
        "label": "Mean-Reversion — RSI (surachat / survente)",
        "family": "Mean-Reversion",
    },
}


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI).

    Mesure la vitesse et l'ampleur des variations de prix, sur une échelle de
    0 à 100. Intuition : sur les `period` derniers jours, quelle part du
    mouvement était haussière ?
    - RSI proche de 100 → presque que des hausses → "surachat" (trop monté vite)
    - RSI proche de 0   → presque que des baisses → "survente" (trop chuté vite)
    """
    delta = close.diff()
    gain = delta.clip(lower=0)          # hausses du jour (0 si baisse)
    loss = -delta.clip(upper=0)         # baisses du jour en valeur positive
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss            # "relative strength" : gains / pertes
    rsi = 100 - (100 / (1 + rs))
    return rsi


def min_bars(strategy: str, params: dict) -> int:
    """Nombre minimal de jours nécessaires pour que la stratégie ait un sens.

    Sert aux validations (période trop courte ?) et au découpage walk-forward.
    """
    if strategy == "sma":
        return params["slow"] + 10
    if strategy == "rsi":
        return params["period"] + 10
    raise ValueError(f"Stratégie inconnue : {strategy}")


def generate_signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    """Ajoute les colonnes d'indicateur + `signal` + `position` au DataFrame.

    signal   : 1 = investi, 0 = en cash (état souhaité chaque jour)
    position : +1 le jour d'un achat, -1 le jour d'une vente, 0 sinon.
               Le shift(1) exécute le trade au jour t+1 (anti look-ahead bias) :
               le signal est connu à la clôture de t, on ne peut agir que le
               lendemain.
    """
    df = df.copy()

    if strategy == "sma":
        # MOMENTUM : on suit la tendance. On est investi tant que la moyenne
        # rapide (court terme) est au-dessus de la lente (fond de tendance).
        fast, slow = params["fast"], params["slow"]
        df["SMA_fast"] = df["Close"].rolling(fast).mean()
        df["SMA_slow"] = df["Close"].rolling(slow).mean()
        df.dropna(inplace=True)
        df["signal"] = np.where(df["SMA_fast"] > df["SMA_slow"], 1, 0)

    elif strategy == "rsi":
        # MEAN-REVERSION : on parie sur un retour à la moyenne. On ACHÈTE quand
        # le RSI est bas (survente, on anticipe un rebond) et on VEND quand il
        # est haut (surachat, on anticipe un repli).
        period = params["period"]
        oversold, overbought = params["oversold"], params["overbought"]
        df["RSI"] = compute_rsi(df["Close"], period)
        df.dropna(inplace=True)

        # État avec hystérésis : on entre sous `oversold`, on reste investi
        # jusqu'à dépasser `overbought`, puis on sort. Entre les deux seuils
        # on conserve l'état précédent (ffill).
        raw = np.where(df["RSI"] < oversold, 1.0,
                       np.where(df["RSI"] > overbought, 0.0, np.nan))
        df["signal"] = pd.Series(raw, index=df.index).ffill().fillna(0).astype(int)

    else:
        raise ValueError(f"Stratégie inconnue : {strategy}")

    df["position"] = df["signal"].diff().shift(1).fillna(0)
    return df
