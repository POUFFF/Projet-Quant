import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

st.set_page_config(page_title="SMA Momentum Backtester", layout="wide")

st.title("SMA Momentum Backtester")
st.caption("Stratégie de trading par croisement de moyennes mobiles simples (golden cross / death cross)")

with st.sidebar:
    st.header("Paramètres")

    ticker = st.text_input("Ticker (ex: SPY, AAPL, MSFT)").upper().strip()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Début", value=date(2018, 1, 1))
    with col2:
        # Évite de demander une date de fin égale au jour courant si le marché n'a pas encore fermé
        end_date = st.date_input("Fin", value=date.today())

    sma_short = st.slider("SMA rapide (jours)", min_value=5, max_value=100, value=20, step=1)
    sma_long = st.slider("SMA lente (jours)", min_value=20, max_value=300, value=50, step=5)

    if sma_short >= sma_long:
        st.warning("La SMA rapide doit être plus courte que la SMA lente.")
        st.stop()

    capital = st.number_input(
        "Capital initial ($)",
        min_value=1000,
        max_value=1_000_000,
        value=10_000,
        step=1000
    )

    run = st.button("Lancer le backtest", type="primary", use_container_width=True)


@st.cache_data(ttl=3600)
def load_data(ticker: str, start, end):
    """
    Charge les données avec yfinance et retourne:
    - df: DataFrame avec colonne Close
    - error_message: None si OK, sinon message d'erreur
    """
    try:
        if not ticker:
            return None, "Le ticker est vide."

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        if start_ts >= end_ts:
            return None, "La date de début doit être antérieure à la date de fin."

        # Yahoo Finance traite souvent end comme exclusif
        end_ts = end_ts + pd.Timedelta(days=1)

        df = yf.download(
            ticker,
            start=start_ts,
            end=end_ts,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df is None or df.empty:
            return None, f"Aucune donnée retournée pour {ticker}. Vérifie le ticker, la période, ou réessaie plus tard."

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            return None, f"La colonne 'Close' est absente pour {ticker}. Colonnes reçues: {list(df.columns)}"

        df = df[["Close"]].copy()
        df.columns = ["Close"]
        df.dropna(inplace=True)

        if df.empty:
            return None, f"Les données de clôture de {ticker} sont vides après nettoyage."

        return df, None

    except Exception as e:
        return None, f"Erreur lors du chargement des données pour {ticker}: {e}"


def run_backtest(df: pd.DataFrame, short_w: int, long_w: int, capital: float):
    df = df.copy()

    df["SMA_fast"] = df["Close"].rolling(short_w).mean()
    df["SMA_slow"] = df["Close"].rolling(long_w).mean()
    df.dropna(inplace=True)

    if df.empty:
        return None, [], [], capital

    df["signal"] = np.where(df["SMA_fast"] > df["SMA_slow"], 1, 0)
    df["position"] = df["signal"].diff().fillna(0)

    cash = float(capital)
    shares = 0.0
    portfolio_values = []
    buy_signals = []
    sell_signals = []

    for idx, row in df.iterrows():
        price = float(row["Close"])

        if row["position"] == 1 and cash > 0:
            shares = cash / price
            cash = 0.0
            buy_signals.append((idx, price))

        elif row["position"] == -1 and shares > 0:
            cash = shares * price
            shares = 0.0
            sell_signals.append((idx, price))

        portfolio_values.append(cash + shares * price)

    final_value = cash + shares * float(df["Close"].iloc[-1])

    df["portfolio"] = portfolio_values
    df["bh"] = capital * df["Close"] / float(df["Close"].iloc[0])

    df.attrs["short_w"] = short_w
    df.attrs["long_w"] = long_w

    return df, buy_signals, sell_signals, final_value


def compute_metrics(df: pd.DataFrame, capital: float):
    final_val = float(df["portfolio"].iloc[-1])
    total_return = (final_val - capital) / capital * 100
    bh_return = (float(df["bh"].iloc[-1]) - capital) / capital * 100

    daily_returns = df["portfolio"].pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    rolling_max = df["portfolio"].cummax()
    drawdown = (df["portfolio"] - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) * 100 if not drawdown.empty else 0.0

    n_trades = int((df["position"] != 0).sum())

    win_days = int((daily_returns > 0).sum())
    total_days = int(len(daily_returns))
    win_rate = (win_days / total_days * 100) if total_days > 0 else 0.0

    return {
        "final_val": final_val,
        "total_return": total_return,
        "bh_return": bh_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "drawdown": drawdown
    }


def plot_results(df: pd.DataFrame, buy_signals, sell_signals, ticker: str, metrics: dict):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.3, 0.2],
        vertical_spacing=0.04,
        subplot_titles=(
            f"{ticker} — Prix et moyennes mobiles",
            "Valeur du portefeuille vs Buy & Hold",
            "Drawdown"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Prix",
            line=dict(color="#378ADD", width=1.5),
            hovertemplate="%{x|%d %b %Y}<br>Prix: $%{y:.2f}<extra></extra>"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA_fast"],
            name=f"SMA {df.attrs.get('short_w', 'rapide')}",
            line=dict(color="#EF9F27", width=1.5, dash="dot"),
            hovertemplate="SMA rapide: $%{y:.2f}<extra></extra>"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA_slow"],
            name=f"SMA {df.attrs.get('long_w', 'lente')}",
            line=dict(color="#7F77DD", width=1.5),
            hovertemplate="SMA lente: $%{y:.2f}<extra></extra>"
        ),
        row=1,
        col=1
    )

    if buy_signals:
        bx, by = zip(*buy_signals)
        fig.add_trace(
            go.Scatter(
                x=list(bx),
                y=list(by),
                mode="markers",
                name="Achat",
                marker=dict(color="#1D9E75", size=9, symbol="triangle-up"),
                hovertemplate="%{x|%d %b %Y}<br>Achat: $%{y:.2f}<extra></extra>"
            ),
            row=1,
            col=1
        )

    if sell_signals:
        sx, sy = zip(*sell_signals)
        fig.add_trace(
            go.Scatter(
                x=list(sx),
                y=list(sy),
                mode="markers",
                name="Vente",
                marker=dict(color="#E24B4A", size=9, symbol="triangle-down"),
                hovertemplate="%{x|%d %b %Y}<br>Vente: $%{y:.2f}<extra></extra>"
            ),
            row=1,
            col=1
        )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["portfolio"],
            name="Stratégie",
            line=dict(color="#1D9E75", width=2),
            hovertemplate="%{x|%d %b %Y}<br>Portefeuille: $%{y:,.0f}<extra></extra>"
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["bh"],
            name="Buy & Hold",
            line=dict(color="#B4B2A9", width=1.5, dash="dash"),
            hovertemplate="%{x|%d %b %Y}<br>Buy & Hold: $%{y:,.0f}<extra></extra>"
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=metrics["drawdown"] * 100,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#E24B4A", width=1),
            fillcolor="rgba(226,75,74,0.15)",
            hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.1f}%<extra></extra>"
        ),
        row=3,
        col=1
    )

    fig.update_layout(
        height=680,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=60, b=20, l=0, r=0),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.update_yaxes(tickprefix="$", row=1, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=2, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(ticksuffix="%", row=3, col=1, gridcolor="#f0f0f0")
    fig.update_xaxes(gridcolor="#f0f0f0")

    return fig


if run:
    with st.spinner(f"Chargement des données pour {ticker}..."):
        df, load_error = load_data(ticker, start_date, end_date)

    if load_error:
        st.error(load_error)
        st.stop()

    st.success(f"{len(df)} lignes chargées pour {ticker}.")
    st.caption(f"Période couverte : du {df.index.min().date()} au {df.index.max().date()}")

    min_required = sma_long + 10
    if len(df) < min_required:
        st.error(
            f"Pas assez de données pour calculer correctement la stratégie sur **{ticker}**. "
            f"{len(df)} lignes chargées, mais il en faut au moins **{min_required}**. "
            f"Choisis une période plus longue ou réduis la SMA lente."
        )
        st.stop()

    df, buy_signals, sell_signals, final_value = run_backtest(df, sma_short, sma_long, capital)

    if df is None or df.empty:
        st.error("Le backtest n’a pas pu être exécuté, car les données après calcul des moyennes mobiles sont insuffisantes.")
        st.stop()

    metrics = compute_metrics(df, capital)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Rendement stratégie",
        f"{metrics['total_return']:+.1f}%",
        delta=f"{metrics['total_return'] - metrics['bh_return']:+.1f}% vs B&H"
    )
    col2.metric("Buy & Hold", f"{metrics['bh_return']:+.1f}%")
    col3.metric("Ratio de Sharpe", f"{metrics['sharpe']:.2f}")
    col4.metric("Max drawdown", f"{metrics['max_dd']:.1f}%")
    col5.metric("Nb. de trades", metrics["n_trades"])

    st.divider()

    fig = plot_results(df, buy_signals, sell_signals, ticker, metrics)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    with st.expander("Voir les données brutes"):
        st.dataframe(
            df[["Close", "SMA_fast", "SMA_slow", "portfolio", "bh"]]
            .tail(100)
            .style.format({
                "Close": "${:.2f}",
                "SMA_fast": "${:.2f}",
                "SMA_slow": "${:.2f}",
                "portfolio": "${:,.0f}",
                "bh": "${:,.0f}"
            }),
            use_container_width=True
        )

else:
    st.info("Configure les paramètres dans la barre latérale et clique sur **Lancer le backtest**.")

    st.markdown("""
    ### Comment ça fonctionne

    **Stratégie momentum SMA croisée :**
    - **Achat (golden cross)** : quand la SMA rapide passe au-dessus de la SMA lente → signal haussier
    - **Vente (death cross)** : quand la SMA rapide repasse en dessous → signal baissier

    **Métriques calculées :**
    - **Rendement total** : performance de la stratégie vs Buy & Hold
    - **Ratio de Sharpe** : rendement ajusté au risque (> 1 = bon, > 2 = excellent)
    - **Max drawdown** : perte maximale depuis un sommet
    - **Nombre de trades** : nombre de signaux générés
    """)