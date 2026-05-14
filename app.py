import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Scanner Probabilístico B3",
    layout="wide"
)

# =========================================================
# LISTA DE ATIVOS
# =========================================================

ATIVOS = [

    "GARE11.SA",
    "HGLG11.SA",
    "XPLG11.SA",
    "VILG11.SA",
    "BRCO11.SA",
    "BTLG11.SA",
    "XPML11.SA",
    "VISC11.SA",
    "HSML11.SA",
    "MALL11.SA",
    "KNRI11.SA",
    "JSRE11.SA",
    "PVBI11.SA",
    "HGRE11.SA",
    "MXRF11.SA",
    "KNCR11.SA",
    "KNIP11.SA",
    "CPTS11.SA",
    "IRDM11.SA",
    "TGAR11.SA",
    "TRXF11.SA",
    "HGRU11.SA",
    "ALZR11.SA",
    "XPCA11.SA",
    "VGIA11.SA",
    "RBRR11.SA",
    "KNSC11.SA",
    "HGCR11.SA",
    "MCCI11.SA",
    "RECR11.SA",
    "VRTA11.SA",
    "BCFF11.SA",
    "HFOF11.SA",
    "XPSF11.SA",
    "RBRP11.SA",
    "RBRF11.SA",
    "RZTR11.SA",
    "RURA11.SA",
    "VGIR11.SA",
    "CVBI11.SA",
    "UTLL11.SA",
    "GGRC11.SA",
    "AUVP11.SA",
    "IEEX11.SA",

    "TAEE11.SA",
    "CMIG4.SA",
    "CPFE3.SA",
    "EQTL3.SA",
    "ELET3.SA",
    "ELET6.SA",
    "ALUP11.SA",
    "TRPL4.SA",
    "NEOE3.SA",
    "ENGI11.SA",
    "SBSP3.SA",
    "SAPR11.SA",
    "CSMG3.SA",

    "BBAS3.SA",
    "ITUB4.SA",
    "ITSA4.SA",
    "BBDC4.SA",
    "BBDC3.SA",
    "SANB11.SA",
    "BPAC11.SA",
    "BRSR6.SA",

    "VALE3.SA",
    "PETR4.SA",
    "PETR3.SA",
    "WEGE3.SA",
    "SUZB3.SA",
    "KLBN11.SA",
    "JBSS3.SA",
    "PRIO3.SA",
    "RECV3.SA",
    "EGIE3.SA",
    "VIVT3.SA",
    "TOTS3.SA",
    "RAIL3.SA",

    "AAPL34.SA",
    "MSFT34.SA",
    "GOGL34.SA",
    "AMZO34.SA",
    "META34.SA",
    "NVDC34.SA",
    "JPMC34.SA",
    "DISB34.SA",
    "SBUX34.SA",

    "BOVA11.SA",
    "SMAL11.SA",
    "IVVB11.SA",
    "DIVO11.SA"

]

# =========================================================
# FUNÇÕES
# =========================================================

def ajustar_colunas(df):

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def calcular_indicadores(df):

    df = df.copy()

    df["EMA169"] = ta.trend.EMAIndicator(
        close=df["Close"],
        window=169
    ).ema_indicator()

    adx = ta.trend.ADXIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    df["DI_POS"] = adx.adx_pos()
    df["DI_NEG"] = adx.adx_neg()
    df["ADX"] = adx.adx()

    atr = ta.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    df["ATR"] = atr.average_true_range()

    df["VOL_MEDIA20"] = df["Volume"].rolling(20).mean()

    df["VOL_REL"] = (
        df["Volume"] / df["VOL_MEDIA20"]
    )

    return df


def tendencia_ok(df):

    ultimo = df.iloc[-1]

    condicao = (
        (ultimo["Close"] > ultimo["EMA169"]) and
        (ultimo["DI_POS"] > ultimo["DI_NEG"])
    )

    return bool(condicao)


def tendencia_semanal_ok(df_diario):

    semanal = df_diario.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    semanal = calcular_indicadores(semanal)

    if len(semanal) < 180:
        return False

    ultimo = semanal.iloc[-1]

    condicao = (
        (ultimo["Close"] > ultimo["EMA169"]) and
        (ultimo["DI_POS"] > ultimo["DI_NEG"])
    )

    return bool(condicao)


def volume_ok(df):

    ultimo = df.iloc[-1]

    return bool(ultimo["VOL_REL"] > 1)


def detectar_123_compra(df):

    if len(df) < 30:
        return False

    dados = df.tail(20).copy()

    lows = dados["Low"].values
    highs = dados["High"].values

    ponto1_idx = np.argmin(lows[:10])

    if ponto1_idx >= 8:
        return False

    trecho_p2 = highs[ponto1_idx + 1:15]

    if len(trecho_p2) == 0:
        return False

    ponto2_idx = (
        ponto1_idx +
        np.argmax(trecho_p2) +
        1
    )

    if ponto2_idx <= ponto1_idx:
        return False

    trecho3 = lows[ponto2_idx + 1:]

    if len(trecho3) < 3:
        return False

    ponto3_rel = np.argmin(trecho3)

    ponto3_idx = (
        ponto2_idx +
        1 +
        ponto3_rel
    )

    low1 = lows[ponto1_idx]
    low3 = lows[ponto3_idx]

    if low3 <= low1:
        return False

    topo2 = highs[ponto2_idx]

    fechamento = float(
        dados["Close"].iloc[-1]
    )

    rompimento = fechamento > topo2

    return bool(rompimento)


def calcular_liquidez(df):

    financeiro = (
        df["Close"] * df["Volume"]
    )

    return float(
        financeiro.tail(20).mean()
    )


def backtest_probabilidade(df):

    ocorrencias = 0
    gains = 0
    losses = 0

    for i in range(250, len(df) - 15):

        trecho = df.iloc[:i].copy()

        try:

            if not tendencia_ok(trecho):
                continue

            if not volume_ok(trecho):
                continue

            if not detectar_123_compra(trecho):
                continue

            entrada = float(
                trecho.iloc[-1]["Close"]
            )

            atr = float(
                trecho.iloc[-1]["ATR"]
            )

            stop = entrada - atr
            alvo = entrada + (2 * atr)

            futuro = df.iloc[i:i + 15]

            resultado = None

            for _, candle in futuro.iterrows():

                low = float(candle["Low"])
                high = float(candle["High"])

                if low <= stop:
                    resultado = "loss"
                    break

                if high >= alvo:
                    resultado = "gain"
                    break

            if resultado is not None:

                ocorrencias += 1

                if resultado == "gain":
                    gains += 1
                else:
                    losses += 1

        except:
            continue

    if ocorrencias == 0:
        return 0, 0, 0, 0

    probabilidade = round(
        (gains / ocorrencias) * 100,
        2
    )

    expectativa = round(
        ((gains / ocorrencias) * 2) -
        ((losses / ocorrencias) * 1),
        2
    )

    return (
        probabilidade,
        ocorrencias,
        gains,
        expectativa
    )


def gerar_score(probabilidade, vol_rel, expectativa):

    score = 0

    score += probabilidade * 0.5
    score += min(vol_rel * 10, 20)
    score += max(expectativa * 20, 0)

    return round(score, 2)


def criar_grafico(df, ticker):

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA169"],
            mode="lines",
            name="EMA169"
        )
    )

    fig.update_layout(
        title=ticker,
        height=600,
        xaxis_rangeslider_visible=False
    )

    return fig

# =========================================================
# INTERFACE
# =========================================================

st.title("SCANNER PROBABILÍSTICO B3")

st.markdown("""

### Estratégia Utilizada

- EMA169
- DMI (DI+ > DI−)
- Tendência semanal confirmando
- Padrão 1,2,3 de compra
- Volume acima da média
- Stop = 1 ATR
- Alvo = 2 ATR
- Ranking probabilístico

""")

if st.button("ESCANEAR MERCADO"):

    resultados = []

    progresso = st.progress(0)

    status = st.empty()

    total = len(ATIVOS)

    for i, ticker in enumerate(ATIVOS):

        try:

            status.text(
                f"Analisando {ticker}..."
            )

            df = yf.download(
                ticker,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                continue

            df = ajustar_colunas(df)

            if len(df) < 300:
                continue

            liquidez = calcular_liquidez(df)

            if liquidez < 2_000_000:
                continue

            df = calcular_indicadores(df)

            df.dropna(inplace=True)

            if len(df) < 250:
                continue

            if not tendencia_ok(df):
                continue

            if not tendencia_semanal_ok(df):
                continue

            if not volume_ok(df):
                continue

            if not detectar_123_compra(df):
                continue

            (
                probabilidade,
                ocorrencias,
                gains,
                expectativa
            ) = backtest_probabilidade(df)

            if ocorrencias < 10:
                continue

            ultimo = df.iloc[-1]

            entrada = round(
                float(ultimo["Close"]),
                2
            )

            atr = round(
                float(ultimo["ATR"]),
                2
            )

            stop = round(
                entrada - atr,
                2
            )

            alvo = round(
                entrada + (2 * atr),
                2
            )

            score = gerar_score(
                probabilidade,
                float(ultimo["VOL_REL"]),
                expectativa
            )

            resultados.append({

                "Ativo":
                    ticker.replace(".SA", ""),

                "Probabilidade":
                    probabilidade,

                "Ocorrências":
                    ocorrencias,

                "Gains":
                    gains,

                "Expectativa":
                    expectativa,

                "Volume":
                    round(
                        float(
                            ultimo["VOL_REL"]
                        ),
                        2
                    ),

                "ATR":
                    atr,

                "Entrada":
                    entrada,

                "Stop":
                    stop,

                "Alvo":
                    alvo,

                "Score":
                    score,

                "Grafico":
                    df.tail(200)

            })

        except Exception as e:

            st.warning(
                f"Erro em {ticker}: {e}"
            )

        progresso.progress(
            (i + 1) / total
        )

    status.text("Análise concluída.")

    if len(resultados) == 0:

        st.error(
            "Nenhum ativo encontrado."
        )

    else:

        resultados_df = pd.DataFrame(
            resultados
        )

        resultados_df = (
            resultados_df
            .sort_values(
                by="Score",
                ascending=False
            )
            .reset_index(drop=True)
        )

        resultados_df.index += 1

        st.success(
            f"{len(resultados_df)} ativos encontrados."
        )

        tabela = resultados_df.drop(
            columns=["Grafico"]
        )

        st.dataframe(
            tabela,
            use_container_width=True,
            height=500
        )

        st.divider()

        st.subheader(
            "DETALHAMENTO DOS ATIVOS"
        )

        for rank, linha in (
            resultados_df.iterrows()
        ):

            with st.expander(
                f"#{rank} - "
                f"{linha['Ativo']} | "
                f"Score {linha['Score']}"
            ):

                col1, col2, col3 = (
                    st.columns(3)
                )

                with col1:

                    st.metric(
                        "Probabilidade",
                        f"{linha['Probabilidade']}%"
                    )

                    st.metric(
                        "Expectativa",
                        linha['Expectativa']
                    )

                with col2:

                    st.metric(
                        "Entrada",
                        linha['Entrada']
                    )

                    st.metric(
                        "Stop",
                        linha['Stop']
                    )

                with col3:

                    st.metric(
                        "Alvo",
                        linha['Alvo']
                    )

                    st.metric(
                        "Volume Relativo",
                        linha['Volume']
                    )

                st.write(
                    f"Ocorrências históricas: "
                    f"{linha['Ocorrências']}"
                )

                st.write(
                    f"Gains antes do stop: "
                    f"{linha['Gains']}"
                )

                fig = criar_grafico(
                    linha['Grafico'],
                    linha['Ativo']
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

st.divider()

st.caption(
    "Scanner Probabilístico B3 "
    "| EMA169 + DMI + ATR"
)
