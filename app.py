# requirements.txt

```txt
streamlit==1.45.1
pandas==2.2.3
numpy==2.2.5
yfinance==0.2.61
plotly==6.0.1
ta==0.11.0
scipy==1.15.3
```

---

# app.py

```python
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime
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
    # Bancos
    "BBAS3.SA","ITUB4.SA","ITSA4.SA","BBDC4.SA","BBDC3.SA",
    "SANB11.SA","BPAC11.SA","BRSR6.SA","BMGB4.SA",

    # Energia / Petróleo
    "PETR4.SA","PETR3.SA","PRIO3.SA","RECV3.SA","RRRP3.SA",

    # Mineração / Siderurgia
    "VALE3.SA","CSNA3.SA","GGBR4.SA","USIM5.SA",

    # Varejo
    "LREN3.SA","MGLU3.SA","ARZZ3.SA","ALOS3.SA",

    # Industriais
    "WEGE3.SA","EMBR3.SA","TUPY3.SA",

    # Seguros
    "PSSA3.SA","CXSE3.SA",

    # Elétricas
    "TAEE11.SA","EGIE3.SA","CPLE6.SA","ELET3.SA",

    # ETFs
    "BOVA11.SA","SMAL11.SA","IVVB11.SA",

    # BDRs
    "AAPL34.SA","MSFT34.SA","GOGL34.SA","AMZO34.SA"
]

# =========================================================
# FUNÇÕES
# =========================================================

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
    df["VOL_REL"] = df["Volume"] / df["VOL_MEDIA20"]

    return df


def tendencia_ok(df):
    ultimo = df.iloc[-1]

    return (
        ultimo["Close"] > ultimo["EMA169"] and
        ultimo["DI_POS"] > ultimo["DI_NEG"]
    )


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

    return (
        ultimo["Close"] > ultimo["EMA169"] and
        ultimo["DI_POS"] > ultimo["DI_NEG"]
    )


def volume_ok(df):
    ultimo = df.iloc[-1]
    return ultimo["VOL_REL"] > 1


def detectar_123_compra(df):
    if len(df) < 30:
        return False, None, None, None

    dados = df.tail(20).copy()

    lows = dados["Low"].values
    highs = dados["High"].values

    ponto1_idx = np.argmin(lows[:10])

    if ponto1_idx >= 8:
        return False, None, None, None

    ponto2_idx = ponto1_idx + np.argmax(highs[ponto1_idx+1:15]) + 1

    if ponto2_idx <= ponto1_idx:
        return False, None, None, None

    trecho3 = lows[ponto2_idx+1:]

    if len(trecho3) < 3:
        return False, None, None, None

    ponto3_rel = np.argmin(trecho3)
    ponto3_idx = ponto2_idx + 1 + ponto3_rel

    low1 = lows[ponto1_idx]
    low3 = lows[ponto3_idx]

    if low3 <= low1:
        return False, None, None, None

    topo2 = highs[ponto2_idx]
    fechamento = dados["Close"].iloc[-1]

    rompimento = fechamento > topo2

    return rompimento, low1, topo2, low3


def calcular_liquidez(df):
    df = df.copy()
    df["Financeiro"] = df["Close"] * df["Volume"]
    liquidez_media = df["Financeiro"].tail(20).mean()
    return liquidez_media


def backtest_probabilidade(df):
    ocorrencias = 0
    gains = 0
    losses = 0

    for i in range(250, len(df)-15):

        trecho = df.iloc[:i].copy()

        if not tendencia_ok(trecho):
            continue

        if not volume_ok(trecho):
            continue

        padrao, _, _, _ = detectar_123_compra(trecho)

        if not padrao:
            continue

        entrada = trecho.iloc[-1]["Close"]
        atr = trecho.iloc[-1]["ATR"]

        stop = entrada - (1 * atr)
        alvo = entrada + (2 * atr)

        futuro = df.iloc[i:i+15]

        resultado = None

        for _, candle in futuro.iterrows():

            if candle["Low"] <= stop:
                resultado = "loss"
                break

            if candle["High"] >= alvo:
                resultado = "gain"
                break

        if resultado is not None:
            ocorrencias += 1

            if resultado == "gain":
                gains += 1
            else:
                losses += 1

    if ocorrencias == 0:
        return 0, 0, 0, 0

    probabilidade = round((gains / ocorrencias) * 100, 2)

    expectativa = round(
        ((gains / ocorrencias) * 2) - ((losses / ocorrencias) * 1),
        2
    )

    return probabilidade, ocorrencias, gains, expectativa


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
        title=f"{ticker}",
        height=600,
        xaxis_rangeslider_visible=False
    )

    return fig

# =========================================================
# TÍTULO
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

# =========================================================
# BOTÃO
# =========================================================

if st.button("ESCANEAR MERCADO"):

    resultados = []

    progresso = st.progress(0)
    status = st.empty()

    total = len(ATIVOS)

    for i, ticker in enumerate(ATIVOS):

        try:
            status.text(f"Analisando {ticker}...")

            df = yf.download(
                ticker,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                continue

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

            padrao, p1, p2, p3 = detectar_123_compra(df)

            if not padrao:
                continue

            probabilidade, ocorrencias, gains, expectativa = backtest_probabilidade(df)

            if ocorrencias < 10:
                continue

            ultimo = df.iloc[-1]

            score = gerar_score(
                probabilidade,
                ultimo["VOL_REL"],
                expectativa
            )

            entrada = round(ultimo["Close"], 2)
            atr = round(ultimo["ATR"], 2)
            stop = round(entrada - atr, 2)
            alvo = round(entrada + (2 * atr), 2)

            resultados.append({
                "Ranking": 0,
                "Ativo": ticker.replace(".SA", ""),
                "Probabilidade Histórica (%)": probabilidade,
                "Ocorrências": ocorrencias,
                "Gains": gains,
                "Expectativa Matemática": expectativa,
                "Volume Relativo": round(ultimo["VOL_REL"], 2),
                "ATR": atr,
                "Entrada": entrada,
                "Stop": stop,
                "Alvo": alvo,
                "Score": score,
                "Gráfico": df.tail(200)
            })

        except Exception as e:
            st.warning(f"Erro em {ticker}: {e}")

        progresso.progress((i + 1) / total)

    status.text("Análise concluída.")

    if len(resultados) == 0:
        st.error("Nenhum ativo encontrado nas condições atuais.")

    else:

        resultados_df = pd.DataFrame(resultados)

        resultados_df = resultados_df.sort_values(
            by="Score",
            ascending=False
        ).reset_index(drop=True)

        resultados_df["Ranking"] = resultados_df.index + 1

        st.success(f"{len(resultados_df)} ativos encontrados.")

        tabela = resultados_df.drop(columns=["Gráfico"])

        st.dataframe(
            tabela,
            use_container_width=True,
            height=500
        )

        st.divider()

        st.subheader("DETALHAMENTO DOS ATIVOS")

        for _, linha in resultados_df.iterrows():

            with st.expander(f"{linha['Ranking']}º - {linha['Ativo']} | Score {linha['Score']}"):

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Probabilidade Histórica",
                        f"{linha['Probabilidade Histórica (%)']}%"
                    )

                    st.metric(
                        "Expectativa Matemática",
                        linha['Expectativa Matemática']
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
                        linha['Volume Relativo']
                    )

                st.write(f"Ocorrências históricas analisadas: {linha['Ocorrências']}")
                st.write(f"Quantidade de gains antes do stop: {linha['Gains']}")

                fig = criar_grafico(linha['Gráfico'], linha['Ativo'])
                st.plotly_chart(fig, use_container_width=True)

# =========================================================
# RODAPÉ
# =========================================================

st.divider()

st.caption(
    "Scanner Probabilístico B3 | EMA169 + DMI + Padrão 1,2,3 + ATR"
)

```

---

# COMO PUBLICAR NO GITHUB E STREAMLIT

## 1. Criar uma conta

* GitHub
* Streamlit Community Cloud

---

## 2. Criar repositório no GitHub

Nome sugerido:

```text
scanner-probabilistico-b3
```

---

## 3. Subir os arquivos

Arquivos obrigatórios:

```text
app.py
requirements.txt
```

---

## 4. Publicar no Streamlit

Entrar em:

[https://streamlit.io/cloud](https://streamlit.io/cloud)

Selecionar:

* repositório
* branch principal
* app.py

Depois clicar:

```text
DEPLOY
```

---

# OBSERVAÇÕES IMPORTANTES

## O app foi criado para:

* gráfico diário
* confirmação semanal
* swing trade
* tendência forte
* operações compradas

---

## Estrutura probabilística usada

* stop = 1 ATR
* alvo = 2 ATR
* cálculo histórico
* ranking do melhor para o pior

---

## Tudo está em português

* métricas
* ranking
* resultados
* estatísticas
* interface

---

## Próximas melhorias futuras possíveis

* score avançado
* exportação PDF
* alertas automáticos
* banco de dados
* machine learning
* heatmap de setores
* expectativa matemática avançada
* detecção mais refinada do 1,2,3

