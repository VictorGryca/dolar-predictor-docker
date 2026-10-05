"""Treino do modelo USD/BRL (D+1). Transcrição do notebooks/treino.ipynb.

Roda no estágio `train` do Dockerfile: baixa BRL=X do Yahoo (ou usa o CSV de fallback),
avalia Ridge contra o naive num split cronológico, reajusta com todos os dados e grava
model.joblib + meta.json no diretório de saída.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, N_LAGS, features_from_closes

TICKER = "BRL=X"
TEST_FRAC = 0.2
RIDGE_ALPHA = 1.0


def carregar_dados(start, csv_fallback):
    try:
        import yfinance as yf

        raw = yf.download(TICKER, start=start, interval="1d", auto_adjust=False, progress=False)
        if raw.empty:
            raise ValueError("download vazio")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Close"]].rename(columns={"Close": "close"}).dropna()
        df.index.name = "date"
        fonte = "yahoo"
    except Exception as e:
        print(f"download falhou ({e}); usando {csv_fallback}")
        df = pd.read_csv(csv_fallback, index_col="date", parse_dates=True)
        fonte = f"csv:{Path(csv_fallback).name}"

    # candle do dia corrente é parcial durante o pregão
    hoje = pd.Timestamp.today().normalize()
    df = df[df.index < hoje]
    return df, fonte


def montar_dataset(close):
    ret = np.log(close).diff()
    feat = pd.DataFrame(index=close.index)
    for i in range(N_LAGS - 1, -1, -1):
        feat["r_t" if i == 0 else f"r_t-{i}"] = ret.shift(i)
    feat["target"] = ret.shift(-1)
    feat["close"] = close
    feat["close_next"] = close.shift(-1)
    return feat.dropna()


def novo_modelo():
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--csv", default="data/brl_x.csv")
    args = parser.parse_args()

    df, fonte = carregar_dados(args.start, args.csv)
    print(f"fonte: {fonte} | {df.index.min().date()} -> {df.index.max().date()} | {len(df)} linhas")

    feat = montar_dataset(df["close"])

    # garante que o dataset de treino e a função usada pela API montam a mesma entrada
    t = len(df) - 2
    x_api = features_from_closes(df["close"].iloc[t - N_LAGS : t + 1].to_numpy())
    np.testing.assert_allclose(x_api[0], feat.loc[df.index[t], FEATURES].to_numpy())

    n_test = int(len(feat) * TEST_FRAC)
    train, test = feat.iloc[:-n_test], feat.iloc[-n_test:]

    modelo = novo_modelo().fit(train[FEATURES].to_numpy(), train["target"].to_numpy())
    pred_ret = modelo.predict(test[FEATURES].to_numpy())
    pred_close = test["close"].to_numpy() * np.exp(pred_ret)
    naive_close = test["close"].to_numpy()
    real = test["close_next"].to_numpy()

    metricas = {
        "periodo": f"{test.index.min().date()} -> {test.index.max().date()}",
        "linhas": len(test),
        "naive_mae": float(mean_absolute_error(real, naive_close)),
        "ridge_mae": float(mean_absolute_error(real, pred_close)),
        "naive_rmse": float(np.sqrt(mean_squared_error(real, naive_close))),
        "ridge_rmse": float(np.sqrt(mean_squared_error(real, pred_close))),
        "ridge_acerto_direcao": float((np.sign(pred_ret) == np.sign(test["target"].to_numpy())).mean()),
        "sempre_alta_acerto_direcao": float((test["target"] > 0).mean()),
    }
    print(json.dumps(metricas, indent=2, ensure_ascii=False))

    # modelo final: todos os dados, para usar as cotações mais recentes
    final = novo_modelo().fit(feat[FEATURES].to_numpy(), feat["target"].to_numpy())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / "model.joblib")
    meta = {
        "ticker": TICKER,
        "model": f"StandardScaler + Ridge(alpha={RIDGE_ALPHA})",
        "target": "log-retorno do dia seguinte",
        "n_lags": N_LAGS,
        "features": FEATURES,
        "closes_esperados": N_LAGS + 1,
        "fonte_dados": fonte,
        "dados_ate": str(df.index.max().date()),
        "linhas_treino_final": len(feat),
        "metricas_teste": metricas,
        "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_date_arg": os.getenv("TRAIN_DATE"),
        "sklearn_version": sklearn.__version__,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"artefato salvo em {out.resolve()}")


if __name__ == "__main__":
    main()
