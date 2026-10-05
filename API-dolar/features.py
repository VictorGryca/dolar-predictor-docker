"""Lógica de features compartilhada entre o treino (train.py) e a API (app.py).

Fica num arquivo só para os dois estágios montarem a entrada do modelo do mesmo jeito.
"""
import numpy as np

N_LAGS = 5
# do mais antigo para o mais recente; r_t = log(close_t / close_{t-1})
FEATURES = [f"r_t-{i}" for i in range(N_LAGS - 1, 0, -1)] + ["r_t"]


def features_from_closes(closes, n_lags=N_LAGS):
    """Recebe os últimos n_lags + 1 fechamentos e devolve a matriz (1, n_lags) de log-retornos."""
    closes = np.asarray(closes, dtype=float)
    if closes.ndim != 1 or len(closes) != n_lags + 1:
        raise ValueError(f"esperava {n_lags + 1} fechamentos, recebeu {len(closes)}")
    if np.any(closes <= 0):
        raise ValueError("fechamentos precisam ser positivos")
    return np.diff(np.log(closes)).reshape(1, -1)


def price_from_return(last_close, log_return):
    """Reconstrói o preço de D+1 a partir do último fechamento e do log-retorno previsto."""
    return float(last_close * np.exp(log_return))
