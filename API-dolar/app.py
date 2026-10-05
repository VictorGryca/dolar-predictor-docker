"""API de inferência do modelo USD/BRL (D+1).

Carrega models/model.joblib e models/meta.json na subida. O artefato vem do estágio
`train` do Dockerfile (COPY --from=train).
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import features_from_closes, price_from_return

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "models"))
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_DIR / "model.joblib")
    state["meta"] = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    yield
    state.clear()


app = FastAPI(title="dolar-predictor", version="1.0", lifespan=lifespan)


class PredictIn(BaseModel):
    closes: list[float] = Field(
        ...,
        description="Últimos fechamentos de BRL=X, do mais antigo para o mais recente (quantidade em /health: closes_esperados).",
        examples=[[5.1905, 5.1875, 5.2229, 5.2029, 5.1791, 5.2226]],
    )


class PredictOut(BaseModel):
    ultimo_fechamento: float
    previsao_d1: float
    variacao_pct: float
    modelo_dados_ate: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "meta": state.get("meta")}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    meta = state["meta"]
    try:
        x = features_from_closes(body.closes, meta["n_lags"])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    r_prev = float(state["model"].predict(x)[0])
    ultimo = body.closes[-1]
    previsao = price_from_return(ultimo, r_prev)
    return PredictOut(
        ultimo_fechamento=ultimo,
        previsao_d1=round(previsao, 4),
        variacao_pct=round((previsao / ultimo - 1) * 100, 4),
        modelo_dados_ate=meta["dados_ate"],
    )
