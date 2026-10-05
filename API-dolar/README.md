# API-dolar

Build multistage: o estágio `train` roda `train.py` e gera `model.joblib` + `meta.json`; o estágio `serve` copia esses arquivos com `COPY --from=train` e sobe a FastAPI na porta 8080.

`data/brl_x.csv` é o fallback do treino: se o download do Yahoo falhar no build, o treino usa esse CSV.

## Build (PowerShell, de dentro desta pasta)

```powershell
docker build -t dolar-api --build-arg TRAIN_DATE=$(Get-Date -Format yyyy-MM-dd) .
```

Mudar `TRAIN_DATE` força o retreino; com o mesmo valor, o Docker reaproveita a camada de treino do cache.

## Run

```powershell
docker run --rm -p 8080:8080 dolar-api
```

## Testes

```powershell
Invoke-RestMethod http://localhost:8080/health

Invoke-RestMethod -Method Post -Uri http://localhost:8080/predict `
  -ContentType "application/json" `
  -Body '{"closes":[5.1905,5.1875,5.2229,5.2029,5.1791,5.2226]}'
```

Swagger em http://localhost:8080/docs.

O `/predict` recebe os últimos `closes_esperados` fechamentos (ver `/health`), do mais antigo para o mais recente, e devolve a previsão do fechamento seguinte.
