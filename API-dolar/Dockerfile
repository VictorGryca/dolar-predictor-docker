# Estágio 1: treino. Baixa os dados, treina e grava o artefato em /out.
FROM python:3.11-slim AS train
WORKDIR /train
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt
COPY features.py train.py ./
COPY data/ ./data/
# Mudar TRAIN_DATE invalida o cache a partir daqui e força o retreino.
# Fica depois do pip install para não reinstalar as dependências.
ARG TRAIN_DATE=manual
RUN echo "TRAIN_DATE=${TRAIN_DATE}" && python train.py --out /out

# Estágio 2: inferência. Só as dependências da API e o artefato do estágio 1.
FROM python:3.11-slim AS serve
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY features.py app.py ./
COPY --from=train /out/ ./models/
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
