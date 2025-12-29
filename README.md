# Tech Challenge 04 - Deep Learning e IA

Modelo preditivo LSTM (Long Short-Term Memory) para predizer o valor de fechamento de ações da Klabin (KLBN3.SA) com API REST.

## Arquitetura do Projeto

```
tc_deep_learning_ia/
├── data/
│   ├── KLBN_data.csv          # Dados históricos
│   └── scaler.pkl             # Scaler para normalização
├── models/
│   ├── lstm_model.pt          # Modelo treinado
│   ├── model_config.json      # Configurações do modelo
│   └── evaluation_plot.png    # Gráfico de avaliação
├── src/
│   ├── __init__.py            # App FastAPI
│   ├── middleware.py          # CORS, logging e monitoramento
│   ├── mkdocs.py              # Documentação MKDocs
│   ├── financial/
│   │   ├── controller.py      # Rotas da API
│   │   ├── service.py         # Lógica de negócio
│   │   └── preprocessing.py   # Pré-processamento de dados
│   └── model/
│       ├── lstm.py            # Arquitetura LSTM
│       ├── train.py           # Treinamento
│       └── evaluate.py        # Avaliação (MAE, RMSE, MAPE)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Modelo LSTM

### Arquitetura
```
Input (60 dias, 6 features) → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Linear(5)
```

### Features utilizadas
- **Close**: Preço de fechamento
- **Volume**: Volume negociado
- **SMA_7**: Média móvel de 7 dias
- **SMA_21**: Média móvel de 21 dias
- **Returns**: Retorno percentual diário
- **Volatility**: Volatilidade (desvio padrão rolling)

### Hiperparâmetros
- Janela de entrada: 60 dias
- Horizonte de predição: 5 dias
- Batch size: 32
- Learning rate: 0.001
- Early stopping: patience=10

## Instalação

### Pré-requisitos
- Python 3.11+
- pip

### Instalação Local

```bash
# Clonar repositório
git clone <repo-url>
cd tc_deep_learning_ia

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente (Windows)
.\venv\Scripts\activate

# Ativar ambiente (Linux/Mac)
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Instalar hooks pre-commit
pre-commit install
```

### Instalação com Docker

```bash
# Build e execução
docker-compose up --build

# Ou apenas build
docker build -t lstm-api .
docker run -p 8000:8000 -v ./data:/app/data -v ./models:/app/models lstm-api
```

## Execução

### Local

```bash
# Modo desenvolvimento
uvicorn src:app --reload --host 127.0.0.1 --port 8000

# Ou usando fastapi-cli (pode ter problemas de encoding no Windows)
fastapi dev src/
```

### Docker

```bash
docker-compose up
```

A API estará disponível em: http://127.0.0.1:8000

## Documentação da API

- **Swagger UI**: http://127.0.0.1:8000/documentation/swagger
- **ReDoc**: http://127.0.0.1:8000/documentation/redoc
- **OpenAPI JSON**: http://127.0.0.1:8000/documentation/openapi.json

## Endpoints

### Financial Controller

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/v1/api/financial/save-data` | Baixa dados do Yahoo Finance e salva em CSV |
| POST | `/v1/api/financial/preprocessing-data` | Pré-processa os dados para treinamento |
| POST | `/v1/api/financial/train` | Treina o modelo LSTM |
| POST | `/v1/api/financial/predict` | Faz predição dos próximos 5 dias |
| GET | `/v1/api/financial/metrics` | Retorna métricas do modelo (MAE, RMSE, MAPE) |

### Monitoramento

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Health check |
| GET | `/metrics` | Métricas da API (uptime, requests, response times) |

## Fluxo de Uso

### 1. Coletar dados
```bash
curl -X POST http://127.0.0.1:8000/v1/api/financial/save-data
```

### 2. Pré-processar dados
```bash
curl -X POST http://127.0.0.1:8000/v1/api/financial/preprocessing-data
```

### 3. Treinar modelo
```bash
curl -X POST http://127.0.0.1:8000/v1/api/financial/train
```

### 4. Fazer predição
```bash
curl -X POST http://127.0.0.1:8000/v1/api/financial/predict
```

### Exemplo de resposta do endpoint /predict
```json
{
  "predictions": [
    {"date": "2025-11-03", "predicted_close": 4.52},
    {"date": "2025-11-04", "predicted_close": 4.48},
    {"date": "2025-11-05", "predicted_close": 4.55},
    {"date": "2025-11-06", "predicted_close": 4.51},
    {"date": "2025-11-07", "predicted_close": 4.49}
  ],
  "model_metrics": {
    "mae": 0.12,
    "rmse": 0.15,
    "mape": 2.8
  },
  "last_known_date": "2025-10-31",
  "currency": "BRL"
}
```

## Métricas de Avaliação

- **MAE (Mean Absolute Error)**: Erro médio absoluto em R$
- **RMSE (Root Mean Square Error)**: Raiz do erro quadrático médio em R$
- **MAPE (Mean Absolute Percentage Error)**: Erro percentual médio (%)

## Tecnologias Utilizadas

- **FastAPI**: Framework web assíncrono
- **PyTorch**: Deep learning framework
- **yfinance**: Download de dados financeiros
- **Pandas/NumPy**: Manipulação de dados
- **scikit-learn**: Normalização (MinMaxScaler)
- **Matplotlib**: Visualizações
- **Docker**: Containerização
