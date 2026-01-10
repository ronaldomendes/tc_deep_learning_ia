# Tech Challenge 04 - Deep Learning e IA

Modelo preditivo LSTM (Long Short-Term Memory) para predizer o valor de fechamento de ações com API REST. **Suporta qualquer ticker** do Yahoo Finance (ações brasileiras como KLBN3.SA, PETR4.SA ou internacionais como AAPL, GOOGL).

## Arquitetura do Projeto

```
tc_deep_learning_ia/
├── data/
│   ├── KLBN3.SA/              # Dados da Klabin
│   │   ├── data.csv           # Dados históricos
│   │   └── scaler.pkl         # Scaler para normalização
│   ├── PETR4.SA/              # Dados da Petrobras
│   │   ├── data.csv
│   │   └── scaler.pkl
│   └── {TICKER}/              # Um diretório por ticker
├── models/
│   ├── KLBN3.SA/              # Modelo da Klabin
│   │   ├── lstm_model.pt      # Modelo treinado
│   │   ├── model_config.json  # Configurações
│   │   └── evaluation_plot.png
│   └── {TICKER}/              # Um diretório por ticker
├── src/
│   ├── __init__.py            # App FastAPI
│   ├── middleware.py          # CORS, logging e monitoramento
│   ├── mkdocs.py              # Documentação MKDocs
│   ├── utils.py               # Utilitários para paths dinâmicos
│   ├── financial/
│   │   ├── controller.py      # Rotas da API (/{ticker}/...)
│   │   ├── service.py         # Lógica de negócio
│   │   └── preprocessing.py   # Pré-processamento de dados
│   └── model/
│       ├── lstm.py            # Arquitetura LSTM
│       ├── train.py           # Treinamento
│       └── evaluate.py        # Avaliação (MAE, RMSE, MAPE)
├── notebooks/
│   ├── eda_klbn3.ipynb                    # Análise exploratória
│   └── analise_performance_modelo.ipynb   # Avaliação de performance
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

### Financial Controller (Multi-Ticker)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/v1/api/financial/{ticker}/save-data` | Baixa dados do Yahoo Finance para o ticker |
| POST | `/v1/api/financial/{ticker}/preprocessing-data` | Pré-processa os dados para treinamento |
| POST | `/v1/api/financial/{ticker}/train` | Treina o modelo LSTM |
| POST | `/v1/api/financial/{ticker}/predict` | Faz predição dos próximos 5 dias |
| GET | `/v1/api/financial/{ticker}/metrics` | Retorna métricas do modelo (MAE, RMSE, MAPE) |
| GET | `/v1/api/financial/tickers` | Lista todos os tickers com modelos treinados |

**Formato de tickers suportados:**
- **Brasileiro**: `KLBN3.SA`, `PETR4.SA`, `VALE3.SA`, `ITUB4.SA`
- **Internacional**: `AAPL`, `GOOGL`, `MSFT`, `AMZN`

### Monitoramento

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Health check |
| GET | `/metrics` | Métricas da API (uptime, requests, response times) |

## Fluxo de Uso

### Exemplo 1: Klabin (KLBN3.SA)

#### 1. Coletar dados
```bash
curl -X POST http://127.0.0.1:8000/v1/api/financial/KLBN3.SA/save-data
```

#### 2. Pré-processar dados
```bash
curl -X POST http://127.0.0.1:8000/v1/api/financial/KLBN3.SA/preprocessing-data
```

#### 3. Treinar modelo
```bash
curl -X POST http://127.0.0.1:8000/v1/api/financial/KLBN3.SA/train
```

#### 4. Fazer predição
```bash
curl -X POST http://127.0.0.1:8000/v1/api/financial/KLBN3.SA/predict
```

### Exemplo 2: Petrobras (PETR4.SA)

```bash
# Fluxo completo
curl -X POST http://127.0.0.1:8000/v1/api/financial/PETR4.SA/save-data
curl -X POST http://127.0.0.1:8000/v1/api/financial/PETR4.SA/preprocessing-data
curl -X POST http://127.0.0.1:8000/v1/api/financial/PETR4.SA/train
curl -X POST http://127.0.0.1:8000/v1/api/financial/PETR4.SA/predict
```

### Exemplo 3: Apple (AAPL)

```bash
# Ticker internacional
curl -X POST http://127.0.0.1:8000/v1/api/financial/AAPL/save-data
curl -X POST http://127.0.0.1:8000/v1/api/financial/AAPL/preprocessing-data
curl -X POST http://127.0.0.1:8000/v1/api/financial/AAPL/train
curl -X POST http://127.0.0.1:8000/v1/api/financial/AAPL/predict
```

### Listar todos os tickers disponíveis

```bash
curl -X GET http://127.0.0.1:8000/v1/api/financial/tickers
```

### Exemplo de resposta do endpoint /predict

```json
{
  "ticker": "KLBN3.SA",
  "predictions": [
    {"date": "2025-11-03", "predicted_close": 4.52},
    {"date": "2025-11-04", "predicted_close": 4.48},
    {"date": "2025-11-05", "predicted_close": 4.55},
    {"date": "2025-11-06", "predicted_close": 4.51},
    {"date": "2025-11-07", "predicted_close": 4.49}
  ],
  "model_metrics": {
    "mae": 0.0829,
    "rmse": 0.1052,
    "mape": 2.26
  },
  "last_known_date": "2025-10-31",
  "currency": "BRL"
}
```

### Exemplo de resposta do endpoint /tickers

```json
{
  "tickers": ["AAPL", "KLBN3.SA", "PETR4.SA"],
  "count": 3
}
```

## Métricas de Avaliação

- **MAE (Mean Absolute Error)**: Erro médio absoluto em R$
- **RMSE (Root Mean Square Error)**: Raiz do erro quadrático médio em R$
- **MAPE (Mean Absolute Percentage Error)**: Erro percentual médio (%)

### Performance do Modelo (KLBN3.SA - Teste)

| Métrica | Valor |
|---------|-------|
| MAE | R$ 0.0829 |
| RMSE | R$ 0.1052 |
| MAPE | 2.26% |
| R² | 0.8920 |

## Tecnologias Utilizadas

- **FastAPI**: Framework web assíncrono
- **PyTorch**: Deep learning framework
- **yfinance**: Download de dados financeiros
- **Pandas/NumPy**: Manipulação de dados
- **scikit-learn**: Normalização (MinMaxScaler)
- **Matplotlib/Plotly**: Visualizações
- **Docker**: Containerização

## Qualidade de Código

Pre-commit hooks configurados:
- **pylint**: Linting automático
- **requirements**: Auto-atualização do requirements.txt
