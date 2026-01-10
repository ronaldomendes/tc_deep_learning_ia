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

- **Docker**: Containerização
- **FastAPI**: Framework web assíncrono
- **Matplotlib**: Visualizações
- **Pandas/NumPy**: Manipulação de dados
- **PyTorch**: Deep learning framework
- **scikit-learn**: Normalização (MinMaxScaler)
- **yfinance**: Download de dados financeiros
