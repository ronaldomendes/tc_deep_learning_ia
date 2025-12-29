## Documentação da API

- **Swagger UI**: [http://127.0.0.1:8000/documentation/swagger](http://127.0.0.1:8000/documentation/swagger)
- **ReDoc**: [http://127.0.0.1:8000/documentation/redoc](http://127.0.0.1:8000/documentation/redoc)
- **OpenAPI JSON**: [http://127.0.0.1:8000/documentation/openapi.json](http://127.0.0.1:8000/documentation/openapi.json)

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
