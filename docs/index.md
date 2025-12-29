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
