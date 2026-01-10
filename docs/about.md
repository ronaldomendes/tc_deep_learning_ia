# Sobre o desafio do Tech Challenge 4

## O problema
Seu desafio é criar um modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de 
fechamento da bolsa de valores de uma empresa à sua escolha e realizar toda a pipeline de desenvolvimento, 
desde a criação do modelo preditivo até o deploy do modelo em uma API que permita a previsão de preços de ações.

Seu Tech Challenge precisa seguir os seguintes requisitos:

### **Coleta e Pré-processamento dos Dados**

- **Coleta de Dados:** utilize um dataset de preços históricos de ações, como o Yahoo Finance ou qualquer outro dataset 
financeiro disponível (dica: utilize a biblioteca **[yfinance](https://ranaroussi.github.io/yfinance/reference/index.html)**). Veja um exemplo a seguir:
```{.py3 linenums=1}
import yfinance as yf 
# Especifique o símbolo da empresa que você vai trabalhar 
# Configure data de início e fim da sua base 

symbol = 'DIS' 
start_date = '2018-01-01' 
end_date = '2024-07-20' 

# Use a função download para obter os dados 
df = yf.download(symbol, start=start_date, end=end_date)
```

### **Desenvolvimento do Modelo LSTM**
- **Construção do Modelo:** implemente um modelo de deep learning utilizando LSTM para capturar padrões temporais nos 
dados de preços das ações.
- **Treinamento:** treine o modelo utilizando uma parte dos dados e ajuste os hiperparâmetros para otimizar o desempenho.
- **Avaliação:** avalie o modelo utilizando dados de validação e utilize métricas como MAE (Mean Absolute Error), 
RMSE (Root Mean Square Error), MAPE (Erro Percentual Absoluto Médio) ou outra métrica apropriada para medir 
a precisão das previsões.

### **Salvamento e Exportação do Modelo**
- **Salvar o Modelo:** após atingir um desempenho satisfatório, salve o modelo treinado em um formato que 
possa ser utilizado para inferência.

### **Deploy do Modelo**
- **Criação da API:** desenvolva uma API RESTful utilizando Flask ou FastAPI para servir o modelo. A API deve permitir 
que o usuário forneça dados históricos de preços e receba previsões dos preços futuros.

### **Escalabilidade e Monitoramento**
- **Monitoramento:** configure ferramentas de monitoramento para rastrear a performance do modelo em produção, 
incluindo tempo de resposta e utilização de recursos.

### **Entregáveis**
- Código-fonte do modelo LSTM no seu repositório do GIT + documentação do projeto.
- Scripts ou contêineres Docker para deploy da API.
- Link para a API em produção, caso tenha sido deployada em um ambiente de nuvem.
- Vídeo mostrando e explicando todo o funcionamento da API.

Este desafio permitirá que você demonstre habilidades avançadas em deep learning, especificamente no uso de LSTM para 
séries temporais, bem como em práticas de deploy em ambientes de produção. Boa sorte e conte conosco caso tenha alguma 
dúvida no desenvolvimento do projeto!