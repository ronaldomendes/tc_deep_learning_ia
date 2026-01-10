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