from fastapi import FastAPI

from src.financial.controller import financial_router
from src.middleware import register_middleware

app = FastAPI(
    title="Tech Challenge 04 - Collection API",
    version="1.0.0",
    docs_url="/documentation/swagger",
    redoc_url="/documentation/redoc",
    openapi_url="/documentation/openapi.json"
)

@app.get("/health")
def health():
    return {"status": "ok"}

register_middleware(app)

app.include_router(
    financial_router,
    prefix="/v1/api/financial",
    tags=["Finance Controller"]
)
