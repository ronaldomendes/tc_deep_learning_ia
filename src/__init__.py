"""
Main class of the application.
All configuration below is used for Controller additions, cors configuration,
Swagger and Redoc headers, etc.
"""
from fastapi import FastAPI

from src.financial.controller import financial_router
from src.middleware import register_middleware
from src.mkdocs import serve_docs

tags_metadata = [
    {
        "name": "Finance Controller",
        "description": "Methods responsible for managing data from yfinance library"
    }
]

app = FastAPI(
    version='1.0.0',
    title='Tech Challenge 04 - Collection API',
    description='A Rest API collection for FIAP - Tech Challenge',
    terms_of_service='#',
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    redoc_url='/documentation/redoc',
    docs_url='/documentation/swagger',
    openapi_url='/documentation/openapi.json',
    openapi_tags=tags_metadata
)

register_middleware(app)
app.include_router(financial_router, prefix='/v1/api/financial', tags=['Finance Controller'])
serve_docs(app)
