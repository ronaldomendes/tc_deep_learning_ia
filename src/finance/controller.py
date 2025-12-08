"""
Finance Controller
"""
from fastapi import APIRouter, status
from starlette.responses import JSONResponse

finance_router = APIRouter()


@finance_router.get('/hello')
async def say_hello():
    """Just a hello controller"""
    return JSONResponse(
        content={'message': 'hello from FastAPI'},
        status_code=status.HTTP_418_IM_A_TEAPOT
    )
