"""
Middleware class: responsible for Cors configuration, logging and monitoring
"""
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger('api')

# Disable default uvicorn access logs
uvicorn_logger = logging.getLogger('uvicorn.access')
uvicorn_logger.disabled = True


class APIMetrics:
    """Class to track API metrics"""

    def __init__(self):
        self.request_count: Dict[str, int] = defaultdict(int)
        self.total_requests: int = 0
        self.error_count: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, list] = defaultdict(list)
        self.start_time: datetime = datetime.now()

    def record_request(self, endpoint: str, status_code: int, response_time: float):
        """Record a request metric"""
        self.total_requests += 1
        self.request_count[endpoint] += 1
        self.response_times[endpoint].append(response_time)

        if status_code >= 400:
            error_type = 'client_error' if status_code < 500 else 'server_error'
            self.error_count[error_type] += 1

    def get_summary(self) -> dict:
        """Get metrics summary"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        avg_response_times = {
            endpoint: round(sum(times) / len(times) * 1000, 2)
            for endpoint, times in self.response_times.items()
            if times
        }

        return {
            'uptime_seconds': round(uptime, 2),
            'total_requests': self.total_requests,
            'requests_per_endpoint': dict(self.request_count),
            'error_count': dict(self.error_count),
            'avg_response_time_ms': avg_response_times
        }


# Global metrics instance
metrics = APIMetrics()


def register_middleware(app: FastAPI):
    """Register all middleware for the application"""

    @app.middleware('http')
    async def monitoring_middleware(req: Request, call_next):
        """Middleware for logging and monitoring"""
        start_time = time.time()

        # Process request
        try:
            response = await call_next(req)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            logger.error(json.dumps({
                'level': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'path': req.url.path,
                'error': str(e)
            }))
            raise

        # Calculate response time
        response_time = time.time() - start_time

        # Record metrics
        metrics.record_request(req.url.path, status_code, response_time)

        # Structured logging (JSON format)
        log_data = {
            'level': 'INFO' if status_code < 400 else 'WARNING',
            'timestamp': datetime.now().isoformat(),
            'client_ip': req.client.host if req.client else 'unknown',
            'method': req.method,
            'path': req.url.path,
            'status_code': status_code,
            'response_time_ms': round(response_time * 1000, 2)
        }

        logger.info(json.dumps(log_data))

        return response

    # Metrics endpoint
    @app.get('/metrics', tags=['Monitoring'], include_in_schema=True)
    async def get_api_metrics():
        """Get API performance metrics"""
        return JSONResponse(content=metrics.get_summary())

    # Health check endpoint
    @app.get('/health', tags=['Monitoring'], include_in_schema=True)
    async def health_check():
        """Health check endpoint"""
        return JSONResponse(content={
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
        allow_credentials=True
    )
