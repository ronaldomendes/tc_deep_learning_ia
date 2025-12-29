"""
MKdocs server configuration
"""
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def serve_docs(app: FastAPI):
    """
    I'm adding this configuration below to make it possible
    a visualization of mkdocs documentation page
    """
    app.mount("/site", StaticFiles(directory="site"), name="docs")
    app.mount("/assets", StaticFiles(directory='site/assets'), name='assets')
    app.mount("/search", StaticFiles(directory='site/search'), name='search')

    @app.get("/{full_path:path}", include_in_schema=False)
    def mkdocs(full_path: str):
        """Exposing api documentation and excluding from swagger and redoc"""
        base_dir = "site"
        target = os.path.join(base_dir, full_path)

        if os.path.isdir(target):
            target = os.path.join(target, "index.html")

        if os.path.exists(target):
            return FileResponse(target)
        return FileResponse(os.path.join(base_dir, "index.html"))
