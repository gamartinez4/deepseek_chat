"""Main entry point for the application."""
from fastapi import FastAPI

from .server import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 