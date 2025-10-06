from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from pathlib import Path
import uvicorn
import json

app = FastAPI()

MODEL_DIR = Path("ml/models")
TICKERS = ["VALE3_SA", "PETR4_SA"]

# Cache de modelos
models_cache = {}

@app.on_event("startup")
async def load_models():
    """Carrega modelos e métricas na memória"""
    for ticker in TICKERS:
        model_path = MODEL_DIR / f"model_{ticker}.joblib"
        metrics_path = MODEL_DIR / f"metrics_{ticker}.json"
        
        if model_path.exists():
            models_cache[ticker] = {
                "model": joblib.load(model_path),
                "metrics": json.load(open(metrics_path)) if metrics_path.exists() else None
            }

@app.get("/model-metrics/{ticker}")
async def get_metrics(ticker: str):
    """Retorna métricas detalhadas do modelo"""
    if ticker not in models_cache:
        return JSONResponse(
            {"error": f"Modelo {ticker} não encontrado"},
            status_code=404
        )
    
    metrics = models_cache[ticker].get("metrics", {})
    return {
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"),
        "last_training_date": metrics.get("last_date")
    }

# ... (mantenha os outros endpoints existentes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)