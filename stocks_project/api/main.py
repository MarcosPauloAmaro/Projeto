from fastapi import FastAPI, HTTPException
from pathlib import Path
import json
import os
from joblib import load
import uvicorn

# Configuração ABSOLUTA (ajuste para seu PC)
MODEL_DIR = Path(r"C:\Users\marck\Documents\Projeto\acoes\ml\models")
TICKERS = ["VALE3_SA", "PETR4_SA"]

app = FastAPI()

@app.get("/models")
async def get_models():
    """Lista modelos disponíveis"""
    models = {}
    
    for ticker in TICKERS:
        model_path = MODEL_DIR / f"model_{ticker}.joblib"
        metrics_path = MODEL_DIR / f"metrics_{ticker}.json"
        
        if model_path.exists():
            model_info = {"path": str(model_path)}
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    model_info["metrics"] = json.load(f)
            models[ticker] = model_info
    
    if not models:
        raise HTTPException(
            status_code=404,
            detail=f"""
            Nenhum modelo encontrado em {MODEL_DIR}.
            Execute o treinamento com: python ml/train_models.py
            Arquivos esperados: model_VALE3_SA.joblib, model_PETR4_SA.joblib
            """
        )
    return {"models": models}

if __name__ == "__main__":
    print(f"🔍 Procurando modelos em: {MODEL_DIR}")
    print(f"📦 Conteúdo do diretório: {os.listdir(MODEL_DIR) if MODEL_DIR.exists() else 'DIRETÓRIO NÃO EXISTE'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )