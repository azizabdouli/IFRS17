from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import transform, projection

# =======================
# 🚀 Application FastAPI
# =======================
app = FastAPI(
    title="IFRS17 PAA Backend",
    description="API pour l'outil IFRS 17 – Approche PAA",
    version="1.0.0"
)

# Autoriser le frontend Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["http://localhost:8501"] si tu veux restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# 🛣️ Routes principales
# =======================
app.include_router(transform.router, prefix="/transform", tags=["Transform"])
app.include_router(projection.router, prefix="/projection", tags=["Projection"])

@app.get("/")
def root():
    return {"message": "✅ Backend IFRS17 PAA opérationnel"}
