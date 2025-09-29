from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import transform, projection, ml_router

# =======================
# 🚀 Application FastAPI
# =======================
app = FastAPI(
    title="IFRS17 PAA Backend",
    description="API pour l'outil IFRS 17 – Approche PAA avec Machine Learning",
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
app.include_router(ml_router.router, prefix="/ml", tags=["Machine Learning"])

@app.get("/")
def root():
    return {"message": "✅ Backend IFRS17 PAA opérationnel"}
