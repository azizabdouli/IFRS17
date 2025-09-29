from fastapi import APIRouter, UploadFile, File
import pandas as pd
from backend.services.data_mapper import prepare_ppna_data
from backend.services.projection_service import monthly_projection_exact

router = APIRouter()

@router.post("/")
async def projection(file: UploadFile = File(...)):
    """
    📊 Projection mensuelle EXACTE IFRS17
    """
    try:
        content = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(pd.io.common.BytesIO(content))
        else:
            df = pd.read_excel(pd.io.common.BytesIO(content))

        df = prepare_ppna_data(df)
        df_proj = monthly_projection_exact(df)
        df_proj = df_proj.fillna(0)  # ✅ Pour éviter les NaN

        return {
            "rows": len(df_proj),
            "data": df_proj.head(100).to_dict(orient="records")
        }

    except Exception as e:
        return {"error": f"Erreur projection : {e}"}
