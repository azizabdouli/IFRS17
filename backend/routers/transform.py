from fastapi import APIRouter, UploadFile, File
import pandas as pd
from backend.services.data_mapper import prepare_ppna_data

router = APIRouter()

@router.post("/")
async def transform(file: UploadFile = File(...)):
    """
    🔄 Transformation du fichier PPNA en données IFRS17 PAA
    """
    try:
        # Lecture fichier en mémoire
        content = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(pd.io.common.BytesIO(content))
        else:
            df = pd.read_excel(pd.io.common.BytesIO(content))

        df_out = prepare_ppna_data(df)
        df_out = df_out.fillna(0)  # ✅ Pas de NaN pour JSON

        return {
            "rows": len(df_out),
            "data": df_out.head(100).to_dict(orient="records")  # Limite pour rapidité
        }

    except Exception as e:
        return {"error": f"Erreur transformation : {e}"}
