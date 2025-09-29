import pandas as pd
import numpy as np

def monthly_projection_exact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Projection mensuelle EXACTE par contrat :
    - génère 'duree_mois' périodes MS à partir du mois d'effet (début de mois),
    - répartit la prime lissée selon pattern 12 mois si disponible (sinon uniforme),
    - amortit DAC au même pattern (ici uniforme par défaut),
    - retourne colonnes: [ID (si dispo), mois, revenue_mois, dac_amort_mois, CODPROD, Cohorte, Onereux]
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["mois", "revenue_mois", "dac_amort_mois"])

    work = df.copy()

    # Bases
    work['date_effet'] = to_datetime_safe(work['date_effet'])
    work['duree_mois'] = pd.to_numeric(work['duree_mois'], errors='coerce').fillna(0).astype(int)
    work = work[work['duree_mois'] > 0].copy()

    # Identifiant contrat si dispo
    id_col = None
    for c in ['NUMQUITT', 'NUMCONTRAT', 'ID_CONTRAT']:
        if c in work.columns:
            id_col = c
            break

    # Cohorte & groupe onéreux
    work['Cohorte'] = work['date_effet'].dt.year
    work['Onereux'] = work['lrc'] < 0 if 'lrc' in work else False

    # Pattern
    pat_cols = [f"M{i}" for i in range(1, 13)]
    for c in pat_cols:
        if c not in work.columns:
            work[c] = np.nan
    pattern_arr = work.apply(normalise_pattern, axis=1)

    # Prime >= 0 (reconnaissance de service)
    prime_pos = work['prime_brute'].clip(lower=0).fillna(0.0).values.astype(float)
    # DAC (si absente, calcule via DAC_pct si fourni)
    if 'dac' not in work.columns:
        if 'DAC_pct' in work.columns:
            work['dac'] = prime_pos * pd.to_numeric(work['DAC_pct'], errors='coerce').fillna(0.10)
        else:
            work['dac'] = prime_pos * 0.10

    rows = []
    for idx, r in work.iterrows():
        start_m = r['date_effet'].to_period('M').to_timestamp()  # début du mois
        n = int(r['duree_mois'])
        if n <= 0 or pd.isna(start_m):
            continue

        pat12 = pattern_arr.iloc[idx]
        if n <= 12:
            pat = pat12[:n]
        else:
            k = n // 12
            rem = n % 12
            pat = np.concatenate([np.tile(pat12, k), pat12[:rem]])
        pat = pat / pat.sum()  # re-normalise

        rev_mois = (prime_pos[idx] * pat).astype(float)
        dac_amort = (float(r['dac']) * (np.ones(n)/n)).astype(float)  # DAC amortie uniforme

        months = pd.date_range(start=start_m, periods=n, freq='MS')
        for m, rv, da in zip(months, rev_mois, dac_amort):
            out = {
                'mois': m,
                'revenue_mois': rv,
                'dac_amort_mois': da,
                'CODPROD': r.get('CODPROD', None),
                'Cohorte': r.get('Cohorte', None),
                'Onereux': r.get('Onereux', False)
            }
            if id_col:
                out[id_col] = r[id_col]
            rows.append(out)

    proj = pd.DataFrame(rows)
    df = df.replace({np.nan: None})
    return proj