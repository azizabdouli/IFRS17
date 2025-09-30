#!/usr/bin/env python3
"""
Diagnostic des colonnes problématiques
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au PATH
sys.path.append(str(Path(__file__).parent))

def analyze_columns():
    """Analyse des colonnes pour identifier les problèmes"""
    print("🔍 Analyse des colonnes...")
    
    try:
        # Charger échantillon de données
        data_path = "Data/Ppna (4).xlsx"
        df = pd.read_excel(data_path)
        sample_df = df.sample(n=100, random_state=42)  # Petit échantillon
        print(f"📊 Échantillon: {len(sample_df)} lignes, {len(sample_df.columns)} colonnes")
        
        print("\n📋 Types de colonnes:")
        for col in sample_df.columns:
            dtype = sample_df[col].dtype
            unique_count = sample_df[col].nunique()
            sample_values = sample_df[col].dropna().head(3).tolist()
            print(f"   {col}: {dtype} | {unique_count} valeurs uniques | Ex: {sample_values}")
        
        # Identifier les colonnes problématiques
        print("\n⚠️ Colonnes suspectes:")
        for col in sample_df.columns:
            if sample_df[col].dtype == 'object':
                # Vérifier si contient des chaînes très longues
                max_length = sample_df[col].astype(str).str.len().max()
                if max_length > 50:
                    print(f"   {col}: Chaînes très longues (max: {max_length})")
                
                # Vérifier si contient des patterns 'R'
                contains_r_pattern = sample_df[col].astype(str).str.contains('RRR', na=False).any()
                if contains_r_pattern:
                    print(f"   {col}: Contient des patterns 'RRR'")
                    example = sample_df[col].astype(str).str.contains('RRR', na=False)
                    if example.any():
                        print(f"      Exemple: {sample_df[col][example].iloc[0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    analyze_columns()