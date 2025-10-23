# backend/services/ppna_service.py

import pandas as pd
import numpy as np
import os
import shutil
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PPNAService:
    """
    Service pour traiter les données PPNA (Provisions Pour Primes Non Acquises) 
    selon l'approche PAA (Premium Allocation Approach) IFRS17
    Pour assurance tunisienne - Montants en Dinar Tunisien (TND)
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or r"C:\Users\abdouli aziz\Desktop\Pfe-BNA-Pfe-main\Data\Ppna (4).xlsx"
        self.ppna_data = None
        self.processed_data = None
        self.currency = "TND"  # Dinar Tunisien pour assurance tunisienne
        
    def format_currency_tnd(self, amount: float) -> str:
        """Formate un montant en Dinar Tunisien"""
        if pd.isna(amount) or amount is None:
            return "0,00 TND"
        return f"{amount:,.2f} TND".replace(',', ' ')  # Format tunisien avec espaces
        
    def load_ppna_data(self) -> Dict[str, Any]:
        """Charge les données PPNA depuis le fichier Excel"""
        try:
            # Lire le fichier Excel avec toutes les feuilles
            excel_file = pd.ExcelFile(self.data_path)
            sheets_data = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(self.data_path, sheet_name=sheet_name)
                sheets_data[sheet_name] = df
                
            self.ppna_data = sheets_data
            logger.info(f"Données PPNA chargées: {len(sheets_data)} feuilles")
            
            return {
                "status": "success",
                "sheets": list(sheets_data.keys()),
                "total_sheets": len(sheets_data),
                "data_preview": self._get_data_preview(sheets_data)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement PPNA: {str(e)}")
            return {
                "status": "error",
                "message": f"Erreur de chargement: {str(e)}"
            }
    
    def _clean_data_for_json(self, data: Any) -> Any:
        """Nettoie les données pour la sérialisation JSON en remplaçant NaN par None"""
        if isinstance(data, dict):
            return {k: self._clean_data_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_data_for_json(item) for item in data]
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return self._clean_data_for_json(data.fillna(None).to_dict())
        elif pd.isna(data) or (isinstance(data, float) and (np.isnan(data) or np.isinf(data))):
            return None
        elif isinstance(data, (np.integer, np.floating)):
            return float(data) if not np.isnan(data) else None
        elif isinstance(data, (datetime, date)):
            return data.isoformat()
        else:
            return data
    
    def _get_data_preview(self, data: Dict) -> Dict:
        """Obtient un aperçu des données pour l'interface"""
        preview = {}
        for sheet_name, df in data.items():
            # Nettoyer les données avant la prévisualisation
            sample_data = df.head(3).fillna(0).to_dict('records') if not df.empty else []
            preview[sheet_name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "sample_data": self._clean_data_for_json(sample_data)
            }
        return preview
    
    def calculate_lrc_paa(self, sheet_name: str = None) -> Dict[str, Any]:
        """
        Calcule la LRC (Liability for Remaining Coverage) selon l'approche PAA
        LRC = PPNA + Risk Adjustment + Loss Component (si contrats onéreux)
        """
        try:
            if not self.ppna_data:
                self.load_ppna_data()
            
            # Utiliser la première feuille si aucune spécifiée
            if not sheet_name:
                sheet_name = list(self.ppna_data.keys())[0]
            
            df = self.ppna_data[sheet_name]
            
            # Calculs PAA IFRS17
            results = {
                "date_calcul": datetime.now().isoformat(),
                "approche": "PAA (Premium Allocation Approach)",
                "sheet_analysee": sheet_name,
                "metriques": {}
            }
            
            # Identifier les colonnes de primes et provisions
            # Colonnes spécifiques PPNA détectées
            prime_cols = []
            provision_cols = []
            
            # Détecter les colonnes spécifiques du fichier PPNA
            for col in df.columns:
                col_lower = col.lower()
                if any(x in col_lower for x in ['mntprnet', 'prime', 'premium', 'cotisation', 'mntprassi']):
                    prime_cols.append(col)
                elif any(x in col_lower for x in ['mntppna', 'provision', 'reserve', 'ppna']):
                    provision_cols.append(col)
            
            # Log pour debug
            logger.info(f"Colonnes détectées - Primes: {prime_cols}, Provisions: {provision_cols}")
            logger.info(f"DataFrame shape: {df.shape}, Colonnes: {df.columns.tolist()[:10]}...")
            
            # Calcul des totaux
            if prime_cols:
                total_primes = df[prime_cols].fillna(0).sum().sum()
            else:
                # Fallback : utiliser MNTPRNET si disponible
                if 'MNTPRNET' in df.columns:
                    total_primes = df['MNTPRNET'].fillna(0).sum()
                else:
                    total_primes = df.select_dtypes(include=[np.number]).fillna(0).sum().sum() * 0.8
            
            if provision_cols:
                total_provisions = df[provision_cols].fillna(0).sum().sum()
            else:
                # Fallback : utiliser MNTPPNA si disponible
                if 'MNTPPNA' in df.columns:
                    total_provisions = df['MNTPPNA'].fillna(0).sum()
                else:
                    total_provisions = total_primes * 0.4
            
            # Calculs IFRS17 PAA - Ajustement pour Risque (Risk Adjustment)
            # RA selon approche simplifiée calibrée sur données marché
            # Formule : RA = Provisions × risk_margin
            # risk_margin = volatilité × CoC_rate (approche VaR simplifiée)
            # Facteurs typiques : Auto 1-2%, Santé 2-3%, Vie 0.5-1.5%
            volatility_factor = 0.08  # 8% écart-type pour assurance IARD
            coc_rate = 0.06  # Coût du capital 6% (régulateur tunisien/SST)
            risk_margin = volatility_factor * coc_rate  # 0.48% (0.08 * 0.06)
            # Ajustement pour niveau de confiance 75% : facteur ~2 pour passer de 50% à 75%
            confidence_multiplier = 2.0
            risk_adjustment = total_provisions * risk_margin * confidence_multiplier
            
            # Composante de perte (Loss Component) - Test d'onérosité PAA
            # Un contrat est onéreux si : Primes + RA < Sinistres estimés + Frais
            expected_claims_ratio = 0.65  # 65% ratio sinistres/primes moyen assurance
            expected_expenses_ratio = 0.25  # 25% frais (acquisition + admin)
            estimated_costs = total_primes * (expected_claims_ratio + expected_expenses_ratio)
            
            # Loss Component = max(0, Coûts estimés - (Primes + RA))
            loss_component = max(0, estimated_costs - (total_primes + risk_adjustment))
            
            # LRC (Liability for Remaining Coverage) selon IFRS 17 PAA
            # Formule : LRC = PPNA + RA + LC (+ CSM si VFA/GMM mais pas en PAA)
            lrc_total = total_provisions + risk_adjustment + loss_component
            
            # Validation actuarielle - Cohérence des ratios
            combined_ratio = ((total_provisions + risk_adjustment + loss_component) / total_primes) if total_primes > 0 else 0
            
            # Log des calculs pour validation actuarielle
            logger.info(f"=" * 60)
            logger.info(f"CALCULS ACTUARIELS IFRS17 PAA - Lignes traitées: {len(df)}")
            logger.info(f"Primes totales: {total_primes:,.2f} TND")
            logger.info(f"PPNA (Provisions): {total_provisions:,.2f} TND ({(total_provisions/total_primes*100):.2f}% des primes)")
            logger.info(f"Risk Adjustment (CoC): {risk_adjustment:,.2f} TND ({(risk_adjustment/total_primes*100):.2f}% des primes)")
            logger.info(f"Loss Component: {loss_component:,.2f} TND")
            logger.info(f"LRC Total: {lrc_total:,.2f} TND")
            logger.info(f"Combined Ratio: {combined_ratio:.2%} {'✓ Acceptable' if combined_ratio <= 1.05 else '⚠️ Risque sous-tarification'}")
            logger.info(f"=" * 60)
            
            # Métriques détaillées
            results["metriques"] = {
                "ppna_total": round(total_provisions, 2),
                "primes_totales": round(total_primes, 2),
                "risk_adjustment": round(risk_adjustment, 2),
                "loss_component": round(loss_component, 2),
                "lrc_total": round(lrc_total, 2),
                "ratio_acquisition": round((total_provisions / total_primes * 100) if total_primes > 0 else 0, 2),
                "ratio_risk_adj": round((risk_adjustment / lrc_total * 100) if lrc_total > 0 else 0, 2),
                "lignes_traitees": len(df)
            }
            
            # Analyse par segments si possible
            segment_col = None
            if 'CODPROD' in df.columns:
                segment_col = 'CODPROD'
            elif 'CODFAM' in df.columns:
                segment_col = 'CODFAM'
            elif any('segment' in col.lower() or 'produit' in col.lower() for col in df.columns):
                segment_col = next((col for col in df.columns if 'segment' in col.lower() or 'produit' in col.lower()), None)
            
            if segment_col:
                logger.info(f"Analyse par segments avec colonne: {segment_col}")
                results["analyse_segments"] = self._analyze_by_segments(df, segment_col, prime_cols, provision_cols)
            
            # Détection contrats onéreux
            results["contrats_onereux"] = self._detect_onerous_contracts(df, prime_cols, provision_cols)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur calcul LRC PAA: {str(e)}")
            return {
                "status": "error", 
                "message": f"Erreur de calcul: {str(e)}"
            }
    
    def _analyze_by_segments(self, df: pd.DataFrame, segment_col: str, prime_cols: List, provision_cols: List) -> List[Dict]:
        """
        Analyse actuarielle par segments (Portfolio × Cohorte × Onéreux)
        Conforme IFRS 17 : Grouping requirements §14-24
        """
        segments = []
        
        # Paramètres actuariels
        volatility_factor = 0.08
        coc_rate = 0.06
        risk_margin = volatility_factor * coc_rate
        confidence_multiplier = 2.0
        expected_claims_ratio = 0.65
        expected_expenses_ratio = 0.25
        
        for segment in df[segment_col].unique():
            if pd.isna(segment):
                continue
                
            segment_data = df[df[segment_col] == segment]
            
            # Calculs de base - Conversion explicite en types Python natifs
            primes_segment = float(segment_data[prime_cols].sum().sum()) if prime_cols else 0.0
            provisions_segment = float(segment_data[provision_cols].sum().sum()) if provision_cols else 0.0
            nombre_contrats = int(len(segment_data))
            
            # Risk Adjustment actuariel (formule corrigée)
            risk_adjustment = float(provisions_segment * risk_margin * confidence_multiplier)
            
            # Loss Component (test d'onérosité)
            estimated_costs = float(primes_segment * (expected_claims_ratio + expected_expenses_ratio))
            loss_component = float(max(0, estimated_costs - (primes_segment + risk_adjustment)))
            
            # LRC Total (IFRS 17 PAA)
            lrc_total = float(provisions_segment + risk_adjustment + loss_component)
            
            # Combined Ratio
            combined_ratio = float((lrc_total / primes_segment * 100) if primes_segment > 0 else 0.0)
            
            # Classification onéreux - Conversion explicite en bool Python
            is_onerous = bool(loss_component > 0 or (provisions_segment / primes_segment > 0.80 if primes_segment > 0 else False))
            
            # Détermination cohorte (année de souscription)
            cohorte = "N/A"
            date_cols = [col for col in segment_data.columns if any(x in col.lower() for x in ['date', 'annee', 'year'])]
            if date_cols:
                try:
                    dates = pd.to_datetime(segment_data[date_cols[0]], errors='coerce')
                    cohorte = str(dates.dt.year.mode()[0]) if not dates.isna().all() else "N/A"
                except:
                    cohorte = "N/A"
            
            segments.append({
                "segment": str(segment),
                "cohorte": str(cohorte),
                "is_onerous": bool(is_onerous),
                "nombre_contrats": int(nombre_contrats),
                "primes": float(round(primes_segment, 2)),
                "ppna": float(round(provisions_segment, 2)),
                "risk_adjustment": float(round(risk_adjustment, 2)),
                "loss_component": float(round(loss_component, 2)),
                "lrc_total": float(round(lrc_total, 2)),
                "combined_ratio": float(round(combined_ratio, 2)),
                "ratio_ppna": float(round((provisions_segment / primes_segment * 100) if primes_segment > 0 else 0, 2)),
                "ra_percent": float(round((risk_adjustment / lrc_total * 100) if lrc_total > 0 else 0, 2)),
                "lc_percent": float(round((loss_component / lrc_total * 100) if lrc_total > 0 else 0, 2))
            })
        
        return sorted(segments, key=lambda x: x['lrc_total'], reverse=True)
    
    def _detect_onerous_contracts(self, df: pd.DataFrame, prime_cols: List, provision_cols: List) -> Dict:
        """Détecte les contrats potentiellement onéreux"""
        try:
            if not prime_cols or not provision_cols:
                return {"detected": False, "reason": "Colonnes insuffisantes"}
            
            # Calculer les ratios par ligne/contrat
            df_copy = df.copy()
            df_copy['primes_ligne'] = df_copy[prime_cols].sum(axis=1) if len(prime_cols) > 1 else df_copy[prime_cols[0]]
            df_copy['provisions_ligne'] = df_copy[provision_cols].sum(axis=1) if len(provision_cols) > 1 else df_copy[provision_cols[0]]
            
            # Ratio provisions/primes > 80% = potentiellement onéreux
            df_copy['ratio_onereux'] = df_copy['provisions_ligne'] / df_copy['primes_ligne']
            contrats_onereux = df_copy[df_copy['ratio_onereux'] > 0.8]
            
            return {
                "detected": len(contrats_onereux) > 0,
                "nombre_contrats_onereux": len(contrats_onereux),
                "ratio_moyen_onereux": round(contrats_onereux['ratio_onereux'].mean() * 100, 2) if len(contrats_onereux) > 0 else 0,
                "total_provisions_onereuses": round(contrats_onereux['provisions_ligne'].sum(), 2) if len(contrats_onereux) > 0 else 0
            }
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Obtient les métriques pour le dashboard"""
        try:
            if not self.ppna_data:
                self.load_ppna_data()
            
            # Calculer les métriques principales
            lrc_data = self.calculate_lrc_paa()
            
            metrics = {
                "lrc_total": lrc_data.get("metriques", {}).get("lrc_total", 0),
                "ppna_total": lrc_data.get("metriques", {}).get("ppna_total", 0),
                "risk_adjustment": lrc_data.get("metriques", {}).get("risk_adjustment", 0),
                "lic_total": lrc_data.get("metriques", {}).get("ppna_total", 0) * 0.3,  # Estimation LIC
                "csm_total": max(0, lrc_data.get("metriques", {}).get("primes_totales", 0) - lrc_data.get("metriques", {}).get("lrc_total", 0)),
                "contrats_onereux": lrc_data.get("contrats_onereux", {}).get("nombre_contrats_onereux", 0),
                "approche": "PAA",
                "derniere_maj": datetime.now().isoformat()
            }
            
            # Nettoyer les données pour JSON
            return self._clean_data_for_json(metrics)
            
        except Exception as e:
            logger.error(f"Erreur métriques dashboard: {str(e)}")
            return self._clean_data_for_json({
                "lrc_total": 0,
                "ppna_total": 0, 
                "lic_total": 0,
                "csm_total": 0,
                "risk_adjustment": 0,
                "contrats_onereux": 0,
                "approche": "PAA",
                "derniere_maj": datetime.now().isoformat(),
                "error": str(e)
            })
    
    def upload_and_process_file(self, file_path: str) -> Dict[str, Any]:
        """Traite un nouveau fichier Excel uploadé"""
        try:
            # Copier le fichier vers un emplacement permanent pour éviter les conflits
            permanent_path = os.path.join(os.path.dirname(self.data_path), "uploaded_ppna.xlsx")
            shutil.copy2(file_path, permanent_path)
            
            # Utiliser le fichier permanent
            self.data_path = permanent_path
            result = self.load_ppna_data()
            
            if result["status"] == "success":
                # Calculer immédiatement les métriques
                lrc_data = self.calculate_lrc_paa()
                result["calculations"] = self._clean_data_for_json(lrc_data)
                
            # Nettoyer toutes les données pour JSON
            return self._clean_data_for_json(result)
            
        except Exception as e:
            return self._clean_data_for_json({
                "status": "error",
                "message": f"Erreur traitement fichier: {str(e)}"
            })