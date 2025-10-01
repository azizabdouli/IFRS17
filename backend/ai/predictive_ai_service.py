# backend/ai/predictive_ai_service.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache
import json

# ML avancé
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

logger = logging.getLogger(__name__)

class PredictiveAIService:
    """
    Service IA prédictif avancé pour IFRS17
    avec auto-ML et recommandations intelligentes
    """
    
    def __init__(self):
        self.models = {}
        self.predictions_cache = {}
        self.ai_insights = {}
        self.auto_ml_config = self._init_auto_ml_config()
        
        logger.info("🧠 Service IA Prédictif initialisé")
    
    def _init_auto_ml_config(self) -> Dict[str, Any]:
        """Configuration Auto-ML"""
        return {
            "algorithms": {
                "regression": ["random_forest", "xgboost", "lightgbm"],
                "classification": ["random_forest", "svm", "neural_network"],
                "clustering": ["kmeans", "dbscan", "hierarchical"]
            },
            "auto_optimize": True,
            "cross_validation": 5,
            "performance_threshold": {
                "regression": 0.7,  # R²
                "classification": 0.8,  # Accuracy
                "clustering": 0.6  # Silhouette score
            }
        }
    
    async def auto_analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyse automatique intelligente du dataset
        avec recommandations IA
        """
        logger.info("🔍 Analyse automatique IA du dataset...")
        
        analysis = {
            "data_quality": await self._assess_data_quality(df),
            "feature_importance": await self._analyze_feature_importance(df),
            "patterns": await self._detect_patterns(df),
            "recommendations": [],
            "ai_suggestions": []
        }
        
        # Générer recommandations intelligentes
        analysis["recommendations"] = self._generate_smart_recommendations(analysis)
        analysis["ai_suggestions"] = self._generate_ai_suggestions(df, analysis)
        
        return analysis
    
    async def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Évaluation de la qualité des données avec IA"""
        quality = {
            "score": 0.0,
            "issues": [],
            "strengths": [],
            "completeness": 0.0,
            "consistency": 0.0,
            "validity": 0.0
        }
        
        # Complétude
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        quality["completeness"] = max(0, 1 - missing_ratio)
        
        # Cohérence
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Détecter les outliers
            outlier_ratio = 0
            for col in numeric_cols[:5]:  # Limiter pour performance
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                outlier_ratio += len(outliers) / len(df)
            
            quality["consistency"] = max(0, 1 - outlier_ratio / len(numeric_cols))
        else:
            quality["consistency"] = 0.8
        
        # Validité (règles métier IFRS17)
        validity_score = self._check_ifrs17_validity(df)
        quality["validity"] = validity_score
        
        # Score global
        quality["score"] = (quality["completeness"] + quality["consistency"] + quality["validity"]) / 3
        
        # Issues et forces
        if quality["completeness"] < 0.9:
            quality["issues"].append(f"Données manquantes: {missing_ratio:.1%}")
        else:
            quality["strengths"].append("Complétude excellente")
        
        if quality["consistency"] < 0.8:
            quality["issues"].append("Outliers détectés")
        else:
            quality["strengths"].append("Données cohérentes")
        
        return quality
    
    def _check_ifrs17_validity(self, df: pd.DataFrame) -> float:
        """Vérification des règles métier IFRS17"""
        validity_checks = []
        
        # Vérifier les primes positives
        if "prime_brute" in df.columns:
            positive_primes = (df["prime_brute"] > 0).sum() / len(df)
            validity_checks.append(positive_primes)
        
        # Vérifier les dates cohérentes
        if "date_effet" in df.columns:
            try:
                dates = pd.to_datetime(df["date_effet"], errors='coerce')
                valid_dates = dates.notna().sum() / len(df)
                validity_checks.append(valid_dates)
            except:
                validity_checks.append(0.5)
        
        # Vérifier la durée positive
        if "duree_mois" in df.columns:
            positive_duration = (df["duree_mois"] > 0).sum() / len(df)
            validity_checks.append(positive_duration)
        
        return np.mean(validity_checks) if validity_checks else 0.8
    
    async def _analyze_feature_importance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse de l'importance des features avec ML"""
        try:
            # Préparer les données numériques
            numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
            
            if len(numeric_df.columns) < 2:
                return {"importance": {}, "top_features": []}
            
            # Utiliser Random Forest pour l'importance
            if "prime_brute" in numeric_df.columns:
                target = "prime_brute"
                features = [col for col in numeric_df.columns if col != target]
                
                if len(features) > 0:
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(numeric_df[features], numeric_df[target])
                    
                    importance_dict = dict(zip(features, rf.feature_importances_))
                    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    return {
                        "importance": importance_dict,
                        "top_features": [{"feature": f, "importance": round(i, 3)} for f, i in top_features]
                    }
            
            return {"importance": {}, "top_features": []}
            
        except Exception as e:
            logger.error(f"Erreur analyse importance: {e}")
            return {"importance": {}, "top_features": []}
    
    async def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Détection de patterns avec IA"""
        patterns = {
            "temporal": {},
            "seasonal": {},
            "correlations": {},
            "anomalies": {}
        }
        
        try:
            # Patterns temporels
            if "date_effet" in df.columns:
                dates = pd.to_datetime(df["date_effet"], errors='coerce')
                if dates.notna().any():
                    patterns["temporal"] = {
                        "date_range": f"{dates.min()} à {dates.max()}",
                        "years_span": (dates.max() - dates.min()).days / 365.25,
                        "monthly_distribution": dates.dt.month.value_counts().to_dict()
                    }
            
            # Corrélations importantes
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr.append({
                                "var1": corr_matrix.columns[i],
                                "var2": corr_matrix.columns[j],
                                "correlation": round(corr_val, 3)
                            })
                patterns["correlations"] = high_corr[:5]  # Top 5
            
            # Détection d'anomalies
            if len(numeric_df.columns) >= 2:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_scores = iso_forest.fit_predict(numeric_df.fillna(0))
                anomaly_count = (anomaly_scores == -1).sum()
                patterns["anomalies"] = {
                    "count": int(anomaly_count),
                    "percentage": round(anomaly_count / len(df) * 100, 2)
                }
        
        except Exception as e:
            logger.error(f"Erreur détection patterns: {e}")
        
        return patterns
    
    def _generate_smart_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Génération de recommandations intelligentes"""
        recommendations = []
        
        # Basé sur la qualité des données
        quality = analysis["data_quality"]
        if quality["score"] < 0.7:
            recommendations.append("🔧 Améliorer la qualité des données avant l'analyse")
        
        if quality["completeness"] < 0.9:
            recommendations.append("📝 Traiter les valeurs manquantes")
        
        if quality["consistency"] < 0.8:
            recommendations.append("🎯 Identifier et traiter les outliers")
        
        # Basé sur les features
        if analysis["feature_importance"]["top_features"]:
            top_feature = analysis["feature_importance"]["top_features"][0]
            recommendations.append(f"📊 Focus sur '{top_feature['feature']}' (importance: {top_feature['importance']})")
        
        # Basé sur les patterns
        patterns = analysis["patterns"]
        if patterns["anomalies"].get("percentage", 0) > 5:
            recommendations.append("🚨 Investiguer les anomalies détectées")
        
        if len(patterns["correlations"]) > 0:
            recommendations.append("🔗 Explorer les corrélations fortes détectées")
        
        # Recommandations ML
        recommendations.append("🤖 Lancer le clustering pour segmentation")
        recommendations.append("📈 Utiliser les modèles prédictifs")
        
        return recommendations
    
    def _generate_ai_suggestions(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggestions IA avancées"""
        suggestions = []
        
        # Suggestion de modèles basée sur les données
        if len(df) > 1000:
            suggestions.append({
                "type": "model",
                "title": "🎯 Modèle de Clustering recommandé",
                "description": "Dataset volumineux détecté - clustering DBSCAN optimal",
                "action": "run_clustering",
                "priority": "high",
                "params": {"algorithm": "dbscan", "min_samples": 50}
            })
        
        # Suggestion basée sur les features importantes
        if analysis["feature_importance"]["top_features"]:
            suggestions.append({
                "type": "analysis",
                "title": "📊 Analyse de rentabilité",
                "description": "Features importantes détectées pour prédiction rentabilité",
                "action": "profitability_analysis",
                "priority": "medium"
            })
        
        # Suggestion temporelle
        if analysis["patterns"]["temporal"]:
            suggestions.append({
                "type": "temporal",
                "title": "📅 Analyse temporelle",
                "description": "Patterns temporels détectés - analyse de saisonnalité recommandée",
                "action": "temporal_analysis",
                "priority": "low"
            })
        
        return suggestions
    
    async def smart_model_selection(self, df: pd.DataFrame, task_type: str) -> Dict[str, Any]:
        """Sélection intelligente de modèle avec Auto-ML"""
        logger.info(f"🧠 Sélection intelligente de modèle pour: {task_type}")
        
        model_recommendation = {
            "recommended_algorithm": None,
            "performance_estimate": 0.0,
            "reasoning": "",
            "alternatives": []
        }
        
        # Analyse du dataset pour recommandation
        data_characteristics = {
            "size": len(df),
            "features": len(df.columns),
            "missing_ratio": df.isnull().sum().sum() / (len(df) * len(df.columns)),
            "numeric_ratio": len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        }
        
        # Logique de recommandation intelligente
        if task_type == "clustering":
            if data_characteristics["size"] > 10000:
                model_recommendation["recommended_algorithm"] = "dbscan"
                model_recommendation["reasoning"] = "DBSCAN optimal pour gros datasets avec détection automatique du nombre de clusters"
            else:
                model_recommendation["recommended_algorithm"] = "kmeans"
                model_recommendation["reasoning"] = "K-means efficace pour datasets moyens"
        
        elif task_type == "prediction":
            if data_characteristics["features"] > 20:
                model_recommendation["recommended_algorithm"] = "xgboost"
                model_recommendation["reasoning"] = "XGBoost excellent pour nombreuses features"
            else:
                model_recommendation["recommended_algorithm"] = "random_forest"
                model_recommendation["reasoning"] = "Random Forest robuste et interprétable"
        
        # Performance estimée basée sur les caractéristiques
        base_performance = 0.75
        if data_characteristics["missing_ratio"] < 0.1:
            base_performance += 0.1
        if data_characteristics["numeric_ratio"] > 0.8:
            base_performance += 0.05
        
        model_recommendation["performance_estimate"] = min(base_performance, 0.95)
        
        return model_recommendation
    
    async def generate_ai_insights(self, df: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des insights IA à partir des résultats"""
        insights = {
            "key_findings": [],
            "business_impact": [],
            "recommendations": [],
            "risk_alerts": [],
            "opportunities": []
        }
        
        # Analyse des résultats pour insights
        if "cluster_labels" in results:
            n_clusters = len(np.unique(results["cluster_labels"]))
            insights["key_findings"].append(f"📊 {n_clusters} segments distincts identifiés")
            
            if n_clusters > 5:
                insights["business_impact"].append("🎯 Segmentation fine permet personnalisation marketing")
            
        # Alertes risque basées sur anomalies
        if "anomalous_contracts" in results:
            anomaly_count = len(results["anomalous_contracts"])
            if anomaly_count > len(df) * 0.05:  # Plus de 5%
                insights["risk_alerts"].append(f"⚠️ {anomaly_count} contrats anomaliques détectés - révision recommandée")
        
        # Opportunités basées sur performance
        if "predictions" in results:
            high_profit_contracts = np.array(results["predictions"]) > np.percentile(results["predictions"], 75)
            if high_profit_contracts.any():
                insights["opportunities"].append("💰 Contrats haute rentabilité identifiés pour expansion")
        
        return insights
    
    @lru_cache(maxsize=16)
    def get_model_explanation(self, model_type: str) -> str:
        """Explications des modèles en cache"""
        explanations = {
            "xgboost": "🚀 XGBoost: Gradient boosting optimisé, excellent pour prédictions précises",
            "random_forest": "🌲 Random Forest: Ensemble d'arbres, robuste et interprétable",
            "dbscan": "🔍 DBSCAN: Clustering basé densité, détecte automatiquement outliers",
            "kmeans": "📊 K-Means: Clustering centroïde, rapide et efficace"
        }
        return explanations.get(model_type, "Modèle ML avancé")
    
    def save_ai_state(self, filepath: str):
        """Sauvegarde l'état IA"""
        state = {
            "models": {k: v for k, v in self.models.items() if k != "model_object"},
            "insights": self.ai_insights,
            "config": self.auto_ml_config,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 État IA sauvegardé: {filepath}")
    
    def load_ai_state(self, filepath: str):
        """Charge l'état IA"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.ai_insights = state.get("insights", {})
            self.auto_ml_config.update(state.get("config", {}))
            
            logger.info(f"📁 État IA chargé: {filepath}")
        except Exception as e:
            logger.error(f"Erreur chargement état IA: {e}")