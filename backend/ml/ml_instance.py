# backend/ml/ml_instance.py
"""
Instance globale partagée du service ML
Cette instance est utilisée par tous les routers pour garantir
que les données uploadées sont accessibles partout
"""

from backend.ml.optimized_ml_service import EnhancedMLService

# Instance globale unique du service ML
ml_service = EnhancedMLService()
