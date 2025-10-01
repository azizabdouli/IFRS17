# backend/ai/ifrs17_ai_assistant.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime
import asyncio
from functools import lru_cache

# Import pour LLM local ou API
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class IFRS17AIAssistant:
    """
    Assistant IA spécialisé IFRS17 avec capacités conversationnelles
    et analyse intelligente des données d'assurance
    """
    
    def __init__(self):
        self.conversation_history = []
        self.context_memory = {}
        self.ifrs17_knowledge = self._load_ifrs17_knowledge()
        
        # Initialiser le modèle IA si disponible
        self.llm_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            self._initialize_llm()
        
        logger.info("🤖 Assistant IA IFRS17 initialisé")
    
    def _initialize_llm(self):
        """Initialise le modèle de langage local"""
        try:
            # Utiliser un modèle léger pour commencer
            model_name = "microsoft/DialoGPT-medium"
            self.llm_pipeline = pipeline(
                "conversational",
                model=model_name,
                tokenizer=model_name
            )
            logger.info(f"✅ Modèle LLM initialisé: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de charger le LLM local: {e}")
    
    def _load_ifrs17_knowledge(self) -> Dict[str, Any]:
        """Base de connaissances IFRS17 spécialisée"""
        return {
            "definitions": {
                "PAA": "Premium Allocation Approach - Approche d'allocation des primes",
                "CSM": "Contractual Service Margin - Marge de service contractuel",
                "LRC": "Liability for Remaining Coverage - Passif pour couverture restante",
                "LIC": "Liability for Incurred Claims - Passif pour sinistres survenus",
                "FCF": "Fulfilment Cash Flows - Flux de trésorerie d'exécution"
            },
            "calculations": {
                "profitability": "Rentabilité = (Primes - Sinistres - Frais) / Primes",
                "loss_ratio": "Ratio sinistres = Sinistres / Primes acquises",
                "combined_ratio": "Ratio combiné = (Sinistres + Frais) / Primes"
            },
            "models": {
                "profitability": {"performance": "R² = 0.964", "usage": "Analyse rentabilité"},
                "risk_classification": {"performance": "85%+ accuracy", "usage": "Classification risques"},
                "claims_prediction": {"performance": "R² = 0.732", "usage": "Prédiction sinistres"},
                "lrc_prediction": {"performance": "R² = 0.937", "usage": "Prédiction LRC"}
            }
        }
    
    async def process_query(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Traite une requête utilisateur avec intelligence contextuelle
        """
        # Ajouter à l'historique
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user_query": user_query,
            "context": context or {}
        })
        
        # Analyser l'intention
        intent = self._analyze_intent(user_query)
        
        # Générer la réponse
        response = await self._generate_response(user_query, intent, context)
        
        # Ajouter à l'historique
        self.conversation_history[-1]["ai_response"] = response
        
        return response
    
    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyse l'intention de la requête utilisateur"""
        query_lower = query.lower()
        
        intent = {
            "type": "general",
            "confidence": 0.5,
            "keywords": [],
            "data_needed": False,
            "action": None
        }
        
        # Mots-clés IFRS17
        ifrs17_keywords = ["paa", "csm", "lrc", "lic", "fcf", "ifrs17", "prime", "sinistre"]
        found_keywords = [kw for kw in ifrs17_keywords if kw in query_lower]
        intent["keywords"] = found_keywords
        
        # Détection d'intention
        if any(word in query_lower for word in ["analyse", "analyser", "données"]):
            intent["type"] = "analysis"
            intent["confidence"] = 0.8
            intent["data_needed"] = True
        elif any(word in query_lower for word in ["prédiction", "prévoir", "modèle"]):
            intent["type"] = "prediction"
            intent["confidence"] = 0.9
            intent["action"] = "run_model"
        elif any(word in query_lower for word in ["clustering", "groupe", "segmentation"]):
            intent["type"] = "clustering"
            intent["confidence"] = 0.85
            intent["action"] = "clustering"
        elif any(word in query_lower for word in ["définition", "qu'est-ce", "expliquer"]):
            intent["type"] = "explanation"
            intent["confidence"] = 0.9
        elif any(word in query_lower for word in ["performance", "résultat", "métrique"]):
            intent["type"] = "performance"
            intent["confidence"] = 0.8
        
        return intent
    
    async def _generate_response(self, query: str, intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une réponse intelligente basée sur l'intention"""
        
        response = {
            "text": "",
            "suggestions": [],
            "actions": [],
            "data_insights": None,
            "confidence": intent["confidence"]
        }
        
        if intent["type"] == "explanation":
            response = await self._handle_explanation(query, intent)
        elif intent["type"] == "analysis":
            response = await self._handle_analysis(query, context)
        elif intent["type"] == "prediction":
            response = await self._handle_prediction(query, context)
        elif intent["type"] == "performance":
            response = await self._handle_performance(query)
        else:
            response = await self._handle_general(query, intent)
        
        return response
    
    async def _handle_explanation(self, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Gère les demandes d'explication"""
        explanations = []
        
        for keyword in intent["keywords"]:
            if keyword in self.ifrs17_knowledge["definitions"]:
                explanations.append(f"**{keyword.upper()}**: {self.ifrs17_knowledge['definitions'][keyword]}")
        
        if not explanations:
            explanations = ["Je peux vous expliquer les concepts IFRS17 : PAA, CSM, LRC, LIC, FCF..."]
        
        return {
            "text": f"📚 **Explications IFRS17:**\n\n" + "\n\n".join(explanations),
            "suggestions": ["Expliquer PAA", "Définir CSM", "Calculer la rentabilité"],
            "actions": [],
            "confidence": 0.9
        }
    
    async def _handle_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gère les demandes d'analyse de données"""
        if not context or "data" not in context:
            return {
                "text": "🔍 **Analyse de données requise**\n\nPour effectuer une analyse, veuillez d'abord charger vos données IFRS17.",
                "suggestions": ["Charger des données", "Voir exemple d'analyse"],
                "actions": ["upload_data"],
                "confidence": 0.8
            }
        
        # Analyse des données si disponibles
        data_insights = self._analyze_data(context["data"])
        
        return {
            "text": f"📊 **Analyse de vos données IFRS17:**\n\n{data_insights['summary']}",
            "suggestions": ["Clustering", "Prédictions", "Détection d'anomalies"],
            "actions": ["run_clustering", "run_predictions"],
            "data_insights": data_insights,
            "confidence": 0.9
        }
    
    async def _handle_prediction(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gère les demandes de prédiction"""
        available_models = list(self.ifrs17_knowledge["models"].keys())
        
        text = "🔮 **Modèles de prédiction disponibles:**\n\n"
        for model, info in self.ifrs17_knowledge["models"].items():
            text += f"• **{model.title()}**: {info['usage']} (Performance: {info['performance']})\n"
        
        return {
            "text": text,
            "suggestions": [f"Prédiction {model}" for model in available_models],
            "actions": ["select_model", "run_prediction"],
            "confidence": 0.85
        }
    
    async def _handle_performance(self, query: str) -> Dict[str, Any]:
        """Gère les demandes de performance"""
        perf_text = "📈 **Performance des modèles IFRS17:**\n\n"
        
        for model, info in self.ifrs17_knowledge["models"].items():
            perf_text += f"• **{model.replace('_', ' ').title()}**: {info['performance']}\n"
        
        perf_text += "\n🚀 **Optimisations système:**\n"
        perf_text += "• Traitement: 1,171,318 lignes/sec\n"
        perf_text += "• Cache: 90%+ hit rate\n"
        perf_text += "• Mémoire: <3MB utilisation"
        
        return {
            "text": perf_text,
            "suggestions": ["Tests performance", "Optimiser modèles"],
            "actions": ["run_performance_test"],
            "confidence": 0.9
        }
    
    async def _handle_general(self, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Gère les requêtes générales"""
        if self.llm_pipeline and TRANSFORMERS_AVAILABLE:
            try:
                # Utiliser le LLM local
                result = self.llm_pipeline(query)
                ai_text = result.generated_responses[0] if result.generated_responses else query
            except:
                ai_text = self._generate_template_response(query, intent)
        else:
            ai_text = self._generate_template_response(query, intent)
        
        return {
            "text": f"🤖 {ai_text}",
            "suggestions": ["Analyser données", "Prédictions", "Performance"],
            "actions": ["general_help"],
            "confidence": intent["confidence"]
        }
    
    def _generate_template_response(self, query: str, intent: Dict[str, Any]) -> str:
        """Génère une réponse template intelligente"""
        templates = [
            "Je suis votre assistant IA spécialisé IFRS17. Comment puis-je vous aider avec l'analyse de vos contrats d'assurance ?",
            "Excellente question ! Je peux vous aider avec l'analyse IFRS17, les prédictions ML, et l'optimisation de performance.",
            "En tant qu'expert IA IFRS17, je peux analyser vos données, expliquer les concepts, et recommander des actions."
        ]
        
        if intent["keywords"]:
            return f"Je vois que vous vous intéressez à : {', '.join(intent['keywords'])}. {templates[0]}"
        
        return templates[0]
    
    def _analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse intelligente des données IFRS17"""
        try:
            insights = {
                "summary": "",
                "statistics": {},
                "recommendations": [],
                "anomalies": []
            }
            
            # Statistiques de base
            insights["statistics"] = {
                "total_contracts": len(df),
                "total_premium": df.get("prime_brute", pd.Series()).sum() if "prime_brute" in df.columns else 0,
                "columns_count": len(df.columns),
                "missing_data": df.isnull().sum().sum()
            }
            
            # Génération du résumé
            insights["summary"] = f"""
📊 **{insights['statistics']['total_contracts']:,} contrats** analysés
💰 **Prime totale**: {insights['statistics']['total_premium']:,.0f}€
📋 **{insights['statistics']['columns_count']} colonnes** de données
⚠️ **{insights['statistics']['missing_data']} valeurs manquantes**
            """.strip()
            
            # Recommandations intelligentes
            if insights['statistics']['missing_data'] > 0:
                insights["recommendations"].append("🔧 Nettoyer les données manquantes")
            
            if insights['statistics']['total_contracts'] > 10000:
                insights["recommendations"].append("🚀 Utiliser le clustering pour segmenter")
            
            insights["recommendations"].append("📈 Lancer les prédictions ML")
            
            return insights
            
        except Exception as e:
            logger.error(f"Erreur analyse données: {e}")
            return {"summary": "Erreur lors de l'analyse", "statistics": {}, "recommendations": []}
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Retourne l'historique de conversation"""
        return self.conversation_history[-10:]  # Dernières 10 interactions
    
    def clear_history(self):
        """Vide l'historique de conversation"""
        self.conversation_history.clear()
        logger.info("🧹 Historique de conversation vidé")
    
    @lru_cache(maxsize=32)
    def get_quick_help(self, topic: str = "general") -> str:
        """Aide rapide mise en cache"""
        help_topics = {
            "general": "💡 **Aide**: Posez-moi des questions sur IFRS17, l'analyse de données, ou les prédictions ML.",
            "upload": "📤 **Upload**: Glissez votre fichier Excel/CSV avec les colonnes : NUMQUITT, prime_brute, date_effet...",
            "models": "🤖 **Modèles**: 4 modèles ML disponibles pour rentabilité, risques, sinistres, et LRC.",
            "performance": "⚡ **Performance**: Tests automatisés avec 1M+ lignes/sec et cache intelligent."
        }
        return help_topics.get(topic, help_topics["general"])