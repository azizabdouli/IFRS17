// src/app/components/ai-assistant/ai-assistant.component.ts

import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IFRS17ApiService } from '../../services/ifrs17-api.service';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

/**
 * 🧠 ASSISTANT IA IFRS17
 * Interface conversationnelle pour l'assistance en comptabilité d'assurance
 * Fonctionnalités: chat IA, conseils actuariels, analyses prédictives, aide contextuelle
 */

interface Message {
  id: number;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  suggestions?: string[];
  attachments?: any[];
}

@Component({
  selector: 'app-ai-assistant',
  templateUrl: './ai-assistant.component.html',
  styleUrls: ['./ai-assistant.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class AIAssistantComponent implements OnInit, OnDestroy {
  
  @ViewChild('chatContainer') chatContainer!: ElementRef;
  @ViewChild('messageInput') messageInput!: ElementRef;
  
  private destroy$ = new Subject<void>();
  
  // État du chat
  messages: Message[] = [];
  currentMessage = '';
  isTyping = false;
  isConnected = true;
  
  // Suggestions prédéfinies
  suggestionsRapides = [
    'Comment calculer le CSM d\'une cohorte ?',
    'Quels sont les contrats onéreux à surveiller ?',
    'Analyse de la profitabilité du portefeuille automobile',
    'Recommandations pour optimiser les provisions IFRS17',
    'Tendances de sinistralité par ligne d\'activité',
    'Impact des changements d\'hypothèses actuarielles'
  ];

  // Contexte actuariel
  contexteActuariel = {
    terminologieIFRS17: [
      'LRC - Liability for Remaining Coverage',
      'LIC - Liability for Incurred Claims', 
      'CSM - Contractual Service Margin',
      'RA - Risk Adjustment',
      'VIF - Value in Force',
      'PVFCF - Present Value of Future Cash Flows'
    ],
    domaines: [
      'Comptabilité IFRS17',
      'Actuariat Vie et Non-Vie',
      'Gestion des Risques',
      'Modélisation Financière',
      'Analyse Prédictive',
      'Optimisation de Portefeuille'
    ]
  };

  constructor(private ifrs17Service: IFRS17ApiService) {}

  ngOnInit(): void {
    this.initializeChat();
    this.loadChatHistory();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  // ================================
  // 🚀 INITIALISATION
  // ================================

  private initializeChat(): void {
    // Message de bienvenue
    const welcomeMessage: Message = {
      id: 1,
      type: 'assistant',
      content: `🏛️ **Bonjour ! Je suis votre assistant IA spécialisé en IFRS17.**

Je peux vous aider avec :
• **Comptabilité d'assurance** selon la norme IFRS17
• **Analyses actuarielles** et calculs de provisions
• **Détection de contrats onéreux** et optimisation
• **Interprétation des données** et recommandations ML
• **Formation** sur les concepts IFRS17

Comment puis-je vous assister aujourd'hui ?`,
      timestamp: new Date(),
      suggestions: this.suggestionsRapides.slice(0, 3)
    };

    this.messages = [welcomeMessage];
  }

  private loadChatHistory(): void {
    // Simulation du chargement de l'historique
    console.log('📚 Chargement de l\'historique des conversations...');
  }

  // ================================
  // 💬 GESTION DES MESSAGES
  // ================================

  sendMessage(): void {
    if (!this.currentMessage.trim()) return;

    // Ajouter le message utilisateur
    const userMessage: Message = {
      id: this.messages.length + 1,
      type: 'user',
      content: this.currentMessage,
      timestamp: new Date()
    };

    this.messages.push(userMessage);
    
    // Traiter la réponse IA
    this.processAIResponse(this.currentMessage);
    
    // Nettoyer l'input
    this.currentMessage = '';
    
    // Scroll vers le bas
    setTimeout(() => this.scrollToBottom(), 100);
  }

  private processAIResponse(userInput: string): void {
    this.isTyping = true;
    
    // Simulation d'analyse et génération de réponse
    setTimeout(() => {
      const response = this.generateAIResponse(userInput);
      
      const aiMessage: Message = {
        id: this.messages.length + 1,
        type: 'assistant',
        content: response.content,
        timestamp: new Date(),
        suggestions: response.suggestions
      };

      this.messages.push(aiMessage);
      this.isTyping = false;
      
      setTimeout(() => this.scrollToBottom(), 100);
    }, 1500 + Math.random() * 1000);
  }

  private generateAIResponse(input: string): { content: string; suggestions: string[] } {
    const lowerInput = input.toLowerCase();
    
    // Réponses contextuelles basées sur les mots-clés IFRS17
    if (lowerInput.includes('csm') || lowerInput.includes('contractual service margin')) {
      return {
        content: `📊 **Contractual Service Margin (CSM)**

Le CSM représente le profit non encore reconnu dans les contrats d'assurance. Voici les points clés :

**🔸 Calcul initial :**
• CSM = PVFCF - Ajustement de risque - Flux de trésorerie à l'origine
• Si négatif → Contrat onéreux (perte immédiate)

**🔸 Évolution :**
• Décharge progressive sur la durée du contrat
• Ajustements pour modifications d'estimations
• Impact des changements d'hypothèses actuarielles

**🔸 Suivi recommandé :**
• Analyse par cohorte mensuelle
• Sensibilité aux hypothèses
• Reconciliation des mouvements`,
        suggestions: [
          'Montrer l\'évolution CSM par cohorte',
          'Analyser la sensibilité du CSM aux hypothèses',
          'Calculer l\'impact d\'un changement de taux'
        ]
      };
    }

    if (lowerInput.includes('contrat onéreux') || lowerInput.includes('onereux')) {
      return {
        content: `🔴 **Contrats Onéreux - Détection et Gestion**

**📈 Situation actuelle :**
• **1,847 contrats** identifiés comme onéreux
• **Impact financier :** -23,4 M€
• **Lignes les plus impactées :** Santé (65%), Automobile (28%)

**🎯 Recommandations immédiates :**
1. **Révision tarifaire** pour les nouveaux contrats
2. **Optimisation des provisions** sur l'existant  
3. **Analyse prédictive** pour anticiper les dérives
4. **Actions correctives** sur les garanties

**🔍 Monitoring suggéré :**
• Alertes automatiques si CSM < 0
• Suivi hebdomadaire par gestionnaire
• Analyse des causes racines`,
        suggestions: [
          'Voir la liste détaillée des contrats onéreux',
          'Analyser les causes par ligne d\'activité',
          'Proposer un plan d\'action correctif'
        ]
      };
    }

    if (lowerInput.includes('lrc') || lowerInput.includes('liability remaining coverage')) {
      return {
        content: `📋 **Liability for Remaining Coverage (LRC)**

**💰 Situation actuelle :**
• **LRC Total :** 2,45 Md€
• **Évolution :** +3,2% vs mois précédent
• **Composition :** CSM (13,3%) + RA (5,1%) + PVFCF (81,6%)

**🔍 Analyse par ligne :**
• **Automobile :** 862 M€ (35,2%)
• **Habitation :** 704 M€ (28,7%)  
• **Vie Individuelle :** 453 M€ (18,5%)
• **Santé :** 297 M€ (12,1%)

**⚡ Actions recommandées :**
• Validation mensuelle des hypothèses
• Réconciliation avec les systèmes sources
• Test de suffisance des passifs`,
        suggestions: [
          'Détailler l\'évolution LRC par cohorte',
          'Analyser les écarts vs budget',
          'Projeter l\'évolution sur 12 mois'
        ]
      };
    }

    if (lowerInput.includes('machine learning') || lowerInput.includes('prédiction') || lowerInput.includes('ml')) {
      return {
        content: `🤖 **Analyses ML & Prédictives IFRS17**

**🎯 Modèles disponibles :**
• **Profitabilité :** 96,4% précision (Excellent)
• **Classification Risque :** 86,5% précision (Bon)
• **Contrats Onéreux :** 78,9% précision (Satisfaisant)
• **Prédiction Sinistres :** 82,1% précision (Bon)

**📊 Analyses en cours :**
• Profitabilité future Auto (75% complété)
• Détection contrats onéreux Santé (Terminé)

**💡 Recommandations :**
1. **Réentraîner** le modèle contrats onéreux
2. **Étendre** l'analyse prédictive aux autres lignes
3. **Automatiser** la détection d'anomalies
4. **Intégrer** les résultats ML dans le reporting`,
        suggestions: [
          'Lancer une nouvelle prédiction',
          'Voir les détails des modèles ML',
          'Exporter les résultats prédictifs'
        ]
      };
    }

    // Réponse générique avec conseils actuariels
    return {
      content: `🧠 **Assistant IA IFRS17 à votre service !**

Je comprends votre question sur "${input}". Voici quelques domaines où je peux vous aider :

**📚 Expertise disponible :**
• **Comptabilité IFRS17** - Calculs, provisions, reporting
• **Actuariat** - Modélisation, hypothèses, projections  
• **Analyse des données** - Interprétation, tendances, KPIs
• **Recommandations ML** - Optimisation, prédictions
• **Formation** - Concepts, méthodologies, bonnes pratiques

**🎯 Pour une réponse plus précise, vous pouvez mentionner :**
• Des termes techniques (CSM, LRC, LIC, RA...)
• Une ligne d'activité spécifique
• Un type d'analyse souhaité

Comment puis-je vous aider plus spécifiquement ?`,
      suggestions: [
        'Expliquer les concepts IFRS17 de base',
        'Analyser les performances du portefeuille',
        'Donner des conseils d\'optimisation'
      ]
    };
  }

  // ================================
  // 🎯 ACTIONS RAPIDES
  // ================================

  sendSuggestion(suggestion: string): void {
    this.currentMessage = suggestion;
    this.sendMessage();
  }

  clearChat(): void {
    this.messages = [];
    this.initializeChat();
  }

  exportChat(): void {
    const chatData = {
      conversation: this.messages,
      timestamp: new Date(),
      participant: 'Assistant IA IFRS17'
    };
    
    console.log('💾 Export de la conversation:', chatData);
    // Ici, integration avec service d'export
  }

  // ================================
  // 🔧 UTILITAIRES
  // ================================

  private scrollToBottom(): void {
    try {
      this.chatContainer.nativeElement.scrollTop = this.chatContainer.nativeElement.scrollHeight;
    } catch(err) {
      console.log('Erreur scroll:', err);
    }
  }

  onKeyPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  formatMessage(content: string): string {
    // Conversion markdown simple pour l'affichage
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/•/g, '&bull;')
      .replace(/\n/g, '<br>');
  }

  getMessageTime(timestamp: Date): string {
    return timestamp.toLocaleTimeString('fr-FR', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  }

  // ================================
  // 📊 MÉTHODES DE FILTRAGE
  // ================================

  getUserMessagesCount(): number {
    return this.messages.filter(m => m.type === 'user').length;
  }

  getAssistantMessagesCount(): number {
    return this.messages.filter(m => m.type === 'assistant').length;
  }
}