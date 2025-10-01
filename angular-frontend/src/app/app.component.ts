// src/app/app.component.ts

import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';

/**
 * 🏢 Composant principal de l'application IFRS17
 * Interface de comptabilité d'assurance avec terminologie technique
 */

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  standalone: true,
  imports: [CommonModule, RouterOutlet]
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'IFRS17 - Comptabilité Assurance';
  
  // État de l'application
  isLoading = false;
  apiConnected = false;
  currentUser: any = null;
  notifications: any[] = [];
  
  // Navigation state
  isNavigationOpen = false;
  
  // AI Assistant state
  showAIAssistant = false;

  // Métriques en temps réel
  metriquesTempsReel = {
    nombreContrats: 0,
    lrcTotal: 0,
    licTotal: 0,
    csmTotal: 0,
    contratsOnereux: 0,
    ratioSolvabilite: 0
  };

  constructor() {}

  ngOnInit(): void {
    this.initializeApplication();
    this.verifierConnexionAPI();
    this.chargerMetriquesTempsReel();
  }

  ngOnDestroy(): void {
    // Nettoyage des ressources
  }

  private initializeApplication(): void {
    console.log('🏢 Initialisation IFRS17 Comptabilité Assurance');
    console.log('📊 Version: 3.0.0 - Interface Angular');
    console.log('🧠 IA: Activée');
    console.log('🤖 ML: 5 modèles disponibles');
  }

  private verifierConnexionAPI(): void {
    // Logique de vérification API
    this.apiConnected = true; // Temporaire
  }

  private chargerMetriquesTempsReel(): void {
    // Simulation des métriques
    this.metriquesTempsReel = {
      nombreContrats: 45678,
      lrcTotal: 1250000000,
      licTotal: 850000000,
      csmTotal: 125000000,
      contratsOnereux: 1234,
      ratioSolvabilite: 198.5
    };
  }
  
  toggleAIAssistant(): void {
    this.showAIAssistant = !this.showAIAssistant;
    console.log(`Assistant IA ${this.showAIAssistant ? 'activé' : 'désactivé'}`);
  }

  onMenuSelection(menu: string): void {
    console.log(`Navigation vers: ${menu}`);
  }

  onNotificationDismiss(index: number): void {
    this.notifications.splice(index, 1);
  }
}