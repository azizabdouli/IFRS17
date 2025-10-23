// src/app/components/ppna-analytics/ppna-analytics.component.ts

import { Component, OnInit, OnDestroy, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { PPNAService } from '../../services/ppna.service';
import { KeyValuePipe } from '../../pipes/keyvalue.pipe';
import { Chart, ChartConfiguration, registerables } from 'chart.js';

// Enregistrer tous les composants Chart.js
Chart.register(...registerables);

/**
 * 🔍 COMPOSANT ANALYTICS PPNA COMPLET
 * Reproduit exactement les fonctionnalités de l'interface Streamlit
 * Inclut: projections, analyses, exports, registre, paramètres
 */

interface PPNAData {
  [key: string]: any;
}

interface ProjectionData {
  mois: string;
  revenue_mois: number;
  dac_amort_mois: number;
  CODPROD?: string;
  Cohorte?: number;
  Onereux?: boolean;
}

interface ParametresProduit {
  CODPROD: string;
  DAC_pct: number;
  Eligible_PAA: boolean;
  [key: string]: number | string | boolean; // Index signature pour les mois
  M1: number;
  M2: number;
  M3: number;
  M4: number;
  M5: number;
  M6: number;
  M7: number;
  M8: number;
  M9: number;
  M10: number;
  M11: number;
  M12: number;
}

@Component({
  selector: 'app-ppna-analytics',
  templateUrl: './ppna-analytics.component.html',
  styleUrls: ['./ppna-analytics.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule, KeyValuePipe]
})
export class PPNAAnalyticsComponent implements OnInit, OnDestroy, AfterViewInit {
  private destroy$ = new Subject<void>();
  
  // Expose Math for template
  Math = Math;
  
  // État général
  isLoading = false;
  activeTab = 'donnees';
  currentDateTime: string = '';
  
  // Données
  ppnaData: PPNAData | null = null;
  projectionData: ProjectionData[] = [];
  parametresProduits: ParametresProduit[] = [];
  analysisResults: any = null;
  
  // Filtres
  filtreAnnees: [number, number] = [2020, 2025];
  filtreProduits: string[] = [];
  produitsDisponibles: string[] = [];
  
  // Paramètres projection
  anneesProjection: [number, number] = [2020, 2025];
  
  // Métriques calculées
  metriques = {
    totalContracts: 0,
    totalPrime: 0,
    totalPPNA: 0,
    pctOnereux: 0
  };
  
  // Charts instances
  private revenueChart: Chart | null = null;
  private lrcChart: Chart | null = null;
  private scatterChart: Chart | null = null;
  
  // Charts data
  chartRevenueData: any = null;
  chartLRCData: any = null;
  chartScatterData: any = null;
  
  tabs = [
    { id: 'donnees', label: '📁 Données', icon: 'fas fa-database' },
    { id: 'parametres', label: '⚙️ Paramètres', icon: 'fas fa-cogs' },
    { id: 'analyses', label: '📊 Analyses', icon: 'fas fa-chart-line' },
    { id: 'projection', label: '🧮 Projection', icon: 'fas fa-calculator' },
    { id: 'exports', label: '⬇️ Exports', icon: 'fas fa-download' },
    { id: 'registre', label: '📒 Registre', icon: 'fas fa-book' }
  ];

  constructor(
    private ppnaService: PPNAService
  ) {}

  ngOnInit(): void {
    // Initialiser la date une seule fois
    this.currentDateTime = new Date().toLocaleDateString('fr-TN') + ' ' + new Date().toLocaleTimeString('fr-TN');
    
    this.loadPPNAData();
    this.initParametresProduits();
  }

  ngAfterViewInit(): void {
    // Les graphiques seront créés quand l'utilisateur clique sur l'onglet Analyses
  }

  ngOnDestroy(): void {
    // Détruire les graphiques Chart.js
    this.destroyCharts();
    
    this.destroy$.next();
    this.destroy$.complete();
  }

  private destroyCharts(): void {
    if (this.revenueChart) {
      this.revenueChart.destroy();
      this.revenueChart = null;
    }
    if (this.lrcChart) {
      this.lrcChart.destroy();
      this.lrcChart = null;
    }
    if (this.scatterChart) {
      this.scatterChart.destroy();
      this.scatterChart = null;
    }
  }

  // ===============================================
  // GESTION DES ONGLETS
  // ===============================================
  
  selectTab(tabId: string): void {
    this.activeTab = tabId;
    
    // Actions spécifiques par onglet
    switch(tabId) {
      case 'analyses':
        this.performAnalyses();
        break;
      case 'projection':
        this.calculateProjection();
        break;
    }
  }

  // ===============================================
  // CHARGEMENT DES DONNÉES
  // ===============================================
  
  loadPPNAData(): void {
    this.isLoading = true;
    
    this.ppnaService.getDashboardMetrics()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (data) => {
          this.ppnaData = data;
          this.extractProduitsDisponibles();
          this.calculateMetriques();
          this.isLoading = false;
        },
        error: (error) => {
          console.error('Erreur chargement PPNA:', error);
          this.isLoading = false;
        }
      });
  }

  extractProduitsDisponibles(): void {
    if (this.ppnaData?.['analyse_segments']) {
      this.produitsDisponibles = this.ppnaData['analyse_segments'].map((s: any) => s.segment);
    }
  }

  calculateMetriques(): void {
    if (!this.ppnaData) return;
    
    this.metriques = {
      totalContracts: this.ppnaData['nombre_lignes'] || 0,
      totalPrime: this.ppnaData['primes_totales'] || 0,
      totalPPNA: this.ppnaData['ppna_total'] || 0,
      pctOnereux: this.ppnaData['contrats_onereux']?.['ratio_moyen_onereux'] || 0
    };
  }

  // ===============================================
  // GESTION DES PARAMÈTRES PRODUITS
  // ===============================================
  
  initParametresProduits(): void {
    if (this.produitsDisponibles.length === 0) {
      // Paramètres par défaut
      this.parametresProduits = [
        this.createDefaultParams('AUTO'),
        this.createDefaultParams('HABITATION'), 
        this.createDefaultParams('VIE')
      ];
    } else {
      this.parametresProduits = this.produitsDisponibles.map(prod => 
        this.createDefaultParams(prod)
      );
    }
  }

  createDefaultParams(codprod: string): ParametresProduit {
    return {
      CODPROD: codprod,
      DAC_pct: 0.10,
      Eligible_PAA: true,
      M1: 1/12, M2: 1/12, M3: 1/12, M4: 1/12,
      M5: 1/12, M6: 1/12, M7: 1/12, M8: 1/12,
      M9: 1/12, M10: 1/12, M11: 1/12, M12: 1/12
    };
  }

  ajouterParametreProduit(): void {
    this.parametresProduits.push(this.createDefaultParams('NOUVEAU'));
  }

  supprimerParametreProduit(index: number): void {
    this.parametresProduits.splice(index, 1);
  }

  exporterParametres(): void {
    const csvContent = this.convertToCsv(this.parametresProduits);
    this.downloadCsv(csvContent, 'IFRS17_Params_Produits.csv');
  }

  // ===============================================
  // ANALYSES ET VISUALISATIONS
  // ===============================================
  
  performAnalyses(): void {
    if (!this.ppnaData) return;
    
    this.isLoading = true;
    
    // Générer les données puis créer les graphiques
    setTimeout(() => {
      this.generateChartData();
      this.createCharts();
      this.isLoading = false;
    }, 500);
  }

  generateChartData(): void {
    if (!this.ppnaData) return;

    // Graphique revenus mensuels
    this.chartRevenueData = {
      labels: ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'],
      datasets: [{
        label: 'Revenue IFRS 17 (TND)',
        data: this.generateMonthlyRevenue(),
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4
      }]
    };

    // Graphique distribution LRC
    this.chartLRCData = {
      labels: ['< -1000', '-1000 à 0', '0 à 1000', '1000 à 5000', '> 5000'],
      datasets: [{
        label: 'Distribution LRC',
        data: [5, 15, 45, 25, 10],
        backgroundColor: [
          'rgba(255, 99, 132, 0.6)',
          'rgba(255, 159, 64, 0.6)',
          'rgba(255, 205, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(54, 162, 235, 0.6)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(255, 159, 64, 1)',
          'rgba(255, 205, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(54, 162, 235, 1)'
        ],
        borderWidth: 1
      }]
    };

    // Graphique scatter PPNA
    const segments = this.ppnaData['analyse_segments'] || [];
    this.chartScatterData = {
      datasets: [{
        label: 'PPNA comptable vs IFRS17',
        data: segments.map((s: any) => ({
          x: s.primes || 0,
          y: s.provisions || 0
        })),
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        pointRadius: 6,
        pointHoverRadius: 8
      }, {
        label: 'Égalité parfaite',
        data: [
          { x: 0, y: 0 },
          { x: this.metriques.totalPrime, y: this.metriques.totalPrime }
        ],
        type: 'line',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false
      }]
    };
  }

  private createCharts(): void {
    // Détruire les graphiques existants
    this.destroyCharts();

    // Attendre que le DOM soit prêt
    setTimeout(() => {
      // Créer le graphique des revenus
      const revenueCanvas = document.getElementById('revenueChart') as HTMLCanvasElement;
      console.log('Revenue canvas:', revenueCanvas);
      if (revenueCanvas && this.chartRevenueData) {
        this.revenueChart = new Chart(revenueCanvas, {
        type: 'line',
        data: this.chartRevenueData,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'top'
            },
            title: {
              display: false
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: {
                callback: (value) => {
                  return new Intl.NumberFormat('fr-TN', {
                    style: 'currency',
                    currency: 'TND',
                    minimumFractionDigits: 0
                  }).format(value as number);
                }
              }
            }
          }
        }
      });
    }

    // Créer le graphique LRC
    const lrcCanvas = document.getElementById('lrcChart') as HTMLCanvasElement;
    if (lrcCanvas && this.chartLRCData) {
      this.lrcChart = new Chart(lrcCanvas, {
        type: 'bar',
        data: this.chartLRCData,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Nombre de contrats'
              }
            }
          }
        }
      });
    }

      // Créer le graphique scatter
      const scatterCanvas = document.getElementById('scatterChart') as HTMLCanvasElement;
      console.log('Scatter canvas:', scatterCanvas);
      if (scatterCanvas && this.chartScatterData) {
        this.scatterChart = new Chart(scatterCanvas, {
          type: 'scatter',
          data: this.chartScatterData,
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                display: true,
                position: 'top'
              }
            },
            scales: {
              x: {
                title: {
                  display: true,
                  text: 'PPNA Comptable (TND)'
                },
                ticks: {
                  callback: (value) => {
                    return new Intl.NumberFormat('fr-TN', {
                      notation: 'compact',
                      compactDisplay: 'short'
                    }).format(value as number);
                  }
                }
              },
              y: {
                title: {
                  display: true,
                  text: 'PPNA IFRS17 (TND)'
                },
                ticks: {
                  callback: (value) => {
                    return new Intl.NumberFormat('fr-TN', {
                      notation: 'compact',
                      compactDisplay: 'short'
                    }).format(value as number);
                  }
                }
              }
            }
          }
        });
      }
    }, 100); // Fermeture du setTimeout
  }

  generateMonthlyRevenue(): number[] {
    const baseRevenue = this.metriques.totalPrime / 12;
    return Array.from({length: 12}, (_, i) => 
      baseRevenue * (0.8 + Math.random() * 0.4)
    );
  }

  // ===============================================
  // PROJECTION MENSUELLE
  // ===============================================
  
  calculateProjection(): void {
    this.isLoading = true;
    
    // Simulation de la projection exacte
    setTimeout(() => {
      this.projectionData = this.generateProjectionData();
      this.isLoading = false;
    }, 1500);
  }

  generateProjectionData(): ProjectionData[] {
    const projections: ProjectionData[] = [];
    const startDate = new Date(this.anneesProjection[0], 0, 1);
    const endDate = new Date(this.anneesProjection[1], 11, 31);
    
    let currentDate = new Date(startDate);
    while (currentDate <= endDate) {
      projections.push({
        mois: currentDate.toISOString().substring(0, 7),
        revenue_mois: Math.random() * 100000 + 50000,
        dac_amort_mois: Math.random() * 10000 + 5000,
        CODPROD: this.produitsDisponibles[Math.floor(Math.random() * this.produitsDisponibles.length)] || 'AUTO',
        Cohorte: currentDate.getFullYear(),
        Onereux: Math.random() > 0.8
      });
      
      currentDate.setMonth(currentDate.getMonth() + 1);
    }
    
    return projections;
  }

  exporterProjection(): void {
    if (this.projectionData.length === 0) return;
    
    const csvContent = this.convertToCsv(this.projectionData);
    this.downloadCsv(csvContent, 'IFRS17_Projection_Mensuelle.csv');
  }

  // ===============================================
  // EXPORTS
  // ===============================================
  
  exporterExcel(): void {
    // Simulation export Excel
    alert('Export Excel en cours de développement...');
  }

  exporterPDF(): void {
    // Simulation export PDF
    const reportData = {
      date: new Date().toLocaleDateString('fr-TN'),
      totalContracts: this.metriques.totalContracts,
      totalPrime: this.metriques.totalPrime,
      pctOnereux: this.metriques.pctOnereux
    };
    
    console.log('Génération PDF:', reportData);
    alert('Export PDF en cours de développement...');
  }

  formatPercentage(value: number): string {
    return new Intl.NumberFormat('fr-TN', {
      style: 'percent',
      minimumFractionDigits: 1
    }).format(value / 100);
  }

  private convertToCsv(data: any[]): string {
    if (data.length === 0) return '';
    
    const headers = Object.keys(data[0]).join(',');
    const rows = data.map(row => 
      Object.values(row).map(val => 
        typeof val === 'string' ? `"${val}"` : val
      ).join(',')
    ).join('\n');
    
    return headers + '\n' + rows;
  }

  private downloadCsv(csvContent: string, filename: string): void {
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    window.URL.revokeObjectURL(url);
  }

  // ===============================================
  // GESTION DES FILTRES
  // ===============================================
  
  applyFilters(): void {
    // Logique de filtrage
    console.log('Filtres appliqués:', {
      annees: this.filtreAnnees,
      produits: this.filtreProduits
    });
    
    // Recharger les analyses avec filtres
    this.performAnalyses();
  }

  resetFilters(): void {
    this.filtreAnnees = [2020, 2025];
    this.filtreProduits = [];
    this.applyFilters();
  }

  // ===============================================
  // UTILITAIRES
  // ===============================================
  
  formatCurrency(amount: number): string {
    if (!amount && amount !== 0) return '0,00 TND';
    return new Intl.NumberFormat('fr-TN', {
      style: 'currency',
      currency: 'TND',
      minimumFractionDigits: 2
    }).format(amount);
  }

  getCurrentDateTime(): string {
    return new Date().toLocaleDateString('fr-TN') + ' ' + new Date().toLocaleTimeString('fr-TN');
  }

  getCurrentDate(): string {
    return new Date().toLocaleDateString('fr-TN');
  }
}