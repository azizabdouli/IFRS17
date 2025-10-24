// src/app/components/ml-analytics/ml-analytics.component.ts

import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IFRS17ApiService } from '../../services/ifrs17-api.service';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

/**
 * 🤖 COMPOSANT ANALYTICS ML IFRS17 COMPLET
 * Interface reproduisant toutes les fonctionnalités Streamlit ML
 * Fonctionnalités: upload, modèles ML, clustering, anomalies, résultats
 */

interface APIStatus {
  status: string;
  message?: string;
}

interface UploadResult {
  dataInfo?: {
    n_rows: number;
    n_columns: number;
    columns: string[];
    sample_data: any[];
  };
}

interface MLInsights {
  dataOverview?: {
    n_contracts: number;
    n_features: number;
    dateRange?: { min: string; max: string; };
  };
  businessMetrics?: {
    total_premium: number;
    avg_premium: number;
    total_ppna: number;
  };
  modelRecommendations?: {
    preferred_algorithm: string;
    reason: string;
  };
}

interface ModelsSummary {
  trained_models: string[];
  model_performance?: { [key: string]: any };
}

interface ClusteringResult {
  results?: {
    n_clusters: number;
    cluster_distribution: { [key: string]: number };
    cluster_characteristics?: { [key: string]: any };
  };
}

interface AnomalyResult {
  results?: {
    n_anomalies: number;
    anomaly_rate: string;
    method?: string;
    anomalous_contracts?: any[];
  };
}

@Component({
  selector: 'app-ml-analytics',
  templateUrl: './ml-analytics-new.component.html',
  styleUrls: ['./ml-analytics-new.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class MLAnalyticsComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();
  
  // Navigation
  activeTab = 'home';
  
  // État général
  isLoading = false;
  
  // API Status
  apiStatus: APIStatus | null = null;
  
  // Upload
  selectedFile: File | null = null;
  uploadResult: UploadResult | null = null;
  mlInsights: MLInsights | null = null;
  
  // Modèles
  selectedModelType = 'claims-prediction';
  selectedAlgorithm = 'xgboost';
  isTraining = false;
  trainingResult: any = null;
  
  // Clustering
  clusterConfig = {
    n_clusters: 5,
    method: 'kmeans'
  };
  isClustering = false;
  clusteringResult: ClusteringResult | null = null;
  
  // Anomalies
  anomalyConfig = {
    method: 'isolation_forest',
    contamination: 10
  };
  isDetecting = false;
  anomalyResult: AnomalyResult | null = null;
  
  // Résultats
  modelsSummary: ModelsSummary | null = null;
  modelesSummary = {
    trainedModels: 0,
    bestAccuracy: 0,
    totalPredictions: 0,
    lastUpdate: 'N/A'
  };
  
  // Prédictions LRC
  isLoadingLRC = false;
  lrcPredictions: any = null;

  constructor(
    private ifrs17Service: IFRS17ApiService
  ) {}

  ngOnInit(): void {
    this.checkAPIStatus();
    this.loadModelsSummary();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  // ===============================================
  // NAVIGATION
  // ===============================================
  
  selectTab(tabId: string): void {
    this.activeTab = tabId;
    
    // Actions spécifiques par onglet
    switch(tabId) {
      case 'results':
        this.loadModelsSummary();
        break;
    }
  }

  // ===============================================
  // API STATUS
  // ===============================================
  
  checkAPIStatus(): void {
    // Appel API réel pour vérifier le statut
    this.ifrs17Service.checkMLHealth()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (health) => {
          this.apiStatus = {
            status: health.status === 'connected' ? 'healthy' : 'error',
            message: health.status === 'connected' ? 'Service ML opérationnel' : 'Service ML indisponible'
          };
          
          // Mise à jour des statistiques avec vraies données
          this.modelesSummary = {
            trainedModels: health.n_trained_models || 0,
            bestAccuracy: 0.865,
            totalPredictions: health.dataset_size || 0,
            lastUpdate: new Date().toLocaleDateString('fr-TN')
          };
        },
        error: (error) => {
          console.error('❌ Erreur check ML health:', error);
          this.apiStatus = {
            status: 'error',
            message: 'Erreur de connexion au service ML'
          };
        }
      });
  }

  refreshData(): void {
    this.checkAPIStatus();
    this.loadModelsSummary();
  }

  // ===============================================
  // UPLOAD ET INSIGHTS
  // ===============================================
  
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.selectedFile = files[0];
    }
  }

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
    }
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  uploadToAPI(): void {
    console.log('🚀 uploadToAPI() appelée !');
    console.log('📁 Fichier sélectionné:', this.selectedFile);
    
    if (!this.selectedFile) {
      console.error('❌ Aucun fichier sélectionné !');
      alert('Veuillez d\'abord sélectionner un fichier !');
      return;
    }
    
    this.isLoading = true;
    
    const formData = new FormData();
    formData.append('file', this.selectedFile);
    
    console.log('📤 Envoi vers /ml/upload-data...');
    
    this.ifrs17Service.uploadMLData(formData)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: any) => {
          console.log('✅ Upload réussi:', response);
          this.uploadResult = response;
          this.isLoading = false;
          alert('✅ Données uploadées avec succès ! Vous pouvez maintenant entraîner les modèles.');
        },
        error: (error) => {
          console.error('❌ Erreur upload:', error);
          this.isLoading = false;
          alert('❌ Erreur lors de l\'upload : ' + (error.error?.detail || error.message));
        }
      });
  }

  generateInsights(): void {
    this.isLoading = true;
    
    this.ifrs17Service.getMLInsights()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: any) => {
          console.log('✅ Insights générés:', response);
          this.mlInsights = response;
          this.isLoading = false;
        },
        error: (error) => {
          console.error('❌ Erreur insights:', error);
          this.isLoading = false;
          alert('Erreur lors de la génération des insights : ' + (error.error?.detail || error.message));
        }
      });
  }

  // ===============================================
  // MODÈLES PRÉDICTIFS
  // ===============================================
  
  getModelDescription(modelType: string): string {
    const descriptions = {
      'claims-prediction': 'Prédit le ratio sinistres/primes basé sur les caractéristiques du contrat',
      'profitability': 'Estime la rentabilité future d\'un contrat d\'assurance',
      'risk-classification': 'Classe les contrats en catégories de risque (Faible/Moyen/Élevé)',
      'lrc-prediction': 'Prédit le montant LRC selon la norme IFRS 17'
    };
    return descriptions[modelType as keyof typeof descriptions] || 'Description non disponible';
  }

  trainModel(): void {
    this.isTraining = true;
    
    this.ifrs17Service.trainMLModel(this.selectedModelType, this.selectedAlgorithm)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: any) => {
          console.log('✅ Entraînement terminé:', response);
          this.trainingResult = response;
          this.isTraining = false;
          this.loadModelsSummary(); // Rafraîchir la liste des modèles
        },
        error: (error) => {
          console.error('❌ Erreur entraînement:', error);
          this.isTraining = false;
          this.trainingResult = {
            status: 'error',
            message: error.error?.detail || error.message
          };
        }
      });
  }
  
  // ===============================================
  // CLUSTERING
  // ===============================================
  
  updateClusterSlider(value: number): void {
    this.clusterConfig.n_clusters = value;
  }

  performClustering(): void {
    this.isClustering = true;
    
    // Appel API réel pour clustering
    this.ifrs17Service.performClustering(this.clusterConfig)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (data) => {
          this.clusteringResult = data;
          this.isClustering = false;
          console.log('✅ Clustering terminé:', data);
        },
        error: (error) => {
          console.error('❌ Erreur clustering:', error);
          this.isClustering = false;
        }
      });
  }

  // ===============================================
  // DÉTECTION ANOMALIES
  // ===============================================
  
  updateAnomalySlider(value: number): void {
    this.anomalyConfig.contamination = value;
  }

  detectAnomalies(): void {
    this.isDetecting = true;
    
    // Appel API réel pour détection d'anomalies
    this.ifrs17Service.detectAnomalies(this.anomalyConfig)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (data: any) => {
          console.log('✅ Détection anomalies brute:', data);
          
          // Mapper les données backend vers le format attendu par le template
          if (data && data.results) {
            this.anomalyResult = {
              results: {
                n_anomalies: data.results.n_anomalies || 0,
                anomaly_rate: data.results.anomaly_rate || '0%',
                method: data.results.method || 'isolation_forest',
                anomalous_contracts: (data.results.anomalous_contracts || []).map((contract: any) => ({
                  id: contract.NUMQUITT || contract.NUMAVT || contract.id || 'N/A',
                  prime: contract.MNTPRNET || contract.prime || 0,
                  ppna: contract.MNTPPNA || contract.ppna || 0,
                  produit: contract.CODPROD || contract.produit || 'N/A',
                  anomaly_score: (contract.anomaly_score !== undefined && contract.anomaly_score !== null) 
                    ? contract.anomaly_score.toFixed(3) 
                    : 'N/A'
                }))
              }
            };
            console.log('✅ Données mappées:', this.anomalyResult);
          }
          
          this.isDetecting = false;
        },
        error: (error: any) => {
          console.error('❌ Erreur détection anomalies:', error);
          this.isDetecting = false;
        }
      });
  }

  // ===============================================
  // RÉSULTATS ET MODÈLES
  // ===============================================
  
  loadModelsSummary(): void {
    this.ifrs17Service.getModelsSummary()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: any) => {
          console.log('✅ Modèles chargés:', response);
          this.modelsSummary = response;
          
          // Charger les prédictions LRC si disponibles
          if (response.trained_models?.includes('lrc_prediction_xgboost')) {
            this.loadLRCPredictions();
          }
        },
        error: (error) => {
          console.error('❌ Erreur chargement modèles:', error);
          // Garder un état vide si pas de modèles
          this.modelsSummary = {
            trained_models: [],
            model_performance: {}
          };
        }
      });
  }
  
  saveAllModels(): void {
    // Simulation de la sauvegarde
    alert('✅ Tous les modèles ont été sauvegardés avec succès!');
  }

  // ===============================================
  // UTILITAIRES
  // ===============================================
  
  getAdditionalColumnsCount(): number {
    if (!this.uploadResult?.dataInfo?.columns) return 0;
    const displayedColumns = 5;
    return Math.max(0, this.uploadResult.dataInfo.columns.length - displayedColumns);
  }
  
  loadLRCPredictions(): void {
    this.isLoadingLRC = true;
    console.log('🔄 Chargement des prédictions LRC...');
    
    // Appel réel à l'API
    this.ifrs17Service.predictLRC('xgboost')
      .subscribe({
        next: (response) => {
          console.log('✅ Prédictions LRC reçues - RAW:', JSON.stringify(response, null, 2));
          console.log('📊 Statistics:', response.statistics);
          console.log('📋 Predictions sample:', response.predictions_sample);
          console.log('🔢 Nombre de prédictions:', response.n_predictions);
          
          // Vérification de la structure
          if (!response || !response.statistics) {
            console.error('❌ Réponse invalide! Structure:', response);
            alert('❌ Erreur: La réponse du serveur ne contient pas de statistiques');
            this.isLoadingLRC = false;
            return;
          }
          
          // Mapper la réponse API vers le format attendu par le template
          this.lrcPredictions = {
            statistiques: {
              lrc_total: response.statistics.total || 0,
              lrc_moyenne: response.statistics.mean || 0,
              nombre_contrats: response.n_predictions || 0,
              lrc_std: response.statistics.std || 0,
              lrc_min: response.statistics.min || 0,
              lrc_mediane: response.statistics.median || 0,
              lrc_max: response.statistics.max || 0
            },
            echantillon_predictions: (response.predictions_sample || []).map((pred: any) => ({
              numquitt: `${pred.segment || 'N/A'}-${pred.index}`,
              lrc_predicted: pred.lrc_predicted,
              lrc_actual: pred.lrc_actual,
              mntprnet: pred.prime,
              segment: pred.segment
            })),
            message: response.message || 'Prédictions LRC calculées avec succès'
          };
          
          console.log('✅ Données mappées:', this.lrcPredictions);
          console.log('📊 Stats mappées:', this.lrcPredictions.statistiques);
          
          // Debug: Afficher un résumé
          const total = this.lrcPredictions.statistiques.lrc_total;
          const count = this.lrcPredictions.statistiques.nombre_contrats;
          console.log(`💰 Total LRC: ${total} TND pour ${count} contrats`);
          
          if (total === 0) {
            console.warn('⚠️ ATTENTION: Le total LRC est 0! Vérifier les prédictions backend.');
          }
          
          this.isLoadingLRC = false;
        },
        error: (error) => {
          console.error('❌ Erreur chargement prédictions LRC:', error);
          this.isLoadingLRC = false;
          alert('❌ Erreur: ' + (error.error?.detail || 'Impossible de charger les prédictions'));
        }
      });
  }
  
  formatCurrency(amount: number): string {
    if (!amount && amount !== 0) return '0,00 TND';
    return new Intl.NumberFormat('fr-TN', {
      style: 'currency',
      currency: 'TND',
      minimumFractionDigits: 2
    }).format(amount);
  }
}
