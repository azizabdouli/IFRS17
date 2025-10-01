// src/app/components/data-transformations/data-transformations.component.ts

import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IFRS17ApiService } from '../../services/ifrs17-api.service';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

/**
 * 🔄 COMPOSANT TRANSFORMATIONS DE DONNÉES IFRS17
 * Interface pour les transformations, mappings et traitements de données
 * Équivalent des fonctionnalités Streamlit du frontend
 */

interface TransformationJob {
  id: number;
  nom: string;
  type: 'mapping' | 'aggregation' | 'calculation' | 'validation';
  statut: 'EN_ATTENTE' | 'EN_COURS' | 'TERMINE' | 'ERREUR';
  progression: number;
  fichierSource: string;
  fichierCible: string;
  lignesTraitees: number;
  lignesTotal: number;
  erreurs: string[];
  dateDebut: Date;
  dateFin?: Date;
}

@Component({
  selector: 'app-data-transformations',
  templateUrl: './data-transformations.component.html',
  styleUrls: ['./data-transformations.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class DataTransformationsComponent implements OnInit, OnDestroy {
  
  private destroy$ = new Subject<void>();
  isLoading = false;
  
  // Jobs de transformation
  transformationJobs: TransformationJob[] = [
    {
      id: 1,
      nom: 'Mapping Données Actuarielles',
      type: 'mapping',
      statut: 'TERMINE',
      progression: 100,
      fichierSource: 'donnees_brutes_Q3_2024.xlsx',
      fichierCible: 'donnees_ifrs17_mappees.json',
      lignesTraitees: 125450,
      lignesTotal: 125450,
      erreurs: [],
      dateDebut: new Date(Date.now() - 3600000),
      dateFin: new Date(Date.now() - 1800000)
    },
    {
      id: 2,
      nom: 'Calcul CSM par Cohorte',
      type: 'calculation',
      statut: 'EN_COURS',
      progression: 75,
      fichierSource: 'cohortes_Q3_2024.csv',
      fichierCible: 'csm_calcule.json',
      lignesTraitees: 3750,
      lignesTotal: 5000,
      erreurs: ['Hypothèse manquante pour cohorte AUTO_2023_Q1'],
      dateDebut: new Date(Date.now() - 1200000)
    },
    {
      id: 3,
      nom: 'Validation Cohérence IFRS17',
      type: 'validation',
      statut: 'EN_ATTENTE',
      progression: 0,
      fichierSource: 'donnees_consolidees.json',
      fichierCible: 'rapport_validation.pdf',
      lignesTraitees: 0,
      lignesTotal: 89000,
      erreurs: [],
      dateDebut: new Date()
    }
  ];

  // Métriques de transformation
  metriquesTransformation = {
    totalJobs: 0,
    jobsTermines: 0,
    jobsEnCours: 0,
    jobsEnErreur: 0,
    lignesTraiteesTotal: 0,
    tauxSucces: 0,
    tempsTraitementMoyen: 0
  };

  // Types de transformation disponibles
  typesTransformation = [
    {
      id: 'mapping',
      nom: 'Mapping de Données',
      description: 'Transformation et mapping des données sources vers le format IFRS17',
      icone: 'fas fa-exchange-alt',
      couleur: '#3498db'
    },
    {
      id: 'calculation',
      nom: 'Calculs Actuariels',
      description: 'Calcul automatique des métriques IFRS17 (CSM, LRC, LIC, RA)',
      icone: 'fas fa-calculator',
      couleur: '#2ecc71'
    },
    {
      id: 'aggregation',
      nom: 'Agrégation',
      description: 'Agrégation des données par cohorte, ligne d\'activité ou période',
      icone: 'fas fa-layer-group',
      couleur: '#f39c12'
    },
    {
      id: 'validation',
      nom: 'Validation Qualité',
      description: 'Contrôles de cohérence et validation des données IFRS17',
      icone: 'fas fa-check-circle',
      couleur: '#9b59b6'
    }
  ];

  // Configuration du processus
  configurationTransformation = {
    autoValidation: true,
    toleranceErreur: 0.1,
    sauvegardePeriodique: true,
    notificationsEmail: false,
    formatSortie: 'JSON'
  };

  constructor(private ifrs17Service: IFRS17ApiService) {}

  ngOnInit(): void {
    this.calculerMetriques();
    this.setupAutoRefresh();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  // ================================
  // 📊 CALCULS ET MÉTRIQUES
  // ================================

  private calculerMetriques(): void {
    this.metriquesTransformation = {
      totalJobs: this.transformationJobs.length,
      jobsTermines: this.transformationJobs.filter(j => j.statut === 'TERMINE').length,
      jobsEnCours: this.transformationJobs.filter(j => j.statut === 'EN_COURS').length,
      jobsEnErreur: this.transformationJobs.filter(j => j.statut === 'ERREUR').length,
      lignesTraiteesTotal: this.transformationJobs.reduce((sum, j) => sum + j.lignesTraitees, 0),
      tauxSucces: this.transformationJobs.length > 0 ? 
        (this.transformationJobs.filter(j => j.statut === 'TERMINE').length / this.transformationJobs.length) * 100 : 0,
      tempsTraitementMoyen: this.calculerTempsMoyen()
    };
  }

  private calculerTempsMoyen(): number {
    const jobsTermines = this.transformationJobs.filter(j => j.statut === 'TERMINE' && j.dateFin);
    if (jobsTermines.length === 0) return 0;
    
    const tempsTotal = jobsTermines.reduce((sum, job) => {
      const duree = job.dateFin!.getTime() - job.dateDebut.getTime();
      return sum + duree;
    }, 0);
    
    return tempsTotal / jobsTermines.length / 60000; // en minutes
  }

  private setupAutoRefresh(): void {
    setInterval(() => {
      this.mettreAJourProgression();
    }, 5000); // Mise à jour toutes les 5 secondes
  }

  private mettreAJourProgression(): void {
    // Simulation de la progression des jobs en cours
    this.transformationJobs.forEach(job => {
      if (job.statut === 'EN_COURS' && job.progression < 100) {
        job.progression = Math.min(100, job.progression + Math.random() * 10);
        job.lignesTraitees = Math.floor((job.progression / 100) * job.lignesTotal);
        
        if (job.progression >= 100) {
          job.statut = 'TERMINE';
          job.dateFin = new Date();
        }
      }
    });
    
    this.calculerMetriques();
  }

  // ================================
  // 🔄 ACTIONS TRANSFORMATION
  // ================================

  demarrerTransformation(type: string): void {
    const nouveauJob: TransformationJob = {
      id: this.transformationJobs.length + 1,
      nom: `Transformation ${type} - ${new Date().toLocaleTimeString()}`,
      type: type as any,
      statut: 'EN_COURS',
      progression: 0,
      fichierSource: 'nouveau_fichier.xlsx',
      fichierCible: `resultat_${Date.now()}.json`,
      lignesTraitees: 0,
      lignesTotal: Math.floor(Math.random() * 50000) + 10000,
      erreurs: [],
      dateDebut: new Date()
    };

    this.transformationJobs.unshift(nouveauJob);
    console.log(`🔄 Démarrage transformation: ${type}`);
  }

  arreterTransformation(jobId: number): void {
    const job = this.transformationJobs.find(j => j.id === jobId);
    if (job && job.statut === 'EN_COURS') {
      job.statut = 'ERREUR';
      job.erreurs.push('Transformation interrompue par l\'utilisateur');
      console.log(`⏹️ Arrêt transformation: ${job.nom}`);
    }
  }

  relancerTransformation(jobId: number): void {
    const job = this.transformationJobs.find(j => j.id === jobId);
    if (job && (job.statut === 'ERREUR' || job.statut === 'TERMINE')) {
      job.statut = 'EN_COURS';
      job.progression = 0;
      job.lignesTraitees = 0;
      job.erreurs = [];
      job.dateDebut = new Date();
      job.dateFin = undefined;
      console.log(`🔄 Relance transformation: ${job.nom}`);
    }
  }

  supprimerTransformation(jobId: number): void {
    const index = this.transformationJobs.findIndex(j => j.id === jobId);
    if (index > -1) {
      this.transformationJobs.splice(index, 1);
      this.calculerMetriques();
      console.log(`🗑️ Suppression transformation: ${jobId}`);
    }
  }

  // ================================
  // 📁 GESTION DES FICHIERS
  // ================================

  uploaderFichier(event: any): void {
    const fichier = event.target.files[0];
    if (fichier) {
      console.log(`📁 Upload fichier: ${fichier.name}`);
      
      // Simulation de l'upload
      const nouveauJob: TransformationJob = {
        id: this.transformationJobs.length + 1,
        nom: `Traitement ${fichier.name}`,
        type: 'mapping',
        statut: 'EN_ATTENTE',
        progression: 0,
        fichierSource: fichier.name,
        fichierCible: `traite_${fichier.name.replace('.xlsx', '.json')}`,
        lignesTraitees: 0,
        lignesTotal: Math.floor(Math.random() * 100000) + 50000,
        erreurs: [],
        dateDebut: new Date()
      };

      this.transformationJobs.unshift(nouveauJob);
    }
  }

  telechargerResultat(job: TransformationJob): void {
    if (job.statut === 'TERMINE') {
      console.log(`⬇️ Téléchargement: ${job.fichierCible}`);
      // Simulation du téléchargement
      const donnees = this.genererDonneesResultat(job);
      this.telechargerFichier(donnees, job.fichierCible);
    }
  }

  private genererDonneesResultat(job: TransformationJob): any {
    // Génération de données d'exemple selon le type
    switch (job.type) {
      case 'mapping':
        return {
          metadata: {
            job_id: job.id,
            nom: job.nom,
            lignes_traitees: job.lignesTraitees,
            date_traitement: job.dateFin
          },
          donnees_mappees: Array.from({length: 100}, (_, i) => ({
            id_contrat: `CTR_${i + 1}`,
            cohorte: `AUTO_2024_Q${Math.floor(i/25) + 1}`,
            lrc: Math.random() * 100000 + 50000,
            csm: Math.random() * 20000 + 5000
          }))
        };
      
      case 'calculation':
        return {
          metadata: {
            job_id: job.id,
            calculs_realises: ['CSM', 'LRC', 'Risk_Adjustment'],
            date_calcul: job.dateFin
          },
          resultats: {
            csm_total: 325000000,
            lrc_total: 2450000000,
            ra_total: 125000000
          }
        };
      
      default:
        return { job_id: job.id, statut: 'complete' };
    }
  }

  private telechargerFichier(donnees: any, nomFichier: string): void {
    const blob = new Blob([JSON.stringify(donnees, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = nomFichier;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  // ================================
  // 🔧 MÉTHODES UTILITAIRES
  // ================================

  getStatutColor(statut: string): string {
    const colorMap: { [key: string]: string } = {
      'EN_ATTENTE': '#f39c12',
      'EN_COURS': '#3498db',
      'TERMINE': '#27ae60',
      'ERREUR': '#e74c3c'
    };
    return colorMap[statut] || '#6c757d';
  }

  getTypeIcon(type: string): string {
    const typeInfo = this.typesTransformation.find(t => t.id === type);
    return typeInfo?.icone || 'fas fa-cog';
  }

  getTypeColor(type: string): string {
    const typeInfo = this.typesTransformation.find(t => t.id === type);
    return typeInfo?.couleur || '#6c757d';
  }

  formatDuree(debut: Date, fin?: Date): string {
    const finDate = fin || new Date();
    const duree = finDate.getTime() - debut.getTime();
    const minutes = Math.floor(duree / 60000);
    const secondes = Math.floor((duree % 60000) / 1000);
    
    if (minutes > 0) {
      return `${minutes}m ${secondes}s`;
    }
    return `${secondes}s`;
  }

  formatTaille(lignes: number): string {
    if (lignes > 1000000) {
      return `${(lignes / 1000000).toFixed(1)}M lignes`;
    } else if (lignes > 1000) {
      return `${(lignes / 1000).toFixed(1)}K lignes`;
    }
    return `${lignes} lignes`;
  }
}