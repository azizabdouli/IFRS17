// src/app/services/ifrs17-api.service.ts

import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable, BehaviorSubject, throwError } from 'rxjs';
import { catchError, map, retry } from 'rxjs/operators';
import {
  ContratAssurance,
  Cohorte,
  RapportIFRS17,
  FluxTresorerie,
  CalculActuariel,
  ParametrageIFRS17
} from '../models/ifrs17.models';

/**
 * 🏢 Service API IFRS17 - Interface avec le backend FastAPI
 * Gère toutes les communications avec l'API backend pour les données IFRS17
 */

@Injectable({
  providedIn: 'root'
})
export class IFRS17ApiService {
  private readonly baseUrl = 'http://127.0.0.1:8001';
  private readonly apiVersion = 'v1';
  
  // Observables pour le state management
  private contratsSubject = new BehaviorSubject<ContratAssurance[]>([]);
  private cohortesSubject = new BehaviorSubject<Cohorte[]>([]);
  private loadingSubject = new BehaviorSubject<boolean>(false);
  
  public contrats$ = this.contratsSubject.asObservable();
  public cohortes$ = this.cohortesSubject.asObservable();
  public loading$ = this.loadingSubject.asObservable();

  constructor(private http: HttpClient) {
    this.initializeHeaders();
  }

  private httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-API-Version': this.apiVersion
    })
  };

  private initializeHeaders(): void {
    // Configuration des en-têtes par défaut
    this.httpOptions.headers = this.httpOptions.headers.set(
      'X-Client-Type', 'Angular-IFRS17'
    );
  }

  // =================================
  // 🏢 GESTION DES CONTRATS
  // =================================

  /**
   * Récupère tous les contrats d'assurance
   */
  getContrats(filtres?: any): Observable<ContratAssurance[]> {
    this.setLoading(true);
    
    let params = new HttpParams();
    if (filtres) {
      Object.keys(filtres).forEach(key => {
        if (filtres[key] !== null && filtres[key] !== undefined) {
          params = params.set(key, filtres[key].toString());
        }
      });
    }

    return this.http.get<ContratAssurance[]>(`${this.baseUrl}/contrats`, 
      { ...this.httpOptions, params })
      .pipe(
        map(contrats => {
          this.contratsSubject.next(contrats);
          this.setLoading(false);
          return contrats;
        }),
        catchError(this.handleError),
        retry(2)
      );
  }

  /**
   * Récupère un contrat spécifique par ID
   */
  getContratById(id: string): Observable<ContratAssurance> {
    return this.http.get<ContratAssurance>(`${this.baseUrl}/contrats/${id}`, this.httpOptions)
      .pipe(
        catchError(this.handleError),
        retry(1)
      );
  }

  /**
   * Crée un nouveau contrat
   */
  creerContrat(contrat: Partial<ContratAssurance>): Observable<ContratAssurance> {
    return this.http.post<ContratAssurance>(`${this.baseUrl}/contrats`, contrat, this.httpOptions)
      .pipe(
        map(nouveauContrat => {
          // Mettre à jour la liste des contrats
          const contratsActuels = this.contratsSubject.value;
          this.contratsSubject.next([...contratsActuels, nouveauContrat]);
          return nouveauContrat;
        }),
        catchError(this.handleError)
      );
  }

  /**
   * Met à jour un contrat existant
   */
  mettreAJourContrat(id: string, contrat: Partial<ContratAssurance>): Observable<ContratAssurance> {
    return this.http.put<ContratAssurance>(`${this.baseUrl}/contrats/${id}`, contrat, this.httpOptions)
      .pipe(
        map(contratMisAJour => {
          // Mettre à jour la liste des contrats
          const contratsActuels = this.contratsSubject.value;
          const index = contratsActuels.findIndex(c => c.id === id);
          if (index !== -1) {
            contratsActuels[index] = contratMisAJour;
            this.contratsSubject.next([...contratsActuels]);
          }
          return contratMisAJour;
        }),
        catchError(this.handleError)
      );
  }

  /**
   * Supprime un contrat
   */
  supprimerContrat(id: string): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/contrats/${id}`, this.httpOptions)
      .pipe(
        map(() => {
          // Mettre à jour la liste des contrats
          const contratsActuels = this.contratsSubject.value;
          this.contratsSubject.next(contratsActuels.filter(c => c.id !== id));
        }),
        catchError(this.handleError)
      );
  }

  // =================================
  // 🎯 GESTION DES COHORTES
  // =================================

  /**
   * Récupère toutes les cohortes
   */
  getCohortes(): Observable<Cohorte[]> {
    return this.http.get<Cohorte[]>(`${this.baseUrl}/cohortes`, this.httpOptions)
      .pipe(
        map(cohortes => {
          this.cohortesSubject.next(cohortes);
          return cohortes;
        }),
        catchError(this.handleError)
      );
  }

  /**
   * Crée une nouvelle cohorte
   */
  creerCohorte(cohorte: Partial<Cohorte>): Observable<Cohorte> {
    return this.http.post<Cohorte>(`${this.baseUrl}/cohortes`, cohorte, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  /**
   * Analyse de profitabilité par cohorte
   */
  analyserProfitabiliteCohorte(cohorteId: string): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/ml/predict/profitability`, 
      { cohorte_id: cohorteId }, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  // =================================
  // 💰 CALCULS IFRS17
  // =================================

  /**
   * Calcule la LRC (Liability for Remaining Coverage)
   */
  calculerLRC(contratId: string, parametres?: any): Observable<number> {
    return this.http.post<{ lrc: number }>(`${this.baseUrl}/calculs/lrc`, 
      { contrat_id: contratId, ...parametres }, this.httpOptions)
      .pipe(
        map(response => response.lrc),
        catchError(this.handleError)
      );
  }

  /**
   * Calcule la LIC (Liability for Incurred Claims)
   */
  calculerLIC(contratId: string): Observable<number> {
    return this.http.post<{ lic: number }>(`${this.baseUrl}/calculs/lic`, 
      { contrat_id: contratId }, this.httpOptions)
      .pipe(
        map(response => response.lic),
        catchError(this.handleError)
      );
  }

  /**
   * Calcule la CSM (Contractual Service Margin)
   */
  calculerCSM(contratId: string): Observable<number> {
    return this.http.post<{ csm: number }>(`${this.baseUrl}/calculs/csm`, 
      { contrat_id: contratId }, this.httpOptions)
      .pipe(
        map(response => response.csm),
        catchError(this.handleError)
      );
  }

  /**
   * Détecte les contrats onéreux
   */
  detecterContratsOnereux(): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/ml/predict/onerous-contracts`, 
      {}, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  // =================================
  // 📊 MACHINE LEARNING & IA
  // =================================

  /**
   * Prédiction de risque de contrat
   */
  predireRisqueContrat(contratId: string): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/ml/predict/risk-classification`, 
      { contrat_id: contratId }, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  /**
   * Prédiction de sinistres
   */
  predireSinistres(parametres: any): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/ml/predict/claims-prediction`, 
      parametres, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  /**
   * Chat avec l'assistant IA IFRS17
   */
  chatAvecAssistantIA(message: string): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/ai/chat`, 
      { message }, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  /**
   * Analyse automatique de données
   */
  analyseAutomatiqueDonnees(fichier: FormData): Observable<any> {
    const headers = new HttpHeaders({
      'Accept': 'application/json'
      // Ne pas définir Content-Type pour FormData
    });

    return this.http.post<any>(`${this.baseUrl}/ai/analyze-file`, 
      fichier, { headers })
      .pipe(catchError(this.handleError));
  }

  // =================================
  // 📈 REPORTING IFRS17
  // =================================

  /**
   * Génère un rapport IFRS17
   */
  genererRapportIFRS17(periode: string, typeRapport: string): Observable<RapportIFRS17> {
    return this.http.post<RapportIFRS17>(`${this.baseUrl}/reporting/ifrs17`, 
      { periode, type_rapport: typeRapport }, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  /**
   * Exporte les données vers Excel
   */
  exporterVersExcel(donnees: any, format: string = 'xlsx'): Observable<Blob> {
    return this.http.post(`${this.baseUrl}/export/excel`, donnees, 
      { 
        responseType: 'blob',
        headers: this.httpOptions.headers
      })
      .pipe(catchError(this.handleError));
  }

  /**
   * Tableau de bord en temps réel
   */
  getTableauBordTempsReel(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/dashboard/realtime`, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  // =================================
  // 🔧 GESTION DES PARAMÈTRES
  // =================================

  /**
   * Récupère les paramètres IFRS17
   */
  getParametragesIFRS17(): Observable<ParametrageIFRS17[]> {
    return this.http.get<ParametrageIFRS17[]>(`${this.baseUrl}/parametrage/ifrs17`, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  /**
   * Met à jour les paramètres IFRS17
   */
  mettreAJourParametrage(parametrage: ParametrageIFRS17): Observable<ParametrageIFRS17> {
    return this.http.put<ParametrageIFRS17>(`${this.baseUrl}/parametrage/ifrs17/${parametrage.id}`, 
      parametrage, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  // =================================
  // 🏥 MONITORING & SANTÉ
  // =================================

  /**
   * Vérifie la santé de l'API
   */
  verifierSanteAPI(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/health`, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  /**
   * Récupère les statistiques de l'API
   */
  getStatistiquesAPI(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/stats`, this.httpOptions)
      .pipe(catchError(this.handleError));
  }

  // =================================
  // 🤖 MACHINE LEARNING
  // =================================

  /**
   * Récupère le résumé de tous les modèles ML entraînés avec leurs métriques
   */
  getModelsSummary(): Observable<any> {
    return this.http.get(`${this.baseUrl}/ml/models-summary`, this.httpOptions)
      .pipe(
        catchError(this.handleError),
        retry(1)
      );
  }

  /**
   * Prédiction LRC - Retourne les valeurs prédites de LRC pour tous les contrats
   */
  predictLRC(modelType: string = 'xgboost'): Observable<any> {
    return this.http.post(`${this.baseUrl}/ml/predict/lrc?model_type=${modelType}`, {}, this.httpOptions)
      .pipe(
        catchError(this.handleError),
        retry(1)
      );
  }

  // =================================
  // 🛠️ MÉTHODES UTILITAIRES
  // =================================

  private setLoading(loading: boolean): void {
    this.loadingSubject.next(loading);
  }

  private handleError(error: any): Observable<never> {
    console.error('Erreur API IFRS17:', error);
    
    let messageErreur = 'Une erreur inattendue s\'est produite';
    
    if (error.error instanceof ErrorEvent) {
      // Erreur côté client
      messageErreur = `Erreur client: ${error.error.message}`;
    } else {
      // Erreur côté serveur
      switch (error.status) {
        case 400:
          messageErreur = 'Requête invalide';
          break;
        case 401:
          messageErreur = 'Non autorisé';
          break;
        case 403:
          messageErreur = 'Accès interdit';
          break;
        case 404:
          messageErreur = 'Ressource non trouvée';
          break;
        case 500:
          messageErreur = 'Erreur interne du serveur';
          break;
        default:
          messageErreur = `Erreur ${error.status}: ${error.error?.message || error.message}`;
      }
    }

    return throwError(() => new Error(messageErreur));
  }

  // =================================
  // 📘 GESTION PAA (Premium Allocation Approach)
  // =================================

  /**
   * Initialise un groupe PAA avec des contrats
   */
  initPAAGroup(groupId: string, contracts: any[], revenuePattern: string = 'linear'): Observable<any> {
    return this.http.post(`${this.baseUrl}/paa/groups/init`, contracts, {
      params: { group_id: groupId, revenue_pattern: revenuePattern, persist: 'true' }
    }).pipe(catchError(this.handleError));
  }

  /**
   * Traite une période pour un groupe PAA
   */
  processPAAPeriod(
    groupId: string,
    periodStart: string,
    periodEnd: string,
    incurredClaims: number = 0,
    claimsPaid: number = 0
  ): Observable<any> {
    return this.http.post(`${this.baseUrl}/paa/groups/${groupId}/period`, null, {
      params: {
        period_start: periodStart,
        period_end: periodEnd,
        incurred_claims: incurredClaims.toString(),
        claims_paid: claimsPaid.toString(),
        persist: 'true'
      }
    }).pipe(catchError(this.handleError));
  }

  /**
   * Récupère l'état courant d'un groupe PAA
   */
  getPAAGroupState(groupId: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/paa/groups/${groupId}`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Liste tous les groupes PAA
   */
  listPAAGroups(portfolio?: string): Observable<any> {
    let params = new HttpParams();
    if (portfolio) {
      params = params.set('portfolio', portfolio);
    }
    return this.http.get(`${this.baseUrl}/paa/groups`, { params })
      .pipe(catchError(this.handleError));
  }

  /**
   * Récupère les mouvements IFRS17 d'un groupe
   */
  getPAAMovements(groupId: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/paa/groups/${groupId}/movements`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Effectue un stress test sur un groupe
   */
  paaStressTest(
    groupId: string,
    claimRatioShock: number = 0,
    expenseRatioShock: number = 0
  ): Observable<any> {
    return this.http.post(`${this.baseUrl}/paa/groups/${groupId}/stress-test`, null, {
      params: {
        claim_ratio_shock: claimRatioShock.toString(),
        expense_ratio_shock: expenseRatioShock.toString()
      }
    }).pipe(catchError(this.handleError));
  }

  /**
   * Récupère le résumé portfolio PAA
   */
  getPAAPortfolioSummary(): Observable<any> {
    return this.http.get(`${this.baseUrl}/paa/analytics/portfolio-summary`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Nettoie les observables lors de la destruction du service
   */
  ngOnDestroy(): void {
    this.contratsSubject.complete();
    this.cohortesSubject.complete();
    this.loadingSubject.complete();
  }
}