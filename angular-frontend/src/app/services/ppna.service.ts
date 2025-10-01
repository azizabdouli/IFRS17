// angular-frontend/src/app/services/ppna.service.ts

import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError, BehaviorSubject } from 'rxjs';
import { catchError, map } from 'rxjs/operators';

export interface PPNAMetrics {
  lrc_total: number;
  ppna_total: number;
  risk_adjustment: number;
  lic_total: number;
  csm_total: number;
  contrats_onereux: number;
  onerous_contracts_count: number;
  loss_component: number;
  approche: string;
  derniere_maj: string;
}

export interface LRCCalculation {
  status: string;
  metriques: {
    lrc_total: number;
    ppna_total: number;
    risk_adjustment: number;
    primes_totales: number;
  };
  analyse_segments?: any[];
  contrats_onereux: {
    detected: boolean;
    nombre_contrats_onereux: number;
    ratio_moyen_onereux: number;
    total_provisions_onereuses: number;
  };
}

export interface PPNASheet {
  sheet_name: string;
  total_rows: number;
  total_columns: number;
  columns: string[];
  data: any[];
}

@Injectable({
  providedIn: 'root'
})
export class PPNAService {
  private readonly baseUrl = 'http://localhost:8001/ppna';
  private metricsSubject = new BehaviorSubject<PPNAMetrics | null>(null);
  public metrics$ = this.metricsSubject.asObservable();

  constructor(private http: HttpClient) {
    this.loadInitialData();
  }

  private handleError(error: HttpErrorResponse) {
    console.error('Erreur API PPNA:', error);
    let errorMessage = 'Erreur inconnue';
    
    if (error.error instanceof ErrorEvent) {
      errorMessage = `Erreur client: ${error.error.message}`;
    } else {
      errorMessage = `Erreur serveur ${error.status}: ${error.error?.detail || error.message}`;
    }
    
    return throwError(() => new Error(errorMessage));
  }

  /**
   * Charge les données PPNA initiales
   */
  loadInitialData(): Observable<any> {
    return this.http.get(`${this.baseUrl}/load-data`)
      .pipe(
        catchError(this.handleError),
        map((response: any) => {
          if (response.status === 'success') {
            this.refreshMetrics();
          }
          return response;
        })
      );
  }

  /**
   * Obtient les métriques du dashboard
   */
  getDashboardMetrics(): Observable<PPNAMetrics> {
    return this.http.get<{metrics: PPNAMetrics}>(`${this.baseUrl}/dashboard-metrics`)
      .pipe(
        catchError(this.handleError),
        map(response => {
          const metrics = response.metrics;
          this.metricsSubject.next(metrics);
          return metrics;
        })
      );
  }

  /**
   * Rafraîchit les métriques
   */
  refreshMetrics(): void {
    this.getDashboardMetrics().subscribe({
      next: (metrics) => console.log('Métriques PPNA mises à jour:', metrics),
      error: (err) => console.error('Erreur mise à jour métriques:', err)
    });
  }

  /**
   * Calcule la LRC selon l'approche PAA
   */
  calculateLRC(sheetName?: string): Observable<LRCCalculation> {
    let params: { [key: string]: string } = {};
    if (sheetName) {
      params['sheet_name'] = sheetName;
    }
    
    return this.http.get<LRCCalculation>(`${this.baseUrl}/calculate-lrc`, { params })
      .pipe(
        catchError(this.handleError)
      );
  }

  /**
   * Obtient la liste des feuilles Excel disponibles
   */
  getAvailableSheets(): Observable<{sheets: string[], total_sheets: number}> {
    return this.http.get<{sheets: string[], total_sheets: number}>(`${this.baseUrl}/sheets`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Obtient les données d'une feuille spécifique
   */
  getSheetData(sheetName: string, limit: number = 100): Observable<PPNASheet> {
    return this.http.get<PPNASheet>(`${this.baseUrl}/sheet-data/${encodeURIComponent(sheetName)}?limit=${limit}`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Analyse par segments
   */
  analyzeBySegments(sheetName?: string): Observable<any> {
    let params: { [key: string]: string } = {};
    if (sheetName) {
      params['sheet_name'] = sheetName;
    }
    
    return this.http.get(`${this.baseUrl}/analysis/segments`, { params })
      .pipe(catchError(this.handleError));
  }

  /**
   * Analyse des contrats onéreux
   */
  analyzeOnerousContracts(sheetName?: string): Observable<any> {
    let params: { [key: string]: string } = {};
    if (sheetName) {
      params['sheet_name'] = sheetName;
    }
    
    return this.http.get(`${this.baseUrl}/analysis/onerous-contracts`, { params })
      .pipe(catchError(this.handleError));
  }

  /**
   * Upload d'un nouveau fichier PPNA
   */
  uploadFile(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post(`${this.baseUrl}/upload-file`, formData)
      .pipe(
        catchError(this.handleError),
        map((response: any) => {
          if (response.status === 'success') {
            this.refreshMetrics();
          }
          return response;
        })
      );
  }

  /**
   * Obtient les métriques actuelles (synchrone)
   */
  getCurrentMetrics(): PPNAMetrics | null {
    return this.metricsSubject.value;
  }

  /**
   * Formate les montants financiers en Dinar Tunisien (TND)
   */
  formatCurrency(amount: number): string {
    return new Intl.NumberFormat('ar-TN', {
      style: 'currency',
      currency: 'TND',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  }

  /**
   * Formate les pourcentages (style tunisien)
   */
  formatPercentage(value: number): string {
    return new Intl.NumberFormat('ar-TN', {
      style: 'percent',
      minimumFractionDigits: 1,
      maximumFractionDigits: 2
    }).format(value / 100);
  }
}