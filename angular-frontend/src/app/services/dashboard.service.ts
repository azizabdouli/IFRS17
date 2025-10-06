import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export interface KPIMetrics {
  ppna_count: number;
  onerous_contracts: number;
  compliance_score: number;
  monthly_trend: number;
  accuracy_rate: number;
  processing_time: number;
}

export interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  priority: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  action_url?: string;
  action_text?: string;
  created_at: string;
}

export interface RecommendedAction {
  id: string;
  title: string;
  description: string;
  category: string;
  priority: 'low' | 'medium' | 'high';
  estimated_time: number;
  points_reward: number;
  action_url: string;
  icon: string;
}

export interface DashboardResponse {
  user_id: number;
  kpis: KPIMetrics;
  alerts: Alert[];
  recommended_actions: RecommendedAction[];
  weekly_summary: {
    tasks_completed: number;
    points_earned: number;
    badges_earned: string[];
    accuracy_avg: number;
  };
  achievements: {
    recent_badges: string[];
    next_level_progress: number;
    total_achievements: number;
  };
  contextual_insights: string[];
}

@Injectable({
  providedIn: 'root'
})
export class DashboardService {
  private readonly API_URL = environment.apiUrl;
  
  // BehaviorSubjects pour les données en temps réel
  private dashboardDataSubject = new BehaviorSubject<DashboardResponse | null>(null);
  public dashboardData$ = this.dashboardDataSubject.asObservable();
  
  private alertsSubject = new BehaviorSubject<Alert[]>([]);
  public alerts$ = this.alertsSubject.asObservable();

  constructor(private http: HttpClient) {}

  /**
   * Récupérer le dashboard unifié
   */
  getUnifiedDashboard(): Observable<DashboardResponse> {
    return this.http.get<DashboardResponse>(`${this.API_URL}/dashboard/`).pipe(
      tap(dashboard => {
        this.dashboardDataSubject.next(dashboard);
        this.alertsSubject.next(dashboard.alerts);
      }),
      catchError(this.handleError)
    );
  }

  /**
   * Récupérer les alertes contextuelles
   */
  getAlerts(): Observable<{ alerts_count: number; alerts: Alert[] }> {
    return this.http.get<{ alerts_count: number; alerts: Alert[] }>(`${this.API_URL}/dashboard/alerts`).pipe(
      tap(response => {
        this.alertsSubject.next(response.alerts);
      }),
      catchError(this.handleError)
    );
  }

  /**
   * Récupérer les actions recommandées
   */
  getRecommendedActions(): Observable<{ user_level: string; recommended_actions: RecommendedAction[] }> {
    return this.http.get<{ user_level: string; recommended_actions: RecommendedAction[] }>(`${this.API_URL}/dashboard/recommended-actions`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Récupérer la progression utilisateur
   */
  getUserProgress(): Observable<any> {
    return this.http.get(`${this.API_URL}/dashboard/user-progress`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Attribuer des points pour une action
   */
  awardPoints(points: number, action?: string): Observable<any> {
    const url = `${this.API_URL}/dashboard/award-points/${points}`;
    const params: any = {};
    if (action) {
      params.action = action;
    }
    
    return this.http.post(url, {}, { params }).pipe(
      tap(() => {
        // Recharger le dashboard après attribution de points
        this.refreshDashboard();
      }),
      catchError(this.handleError)
    );
  }

  /**
   * Rafraîchir les données du dashboard
   */
  refreshDashboard(): void {
    this.getUnifiedDashboard().subscribe({
      next: (dashboard) => {
        console.log('Dashboard mis à jour avec succès');
      },
      error: (error) => {
        console.error('Erreur lors du rafraîchissement du dashboard:', error);
      }
    });
  }

  /**
   * Obtenir les données actuelles du dashboard
   */
  getCurrentDashboardData(): DashboardResponse | null {
    return this.dashboardDataSubject.value;
  }

  /**
   * Obtenir les alertes actuelles
   */
  getCurrentAlerts(): Alert[] {
    return this.alertsSubject.value;
  }

  /**
   * Marquer une alerte comme lue/résolue
   */
  markAlertAsRead(alertId: string): void {
    const currentAlerts = this.alertsSubject.value;
    const updatedAlerts = currentAlerts.filter(alert => alert.id !== alertId);
    this.alertsSubject.next(updatedAlerts);
  }

  /**
   * Filtrer les alertes par priorité
   */
  getAlertsByPriority(priority: 'low' | 'medium' | 'high' | 'critical'): Alert[] {
    return this.alertsSubject.value.filter(alert => alert.priority === priority);
  }

  /**
   * Obtenir le nombre d'alertes critiques
   */
  getCriticalAlertsCount(): number {
    return this.alertsSubject.value.filter(alert => alert.priority === 'critical').length;
  }

  /**
   * Gestion des erreurs
   */
  private handleError = (error: any): Observable<never> => {
    console.error('Erreur DashboardService:', error);
    throw error;
  }
}