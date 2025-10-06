import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthService, User, UserProgress, UserLevel } from '../../services/auth.service';
import { DashboardService, DashboardResponse, Alert, RecommendedAction } from '../../services/dashboard.service';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard-professional.scss'],
  standalone: true,
  imports: [CommonModule]
})
export class DashboardComponent implements OnInit, OnDestroy {
  dashboardData: DashboardResponse | null = null;
  currentUser: User | null = null;
  userProgress: UserProgress | null = null;
  alerts: Alert[] = [];
  recommendedActions: RecommendedAction[] = [];
  isLoading = true;
  currentDate = new Date();
  
  // Subscriptions
  private subscriptions: Subscription[] = [];

  constructor(
    public authService: AuthService,
    private dashboardService: DashboardService,
    public router: Router
  ) {}

  ngOnInit() {
    this.loadUserData();
    this.loadDashboardData();
    this.setupSubscriptions();
  }

  ngOnDestroy() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  private loadUserData() {
    this.currentUser = this.authService.getCurrentUser();
    this.userProgress = this.authService.getUserProgress();
  }

  private loadDashboardData() {
    this.isLoading = true;
    
    const dashboardSub = this.dashboardService.getUnifiedDashboard().subscribe({
      next: (dashboard) => {
        this.dashboardData = dashboard;
        this.alerts = dashboard.alerts;
        this.recommendedActions = dashboard.recommended_actions;
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Erreur lors du chargement du dashboard:', error);
        this.isLoading = false;
      }
    });
    
    this.subscriptions.push(dashboardSub);
  }

  private setupSubscriptions() {
    // S'abonner aux mises à jour du dashboard
    const dashboardSub = this.dashboardService.dashboardData$.subscribe(
      dashboard => {
        if (dashboard) {
          this.dashboardData = dashboard;
        }
      }
    );
    
    // S'abonner aux alertes
    const alertsSub = this.dashboardService.alerts$.subscribe(
      alerts => {
        this.alerts = alerts;
      }
    );
    
    this.subscriptions.push(dashboardSub, alertsSub);
  }

  // 🎯 Méthodes d'interaction avec le dashboard

  /**
   * Rafraîchir le dashboard
   */
  refreshDashboard() {
    this.isLoading = true;
    this.dashboardService.refreshDashboard();
  }

  /**
   * Obtenir la classe CSS pour le niveau utilisateur
   */
  getLevelClass(): string {
    if (!this.currentUser) return '';
    
    switch (this.currentUser.level) {
      case UserLevel.DEBUTANT:
        return 'level-debutant';
      case UserLevel.INTERMEDIAIRE:
        return 'level-intermediaire';
      case UserLevel.EXPERT:
        return 'level-expert';
      case UserLevel.MAITRE_IFRS17:
        return 'level-maitre';
      default:
        return '';
    }
  }

  /**
   * Obtenir l'icône pour le niveau utilisateur
   */
  getLevelIcon(): string {
    if (!this.currentUser) return '🌱';
    
    switch (this.currentUser.level) {
      case UserLevel.DEBUTANT:
        return '🌱';
      case UserLevel.INTERMEDIAIRE:
        return '🌿';
      case UserLevel.EXPERT:
        return '🌳';
      case UserLevel.MAITRE_IFRS17:
        return '🏆';
      default:
        return '🌱';
    }
  }

  /**
   * Formater les nombres en devise
   */
  formatCurrency(value: number): string {
    return new Intl.NumberFormat('fr-TN', {
      style: 'currency',
      currency: 'TND',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  }

  /**
   * Formater les pourcentages
   */
  formatPercentage(value: number): string {
    return new Intl.NumberFormat('fr-TN', {
      style: 'percent',
      minimumFractionDigits: 1,
      maximumFractionDigits: 2
    }).format(value / 100);
  }

  /**
   * Obtenir la classe CSS pour les indicateurs
   */
  getIndicatorClass(value: number, threshold: number = 90): string {
    if (value >= threshold) return 'indicator-success';
    if (value >= threshold * 0.75) return 'indicator-warning';
    return 'indicator-danger';
  }

  /**
   * Obtenir la date actuelle formatée
   */
  getCurrentDate(): string {
    return new Date().toLocaleDateString('fr-FR', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  }

  /**
   * Obtenir l'heure actuelle formatée
   */
  getCurrentTime(): string {
    return new Date().toLocaleTimeString('fr-FR', {
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  /**
   * Naviguer vers une route
   */
  navigateTo(route: string): void {
    this.router.navigate([route]);
  }

  /**
   * Fermer une alerte
   */
  dismissAlert(alert: Alert): void {
    this.alerts = this.alerts.filter(a => a !== alert);
  }

  /**
   * Exécuter une action recommandée
   */
  executeAction(action: RecommendedAction): void {
    if (action.action_url) {
      this.navigateTo(action.action_url);
    }
  }
}