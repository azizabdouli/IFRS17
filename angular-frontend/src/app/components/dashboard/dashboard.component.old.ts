import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthService, User, UserProgress, UserLevel } from '../../services/auth.service';
import { DashboardService, DashboardResponse, Alert, RecommendedAction } from '../../services/dashboard.service';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
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
    private authService: AuthService,
    private dashboardService: DashboardService,
    private router: Router
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
    this.alerts = [];
    
    if (metrics.lrc_total < 0) {
      this.alerts.push({
        severity: 'danger',
        title: 'LRC Négatif',
        message: 'Le LRC total est négatif, vérifiez les calculs PAA'
      });
    }
    
    if (metrics.risk_adjustment > metrics.lrc_total * 0.1) {
      this.alerts.push({
        severity: 'warning', 
        title: 'Ajustement Risque Élevé',
        message: 'L\'ajustement pour risque dépasse 10% du LRC'
      });
    }
  }

  // 🎯 Méthodes de navigation
  navigateTo(route: string, queryParams?: any) {
    if (queryParams) {
      this.router.navigate([route], { queryParams });
    } else {
      this.router.navigate([route]);
    }
  }

  // 📤 Méthode d'export
  exportData() {
    // Logique d'export à implémenter
    console.log('Export des données en cours...');
  }

  // 🔧 Méthodes utilitaires
  formatCurrency(value: number): string {
    return new Intl.NumberFormat('fr-TN', {
      style: 'currency',
      currency: 'TND',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  }

  formatPercentage(value: number): string {
    return new Intl.NumberFormat('fr-FR', {
      style: 'percent',
      minimumFractionDigits: 1,
      maximumFractionDigits: 1
    }).format(value / 100);
  }

  getRatioClass(ratio: number): string {
    if (ratio > 100) return 'text-danger font-weight-bold';
    if (ratio > 80) return 'text-warning font-weight-bold';
    return 'text-success';
  }
}