import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { PPNAService, PPNAMetrics } from '../../services/ppna.service';
import { PPNAUploadComponent } from './ppna-upload.component';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
  standalone: true,
  imports: [CommonModule, PPNAUploadComponent]
})
export class DashboardComponent implements OnInit {
  ppnaMetrics: PPNAMetrics | null = null;
  ppnaData: any[] = [];
  alertes: any[] = [];
  alerts: any[] = [];
  isLoading = true;

  constructor(private ppnaService: PPNAService, private router: Router) {}

  ngOnInit() {
    this.loadDashboardData();
  }

  private loadDashboardData() {
    this.ppnaService.getDashboardMetrics().subscribe({
      next: (metrics) => {
        this.ppnaMetrics = {
          ...metrics,
          onerous_contracts_count: (metrics as any).onerous_contracts_count || 0,
          loss_component: (metrics as any).loss_component || 0
        };
        this.alerts = this.alertes;
        if (this.ppnaMetrics) {
          this.checkForAlerts(this.ppnaMetrics);
        }
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Erreur lors du chargement des métriques PPNA:', error);
        this.isLoading = false;
      }
    });
  }

  private checkForAlerts(metrics: PPNAMetrics) {
    this.alertes = [];
    
    if (metrics.lrc_total < 0) {
      this.alertes.push({
        severity: 'danger',
        title: 'LRC Négatif',
        message: 'Le LRC total est négatif, vérifiez les calculs PAA'
      });
    }
    
    if (metrics.risk_adjustment > metrics.lrc_total * 0.1) {
      this.alertes.push({
        severity: 'warning', 
        title: 'Ajustement Risque Élevé',
        message: 'L\'ajustement pour risque dépasse 10% du LRC'
      });
    }
    
    this.alerts = this.alertes;
  }

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

  navigateTo(route: string, queryParams?: any) {
    if (queryParams) {
      this.router.navigate([route], { queryParams });
    } else {
      this.router.navigate([route]);
    }
  }
}