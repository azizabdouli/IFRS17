import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthService, User, UserProgress, UserLevel } from '../../services/auth.service';
import { DashboardService, DashboardResponse, Alert, RecommendedAction } from '../../services/dashboard.service';
import { PPNAService, PPNAMetrics } from '../../services/ppna.service';
import { NgChartsModule } from 'ng2-charts';
import { ChartConfiguration, ChartData } from 'chart.js';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard-professional.scss'],
  standalone: true,
  imports: [CommonModule, NgChartsModule]
})
export class DashboardComponent implements OnInit, OnDestroy {
  dashboardData: DashboardResponse | null = null;
  currentUser: User | null = null;
  userProgress: UserProgress | null = null;
  alerts: Alert[] = [];
  recommendedActions: RecommendedAction[] = [];
  isLoading = true;
  currentDate = new Date();
  // IFRS17 / PPNA metrics
  ppnaMetrics: PPNAMetrics | null = null;
  ppnaSegments: any[] = [];
  loadingPPNA = true;
  ppnaError: string | null = null;
  // Upload state
  uploadingPPNA = false;
  uploadProgress = 0;
  selectedFile: File | null = null;
  lastUploadedFileName: string | null = localStorage.getItem('ppnaLastFileName');
  actuarialNarrative: string[] = [];
  isDragging = false;
  // Chart data for LRC composition
  lrcChartData: ChartData<'doughnut'> | null = null;
  lrcChartOptions: ChartConfiguration<'doughnut'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom', labels: { usePointStyle: true } },
      tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${new Intl.NumberFormat('fr-FR',{style:'currency',currency:'TND',minimumFractionDigits:0}).format(Number(ctx.raw))}` } }
    },
    cutout: '60%'
  };
  
  // Subscriptions
  private subscriptions: Subscription[] = [];

  constructor(
    private authService: AuthService,
    private dashboardService: DashboardService,
    private ppnaService: PPNAService,
    private router: Router
  ) {}

  ngOnInit() {
    this.loadUserData();
    this.loadDashboardData();
    this.setupSubscriptions();
    this.initializePPNAMetrics();
  }

  ngOnDestroy() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  // Méthodes publiques pour accéder aux services depuis le template
  getFullName(): string {
    return this.authService.getFullName();
  }

  getAuthService(): AuthService {
    return this.authService;
  }

  /**
   * Obtenir un message de salutation personnalisé selon l'heure
   */
  getGreeting(): string {
    const hour = new Date().getHours();
    if (hour < 12) return 'Bonjour';
    if (hour < 18) return 'Bon après-midi';
    return 'Bonsoir';
  }

  /**
   * Obtenir la classe CSS pour les ratios
   */
  getRatioBadgeClass(ratio: number): string {
    if (ratio < 0.5) return 'ratio-low';
    if (ratio < 0.8) return 'ratio-medium';
    return 'ratio-high';
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

    // Souscription aux métriques PPNA (IFRS17)
    const ppnaMetricsSub = this.ppnaService.metrics$.subscribe(metrics => {
      if (metrics) {
        this.ppnaMetrics = metrics;
        this.loadingPPNA = false;
        this.updateLRCChart();
      }
    });
    this.subscriptions.push(ppnaMetricsSub);
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

  // =====================================================
  // 📘 IFRS17 / PPNA Integration
  // =====================================================

  private initializePPNAMetrics(): void {
    this.loadingPPNA = true;
    // Déclenche un rafraîchissement (loadInitialData est appelé dans le service constructeur)
    this.ppnaService.refreshMetrics();
    // Charger l'analyse des segments
    const segSub = this.ppnaService.analyzeBySegments().subscribe({
      next: (res: any) => {
        this.ppnaSegments = res?.segments || [];
      },
      error: (err) => {
        console.error('Erreur analyse segments PPNA:', err);
        this.ppnaError = err.message || 'Erreur récupération segments';
      }
    });
    this.subscriptions.push(segSub);
  }

  getRiskAdjustmentPercent(): number {
    if (!this.ppnaMetrics || !this.ppnaMetrics.lrc_total) return 0;
    return +( (this.ppnaMetrics.risk_adjustment / this.ppnaMetrics.lrc_total) * 100 ).toFixed(2);
  }

  getLossComponentPercent(): number {
    if (!this.ppnaMetrics || !this.ppnaMetrics.lrc_total || !this.ppnaMetrics.loss_component) return 0;
    return +( (this.ppnaMetrics.loss_component / this.ppnaMetrics.lrc_total) * 100 ).toFixed(2);
  }

  trackBySegment(index: number, item: any) { return item?.segment || index; }

  refreshPPNA(): void {
    this.loadingPPNA = true;
    this.ppnaService.refreshMetrics();
    const segRefreshSub = this.ppnaService.analyzeBySegments().subscribe({
      next: res => { this.ppnaSegments = res?.segments || []; },
      error: err => { this.ppnaError = err.message || 'Erreur récupération segments'; },
      complete: () => { this.loadingPPNA = false; }
    });
    this.subscriptions.push(segRefreshSub);
  }

  private updateLRCChart(): void {
    if (!this.ppnaMetrics) return;
    const ppna = this.ppnaMetrics.ppna_total || 0;
    const riskAdj = this.ppnaMetrics.risk_adjustment || 0;
    const lossComp = this.ppnaMetrics.loss_component || 0;
    // Remaining (if any) = LRC - (known components)
    const lrc = this.ppnaMetrics.lrc_total || 0;
    const known = ppna + riskAdj + lossComp;
    const autre = lrc > known ? lrc - known : 0;
    this.lrcChartData = {
      labels: ['PPNA', 'Risk Adj.', 'Loss Component', 'Autres'],
      datasets: [{
        data: [ppna, riskAdj, lossComp, autre],
        backgroundColor: ['#2563EB','#F59E0B','#DC2626','#9CA3AF'],
        borderWidth: 1,
        borderColor: '#FFFFFF'
      }]
    };
    this.buildActuarialNarrative(ppna, riskAdj, lossComp, autre, lrc);
  }

  // =====================================================
  // 📤 Upload PPNA
  // =====================================================
  onFileInputChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
    }
  }

  onDropFile(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = false;
    if (event.dataTransfer?.files?.length) {
      const file = event.dataTransfer.files[0];
      if (this.validateExcelFile(file)) {
        this.selectedFile = file;
      }
    }
  }

  onDragOver(event: DragEvent): void { 
    event.preventDefault(); 
    this.isDragging = true;
  }

  validateExcelFile(file: File): boolean {
    const valid = /(.xls|.xlsx)$/i.test(file.name);
    if (!valid) {
      this.ppnaError = 'Format invalide. Seuls les fichiers .xls / .xlsx sont acceptés';
    }
    return valid;
  }

  uploadPPNA(): void {
    if (!this.selectedFile) return;
    if (!this.validateExcelFile(this.selectedFile)) return;
    this.uploadingPPNA = true;
    this.ppnaError = null;
    this.ppnaService.uploadFile(this.selectedFile).subscribe({
      next: (res) => {
        this.uploadingPPNA = false;
        this.lastUploadedFileName = this.selectedFile?.name || null;
        if (this.lastUploadedFileName) {
          localStorage.setItem('ppnaLastFileName', this.lastUploadedFileName);
        }
        this.selectedFile = null;
        // Refresh segments post upload
        this.refreshPPNA();
      },
      error: (err) => {
        this.uploadingPPNA = false;
        this.ppnaError = err.message || 'Erreur upload fichier';
      }
    });
  }

  clearSelectedFile(): void { this.selectedFile = null; }

  // =====================================================
  // 🧾 Export / Narrative
  // =====================================================
  exportSegmentsCSV(): void {
    if (!this.ppnaSegments.length) return;
    const headers = ['Segment','Primes','Provisions','Ratio_Provisions_Primes','Part_Primes'];
    const rows = this.ppnaSegments.map(s => [
      (s.segment || s.code || ''),
      s.primes,
      s.provisions,
      s.ratio_provisions_primes,
      s.part_primes
    ]);
    const csv = [headers, ...rows].map(r => r.join(';')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'segments_ppna.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  private buildActuarialNarrative(ppna: number, riskAdj: number, lossComp: number, autre: number, lrc: number): void {
    const lines: string[] = [];
    const fmt = (v: number) => new Intl.NumberFormat('fr-FR',{style:'currency',currency:'TND',maximumFractionDigits:0}).format(v);
    lines.push(`La LRC totale s'élève à ${fmt(lrc)} dont une composante PPNA de ${fmt(ppna)}.`);
    if (riskAdj > 0) lines.push(`Le risk adjustment représente ${this.getRiskAdjustmentPercent()}% de la LRC (${fmt(riskAdj)}), reflétant l'incertitude non diversifiée.`);
    if (lossComp > 0) lines.push(`Une loss component de ${fmt(lossComp)} est reconnue, indiquant des contrats potentiellement déficitaires.`);
    if (autre > 0) lines.push(`La différence résiduelle ('Autres') de ${fmt(autre)} couvre les ajustements techniques ou arrondis.`);
    if (this.ppnaSegments?.length) {
      const top = this.ppnaSegments[0];
      lines.push(`Le segment principal (${top.segment || top.code}) concentre ${ (top.part_primes || 0).toFixed(2)}% des primes et un ratio provisions/primes de ${(top.ratio_provisions_primes || 0).toFixed(2)}%.`);
    }
    this.actuarialNarrative = lines;
  }
}