import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IFRS17ApiService } from '../../services/ifrs17-api.service';
import { Subject, takeUntil } from 'rxjs';

interface PAAGroup {
  group_id: string;
  portfolio: string;
  lrc_current: number;
  lic_current: number;
  unearned_premium: number;
  onerous_flag: boolean;
  created_at?: string;
}

interface PAAMovement {
  period_label: string;
  earned_premium: number;
  change_in_lrc: number;
  claims_incurred: number;
  claims_paid: number;
  lrc_end: number;
  lic_end: number;
  unearned_premium_end: number;
  onerous_flag: boolean;
}

interface PortfolioSummary {
  total_groups: number;
  total_lrc: number;
  total_lic: number;
  total_unearned_premium: number;
  onerous_groups: number;
  onerous_ratio: number;
}

@Component({
  selector: 'app-paa-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './paa-dashboard.component.html',
  styleUrl: './paa-dashboard.component.scss'
})
export class PaaDashboardComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  // État principal
  groups: PAAGroup[] = [];
  selectedGroup: PAAGroup | null = null;
  movements: PAAMovement[] = [];
  portfolioSummary: PortfolioSummary | null = null;

  // Formulaires
  showInitForm = false;
  showPeriodForm = false;
  showStressForm = false;

  // Init form
  newGroupId = '';
  newPortfolio = 'AUTO';
  contractsJson = `[
  {
    "contract_id": "C1",
    "portfolio": "AUTO",
    "inception": "2025-01-01",
    "expiry": "2025-12-31",
    "written_premium": 1200,
    "expected_claim_ratio": 0.55,
    "expected_expense_ratio": 0.12
  }
]`;

  // Period form
  periodStart = '2025-01-01';
  periodEnd = '2025-01-31';
  incurredClaims = 0;
  claimsPaid = 0;

  // Stress test
  claimShock = 0;
  expenseShock = 0;
  stressResult: any = null;

  // Loading states
  loading = false;
  loadingGroups = false;
  loadingMovements = false;

  constructor(private apiService: IFRS17ApiService) {}

  ngOnInit(): void {
    this.loadData();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadData(): void {
    this.loadGroups();
    this.loadPortfolioSummary();
  }

  loadGroups(): void {
    this.loadingGroups = true;
    this.apiService.listPAAGroups()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (res) => {
          this.groups = res.groups || [];
          this.loadingGroups = false;
        },
        error: (err) => {
          console.error('Erreur chargement groupes:', err);
          this.loadingGroups = false;
        }
      });
  }

  loadPortfolioSummary(): void {
    this.apiService.getPAAPortfolioSummary()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (res) => {
          this.portfolioSummary = res.summary;
        },
        error: (err) => console.error('Erreur summary:', err)
      });
  }

  selectGroup(group: PAAGroup): void {
    this.selectedGroup = group;
    this.loadMovements(group.group_id);
  }

  loadMovements(groupId: string): void {
    this.loadingMovements = true;
    this.apiService.getPAAMovements(groupId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (res) => {
          this.movements = res.movements || [];
          this.loadingMovements = false;
        },
        error: (err) => {
          console.error('Erreur mouvements:', err);
          this.loadingMovements = false;
        }
      });
  }

  // Actions
  initGroup(): void {
    try {
      const contracts = JSON.parse(this.contractsJson);
      this.loading = true;

      this.apiService.initPAAGroup(this.newGroupId, contracts)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (res) => {
            console.log('Groupe initialisé:', res);
            this.showInitForm = false;
            this.loadData();
            this.loading = false;
          },
          error: (err) => {
            console.error('Erreur init:', err);
            alert('Erreur: ' + err.message);
            this.loading = false;
          }
        });
    } catch (e) {
      alert('JSON invalide');
    }
  }

  processPeriod(): void {
    if (!this.selectedGroup) {
      alert('Sélectionnez un groupe');
      return;
    }

    this.loading = true;
    this.apiService.processPAAPeriod(
      this.selectedGroup.group_id,
      this.periodStart,
      this.periodEnd,
      this.incurredClaims,
      this.claimsPaid
    )
    .pipe(takeUntil(this.destroy$))
    .subscribe({
      next: (res) => {
        console.log('Période traitée:', res);
        this.showPeriodForm = false;
        this.loadData();
        if (this.selectedGroup) {
          this.loadMovements(this.selectedGroup.group_id);
        }
        this.loading = false;
      },
      error: (err) => {
        console.error('Erreur période:', err);
        alert('Erreur: ' + err.message);
        this.loading = false;
      }
    });
  }

  runStressTest(): void {
    if (!this.selectedGroup) {
      alert('Sélectionnez un groupe');
      return;
    }

    this.loading = true;
    this.apiService.paaStressTest(
      this.selectedGroup.group_id,
      this.claimShock,
      this.expenseShock
    )
    .pipe(takeUntil(this.destroy$))
    .subscribe({
      next: (res) => {
        this.stressResult = res.results;
        this.loading = false;
      },
      error: (err) => {
        console.error('Erreur stress test:', err);
        alert('Erreur: ' + err.message);
        this.loading = false;
      }
    });
  }

  formatCurrency(value: number): string {
    return new Intl.NumberFormat('fr-FR', {
      style: 'currency',
      currency: 'EUR'
    }).format(value);
  }

  formatPercent(value: number): string {
    return new Intl.NumberFormat('fr-FR', {
      style: 'percent',
      minimumFractionDigits: 1,
      maximumFractionDigits: 1
    }).format(value);
  }
}
