import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ERPService, ERPSummary, ERPDataQuality } from '../../services/erp.service';

@Component({
  selector: 'app-erp-dashboard',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './erp-dashboard.component.html',
  styleUrls: ['./erp-dashboard.component.scss']
})
export class ErpDashboardComponent implements OnInit {
  summary: ERPSummary | null = null;
  dataQuality: ERPDataQuality | null = null;
  loading = true;
  error: string | null = null;

  constructor(private erpService: ERPService) {}

  ngOnInit(): void {
    this.loadDashboard();
  }

  private loadDashboard(): void {
    this.loading = true;
    this.erpService.getSummary().subscribe({
      next: (summary) => {
        this.summary = summary;
        this.loadDataQuality();
      },
      error: (err) => {
        this.error = err.message;
        this.loading = false;
      }
    });
  }

  private loadDataQuality(): void {
    this.erpService.getDataQuality().subscribe({
      next: (quality) => {
        this.dataQuality = quality;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.message;
        this.loading = false;
      }
    });
  }
}
