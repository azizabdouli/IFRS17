// angular-frontend/src/app/components/dashboard/ppna-upload.component.ts

import { Component, OnInit, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PPNAService, PPNAMetrics } from '../../services/ppna.service';

@Component({
  selector: 'app-ppna-upload',
  templateUrl: './ppna-upload.component.html',
  standalone: true,
  imports: [CommonModule]
})
export class PPNAUploadComponent implements OnInit {
  @Input() ppnaMetrics: PPNAMetrics | null = null;

  selectedFile: File | null = null;
  isUploading = false;
  isDragOver = false;
  uploadProgress = 0;
  uploadMessage: { type: string; text: string } | null = null;

  constructor(private ppnaService: PPNAService) {}

  ngOnInit(): void {
    // S'abonner aux métriques PPNA
    this.ppnaService.metrics$.subscribe(metrics => {
      this.ppnaMetrics = metrics;
    });
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = false;

    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.handleFileSelection(files[0]);
    }
  }

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.handleFileSelection(file);
    }
  }

  private handleFileSelection(file: File): void {
    // Vérifier le type de fichier
    const allowedTypes = [
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
      'application/vnd.ms-excel' // .xls
    ];

    if (!allowedTypes.includes(file.type) && 
        !file.name.toLowerCase().endsWith('.xlsx') && 
        !file.name.toLowerCase().endsWith('.xls')) {
      this.showMessage('error', 'Veuillez sélectionner un fichier Excel (.xlsx ou .xls)');
      return;
    }

    // Vérifier la taille (max 50MB pour fichiers PPNA volumineux)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      this.showMessage('error', 'Le fichier est trop volumineux (max 50MB)');
      return;
    }

    this.selectedFile = file;
    this.uploadMessage = null;
    this.showMessage('info', `Fichier sélectionné: ${file.name}`);
  }

  uploadFile(): void {
    if (!this.selectedFile) {
      this.showMessage('warning', 'Veuillez sélectionner un fichier');
      return;
    }

    this.isUploading = true;
    this.uploadProgress = 0;
    this.showMessage('info', 'Traitement du fichier PPNA en cours...');

    // Simulation de progression
    const progressInterval = setInterval(() => {
      this.uploadProgress += 10;
      if (this.uploadProgress >= 90) {
        clearInterval(progressInterval);
      }
    }, 200);

    this.ppnaService.uploadFile(this.selectedFile).subscribe({
      next: (response) => {
        clearInterval(progressInterval);
        this.uploadProgress = 100;
        this.isUploading = false;

        if (response.status === 'success') {
          this.showMessage('success', 
            `Fichier traité avec succès! ${response.processing_result?.analysis?.total_rows || 0} lignes analysées.`);
          
          // Rafraîchir les métriques
          setTimeout(() => {
            this.ppnaService.refreshMetrics();
          }, 1000);
        } else {
          this.showMessage('error', response.message || 'Erreur lors du traitement');
        }
      },
      error: (error) => {
        clearInterval(progressInterval);
        this.uploadProgress = 0;
        this.isUploading = false;
        this.showMessage('error', `Erreur: ${error.message}`);
      }
    });
  }

  loadCurrentData(): void {
    this.showMessage('info', 'Rechargement des données PPNA...');
    
    this.ppnaService.loadInitialData().subscribe({
      next: (response) => {
        if (response.status === 'success') {
          this.showMessage('success', 'Données PPNA rechargées avec succès');
          this.ppnaService.refreshMetrics();
        } else {
          this.showMessage('error', response.message || 'Erreur lors du rechargement');
        }
      },
      error: (error) => {
        this.showMessage('error', `Erreur: ${error.message}`);
      }
    });
  }

  clearFile(): void {
    this.selectedFile = null;
    this.uploadMessage = null;
    this.uploadProgress = 0;
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  formatCurrency(amount: number): string {
    return this.ppnaService.formatCurrency(amount);
  }

  isPPNADataLoaded(): boolean {
    return this.ppnaMetrics !== null;
  }

  getApprocheBadgeClass(): string {
    return this.ppnaMetrics?.approche === 'PAA' ? 'badge bg-success' : 'badge bg-secondary';
  }

  getContratOnereuxSeverity(): string {
    const count = this.ppnaMetrics?.contrats_onereux || 0;
    if (count === 0) return 'badge bg-success';
    if (count < 10) return 'badge bg-warning';
    return 'badge bg-danger';
  }

  private showMessage(type: string, text: string): void {
    this.uploadMessage = { type, text };
    
    // Auto-hide après 5 secondes pour les messages success/info
    if (type === 'success' || type === 'info') {
      setTimeout(() => {
        if (this.uploadMessage?.type === type && this.uploadMessage?.text === text) {
          this.uploadMessage = null;
        }
      }, 5000);
    }
  }
}