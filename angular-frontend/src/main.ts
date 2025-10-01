// src/main.ts

import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { importProvidersFrom } from '@angular/core';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

// Routes
import { routes } from './app/app-routing.module';

// Services
import { IFRS17ApiService } from './app/services/ifrs17-api.service';
import { PPNAService } from './app/services/ppna.service';

// 🚀 Bootstrap de l'application Angular IFRS17 standalone
bootstrapApplication(AppComponent, {
  providers: [
    provideRouter(routes),
    provideHttpClient(),
    importProvidersFrom(
      BrowserAnimationsModule,
      FormsModule,
      ReactiveFormsModule
    ),
    IFRS17ApiService,
    PPNAService
  ]
}).catch(err => {
  console.error('❌ Erreur lors du démarrage de l\'application IFRS17:', err);
});

// 🎯 Configuration de performance pour la comptabilité d'assurance
if ('serviceWorker' in navigator && typeof navigator.serviceWorker !== 'undefined') {
  window.addEventListener('load', () => {
    console.log('📱 Application IFRS17 prête pour utilisation hors ligne');
  });
}

// 🔧 Configuration développement
if (typeof window !== 'undefined') {
  (window as any).global = window;
}