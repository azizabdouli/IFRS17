// src/app/app-routing.module.ts

import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { MLAnalyticsNewComponent } from './components/ml-analytics/ml-analytics-new.component';
import { PPNAAnalyticsComponent } from './components/ppna-analytics/ppna-analytics.component';
import { AIAssistantComponent } from './components/ai-assistant/ai-assistant.component';
import { DataTransformationsComponent } from './components/data-transformations/data-transformations.component';
import { PaaDashboardComponent } from './components/paa-dashboard/paa-dashboard.component';
import { AuthComponent } from './components/auth/auth.component';
import { AuthGuard } from './guards/auth.guard';

export const routes: Routes = [
  // Routes d'authentification (non protégées)
  { 
    path: 'auth', 
    children: [
      { path: 'signin', component: AuthComponent },
      { path: 'signup', component: AuthComponent },
      { path: '', redirectTo: 'signin', pathMatch: 'full' }
    ]
  },
  
  // Routes protégées (nécessitent authentification)
  { 
    path: '', 
    redirectTo: '/dashboard', 
    pathMatch: 'full' 
  },
  { 
    path: 'dashboard', 
    component: DashboardComponent,
    canActivate: [AuthGuard],
    data: { title: 'Tableau de Bord IFRS17' }
  },
  // 📊 Analytics IFRS17
  { 
    path: 'analytics', 
    canActivate: [AuthGuard],
    children: [
      { 
        path: 'ppna', 
        component: PPNAAnalyticsComponent,
        data: { title: 'Analytics PPNA', icon: 'calculator' }
      },
      { 
        path: 'paa', 
        component: PaaDashboardComponent,
        data: { title: 'Dashboard PAA', icon: 'chart-pie' }
      },
      { 
        path: 'ml', 
        component: MLAnalyticsNewComponent,
        data: { title: 'Machine Learning', icon: 'brain' }
      },
      { 
        path: '', 
        redirectTo: 'ppna', 
        pathMatch: 'full' 
      }
    ]
  },
  
  // 🤖 Outils IA
  { 
    path: 'ai-assistant', 
    component: AIAssistantComponent,
    canActivate: [AuthGuard],
    data: { title: 'Assistant IA', icon: 'robot' }
  },
  
  // 🔧 Utilitaires
  { 
    path: 'data-transformations', 
    component: DataTransformationsComponent,
    canActivate: [AuthGuard],
    data: { title: 'Transformations', icon: 'exchange-alt' }
  },
  
  // Redirections pour compatibilité (anciennes URLs)
  { path: 'ppna-analytics', redirectTo: '/analytics/ppna', pathMatch: 'full' },
  { path: 'paa-dashboard', redirectTo: '/analytics/paa', pathMatch: 'full' },
  { path: 'ml-analytics-new', redirectTo: '/analytics/ml', pathMatch: 'full' },
  { path: 'ml-analytics-complete', redirectTo: '/analytics/ml', pathMatch: 'full' },
  {
    path: '**', 
    redirectTo: '/auth/signin'
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes, {
    enableTracing: false, // true pour debug
    useHash: false // URLs propres sans #
  })],
  exports: [RouterModule]
})
export class AppRoutingModule { }