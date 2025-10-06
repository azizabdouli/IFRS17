// src/app/app-routing.module.ts

import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { MLAnalyticsNewComponent } from './components/ml-analytics/ml-analytics-new.component';
import { PPNAAnalyticsComponent } from './components/ppna-analytics/ppna-analytics.component';
import { AIAssistantComponent } from './components/ai-assistant/ai-assistant.component';
import { DataTransformationsComponent } from './components/data-transformations/data-transformations.component';
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
  { 
    path: 'ppna-analytics', 
    component: PPNAAnalyticsComponent,
    canActivate: [AuthGuard],
    data: { title: 'Analytics PPNA IFRS17' }
  },
  { 
    path: 'ml-analytics-complete', 
    component: MLAnalyticsNewComponent,
    canActivate: [AuthGuard],
    data: { title: 'Analytics ML Complet' }
  },
  { 
    path: 'ml-analytics-new', 
    component: MLAnalyticsNewComponent,
    canActivate: [AuthGuard],
    data: { title: 'Analytics ML Nouveau' }
  },
  { 
    path: 'ai-assistant', 
    component: AIAssistantComponent,
    canActivate: [AuthGuard],
    data: { title: 'Assistant IA' }
  },
  { 
    path: 'data-transformations', 
    component: DataTransformationsComponent,
    canActivate: [AuthGuard],
    data: { title: 'Transformations de Données' }
  },
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