// src/app/app-routing.module.ts

import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { MLAnalyticsNewComponent } from './components/ml-analytics/ml-analytics-new.component';
import { PPNAAnalyticsComponent } from './components/ppna-analytics/ppna-analytics.component';
import { AIAssistantComponent } from './components/ai-assistant/ai-assistant.component';
import { DataTransformationsComponent } from './components/data-transformations/data-transformations.component';

export const routes: Routes = [
  { 
    path: '', 
    redirectTo: '/dashboard', 
    pathMatch: 'full' 
  },
  { 
    path: 'dashboard', 
    component: DashboardComponent,
    data: { title: 'Tableau de Bord IFRS17' }
  },
  { 
    path: 'ppna-analytics', 
    component: PPNAAnalyticsComponent,
    data: { title: 'Analytics PPNA IFRS17' }
  },
  { 
    path: 'ml-analytics-complete', 
    component: MLAnalyticsNewComponent,
    data: { title: 'Analytics ML Complet' }
  },
  { 
    path: 'ai-assistant', 
    component: AIAssistantComponent,
    data: { title: 'Assistant IA' }
  },
  { 
    path: 'data-transformations', 
    component: DataTransformationsComponent,
    data: { title: 'Transformations de Données' }
  },
  {
    path: '**', 
    redirectTo: '/dashboard'
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