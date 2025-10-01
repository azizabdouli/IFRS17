// src/app/app.module.ts

import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

// Routing Module
import { AppRoutingModule } from './app-routing.module';

// PrimeNG Modules
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { ChartModule } from 'primeng/chart';
import { TableModule } from 'primeng/table';
import { InputTextModule } from 'primeng/inputtext';
import { DropdownModule } from 'primeng/dropdown';
import { CalendarModule } from 'primeng/calendar';
import { DialogModule } from 'primeng/dialog';
import { ToastModule } from 'primeng/toast';
import { ProgressBarModule } from 'primeng/progressbar';
import { TabViewModule } from 'primeng/tabview';
import { PanelModule } from 'primeng/panel';
import { ToolbarModule } from 'primeng/toolbar';
import { MenubarModule } from 'primeng/menubar';
import { SidebarModule } from 'primeng/sidebar';

// Application Components
import { AppComponent } from './app.component';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { MLAnalyticsComponent } from './components/ml-analytics/ml-analytics.component';
import { AIAssistantComponent } from './components/ai-assistant/ai-assistant.component';
import { DataTransformationsComponent } from './components/data-transformations/data-transformations.component';
import { PPNAUploadComponent } from './components/dashboard/ppna-upload.component';

// Services
import { IFRS17ApiService } from './services/ifrs17-api.service';
import { PPNAService } from './services/ppna.service';
import { MessageService } from 'primeng/api';

@NgModule({
  declarations: [
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    
    // PrimeNG Modules
    ButtonModule,
    CardModule,
    ChartModule,
    TableModule,
    InputTextModule,
    DropdownModule,
    CalendarModule,
    DialogModule,
    ToastModule,
    ProgressBarModule,
    TabViewModule,
    PanelModule,
    ToolbarModule,
    MenubarModule,
    SidebarModule
  ],
  providers: [
    IFRS17ApiService,
    PPNAService,
    MessageService
  ],
  bootstrap: []
})
export class AppModule { }