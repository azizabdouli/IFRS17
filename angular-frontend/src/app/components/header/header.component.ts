import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { AuthService, User } from '../../services/auth.service';

@Component({
  selector: 'app-header',
  template: `
    <!-- 🏠 HEADER AVEC AUTHENTIFICATION -->
    <nav class="main-header" *ngIf="currentUser">
      <div class="container-fluid">
        <div class="row align-items-center">
          
          <!-- 🏢 Logo et Titre -->
          <div class="col-md-3">
            <div class="header-brand" routerLink="/dashboard">
              <div class="brand-icon">
                <i class="fas fa-shield-alt"></i>
              </div>
              <span class="brand-text">IFRS17 Hub</span>
            </div>
          </div>

          <!-- 🧭 Navigation Optimisée -->
          <div class="col-md-6">
            <ul class="main-nav">
              <!-- Dashboard Principal -->
              <li>
                <a routerLink="/dashboard" routerLinkActive="active" [routerLinkActiveOptions]="{exact: true}">
                  <i class="fas fa-home me-2"></i>
                  Accueil
                </a>
              </li>
              
              <!-- Analytics (Menu Déroulant) -->
              <li class="dropdown" (mouseenter)="showAnalyticsMenu = true" (mouseleave)="showAnalyticsMenu = false">
                <a routerLink="/analytics" routerLinkActive="active" class="nav-with-dropdown">
                  <i class="fas fa-chart-bar me-2"></i>
                  Analytics
                  <i class="fas fa-chevron-down ms-1" style="font-size: 0.7rem;"></i>
                </a>
                <div class="dropdown-menu-nav" [class.show]="showAnalyticsMenu">
                  <a routerLink="/analytics/ppna" class="dropdown-item-nav">
                    <i class="fas fa-calculator"></i>
                    <div>
                      <strong>PPNA</strong>
                      <small>Provisions et analyses</small>
                    </div>
                  </a>
                  <a routerLink="/analytics/paa" class="dropdown-item-nav">
                    <i class="fas fa-chart-pie"></i>
                    <div>
                      <strong>PAA Dashboard</strong>
                      <small>Premium Allocation Approach</small>
                    </div>
                  </a>
                  <a routerLink="/analytics/ml" class="dropdown-item-nav">
                    <i class="fas fa-brain"></i>
                    <div>
                      <strong>Machine Learning</strong>
                      <small>Prédictions & modèles</small>
                    </div>
                  </a>
                </div>
              </li>
              
              <!-- ERP Assurance -->
              <li class="dropdown" (mouseenter)="showErpMenu = true" (mouseleave)="showErpMenu = false">
                <a routerLink="/erp/dashboard" routerLinkActive="active" class="nav-with-dropdown">
                  <i class="fas fa-building me-2"></i>
                  ERP Assurance
                  <i class="fas fa-chevron-down ms-1" style="font-size: 0.7rem;"></i>
                </a>
                <div class="dropdown-menu-nav" [class.show]="showErpMenu">
                  <a routerLink="/erp/dashboard" class="dropdown-item-nav">
                    <i class="fas fa-layer-group"></i>
                    <div>
                      <strong>Vue d'ensemble</strong>
                      <small>Indicateurs & qualité</small>
                    </div>
                  </a>
                  <a routerLink="/erp/clients" class="dropdown-item-nav">
                    <i class="fas fa-user-friends"></i>
                    <div>
                      <strong>Clients</strong>
                      <small>Fiches client</small>
                    </div>
                  </a>
                  <a routerLink="/erp/policies" class="dropdown-item-nav">
                    <i class="fas fa-file-contract"></i>
                    <div>
                      <strong>Polices</strong>
                      <small>Contrats & IFRS17</small>
                    </div>
                  </a>
                  <a routerLink="/erp/coverages" class="dropdown-item-nav">
                    <i class="fas fa-shield-alt"></i>
                    <div>
                      <strong>Garanties</strong>
                      <small>Couvertures & franchises</small>
                    </div>
                  </a>
                  <a routerLink="/erp/claims" class="dropdown-item-nav">
                    <i class="fas fa-ambulance"></i>
                    <div>
                      <strong>Sinistres</strong>
                      <small>Suivi & règlements</small>
                    </div>
                  </a>
                  <a routerLink="/erp/invoices" class="dropdown-item-nav">
                    <i class="fas fa-receipt"></i>
                    <div>
                      <strong>Quittances</strong>
                      <small>Facturation & encaissements</small>
                    </div>
                  </a>
                  <a routerLink="/erp/ledger-entries" class="dropdown-item-nav">
                    <i class="fas fa-book"></i>
                    <div>
                      <strong>Écritures</strong>
                      <small>Comptabilité IFRS17</small>
                    </div>
                  </a>
                </div>
              </li>

              <!-- Assistant IA -->
              <li>
                <a routerLink="/ai-assistant" routerLinkActive="active">
                  <i class="fas fa-robot me-2"></i>
                  Assistant IA
                </a>
              </li>
              
              <!-- Transformations -->
              <li>
                <a routerLink="/data-transformations" routerLinkActive="active">
                  <i class="fas fa-exchange-alt me-2"></i>
                  Outils
                </a>
              </li>
            </ul>
          </div>

          <!-- 👤 Profil Utilisateur -->
          <div class="col-md-3">
            <div class="user-menu">
              <div class="user-info" (click)="toggleUserMenu()">
                <div class="user-avatar">
                  <i class="fas fa-user"></i>
                </div>
                <div class="user-details">
                  <span class="user-name">{{currentUser.first_name}} {{currentUser.last_name}}</span>
                  <span class="user-role">{{getRoleLabel(currentUser.role)}}</span>
                </div>
                <i class="fas fa-chevron-down"></i>
              </div>
              
              <div class="user-dropdown" [class.show]="showUserMenu">
                <div class="dropdown-header">
                  <strong>{{currentUser.first_name}} {{currentUser.last_name}}</strong>
                  <small>{{currentUser.email}}</small>
                </div>
                <div class="dropdown-item">
                  <i class="fas fa-user me-2"></i>
                  Mon Profil
                </div>
                <div class="dropdown-item">
                  <i class="fas fa-cog me-2"></i>
                  Paramètres
                </div>
                <div class="dropdown-divider"></div>
                <div class="dropdown-item logout" (click)="logout()">
                  <i class="fas fa-sign-out-alt me-2"></i>
                  Déconnexion
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </nav>
  `,
  styles: [`
    .main-header {
      background: white;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
      padding: 1rem 0;
      position: sticky;
      top: 0;
      z-index: 1000;
    }

    .header-brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      cursor: pointer;
      text-decoration: none;
      color: var(--ifrs17-text-primary);

      .brand-icon {
        width: 40px;
        height: 40px;
        background: var(--ifrs17-gradient-primary);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.1rem;
      }

      .brand-text {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--ifrs17-text-primary);
      }
    }

    .main-nav {
      display: flex;
      list-style: none;
      margin: 0;
      padding: 0;
      gap: 0.5rem;
      justify-content: center;

      li {
        position: relative;

        a {
          display: flex;
          align-items: center;
          padding: 0.75rem 1rem;
          border-radius: 10px;
          text-decoration: none;
          color: var(--ifrs17-text-secondary);
          font-weight: 500;
          transition: all 0.3s ease;

          &:hover, &.active {
            background: var(--ifrs17-primary);
            color: white;
          }

          i {
            font-size: 0.9rem;
          }
        }

        &.dropdown {
          .nav-with-dropdown {
            cursor: pointer;
          }

          .dropdown-menu-nav {
            position: absolute;
            top: 100%;
            left: 0;
            margin-top: 0.5rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            min-width: 280px;
            opacity: 0;
            visibility: hidden;
            transform: translateY(-10px);
            transition: all 0.3s ease;
            z-index: 1000;

            &.show {
              opacity: 1;
              visibility: visible;
              transform: translateY(0);
            }

            .dropdown-item-nav {
              display: flex;
              align-items: center;
              gap: 1rem;
              padding: 1rem;
              text-decoration: none;
              color: var(--ifrs17-text-primary);
              transition: background 0.3s ease;
              border-bottom: 1px solid var(--ifrs17-border);

              &:first-child {
                border-radius: 10px 10px 0 0;
              }

              &:last-child {
                border-bottom: none;
                border-radius: 0 0 10px 10px;
              }

              &:hover {
                background: var(--ifrs17-bg-secondary);
              }

              i {
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, var(--ifrs17-primary) 0%, var(--ifrs17-secondary) 100%);
                color: white;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.1rem;
              }

              div {
                display: flex;
                flex-direction: column;

                strong {
                  font-weight: 600;
                  color: var(--ifrs17-text-primary);
                  margin-bottom: 0.25rem;
                }

                small {
                  font-size: 0.8rem;
                  color: var(--ifrs17-text-secondary);
                }
              }
            }
          }
        }
      }
    }

    .user-menu {
      position: relative;

      .user-info {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 10px;
        transition: background 0.3s ease;

        &:hover {
          background: var(--ifrs17-bg-secondary);
        }

        .user-avatar {
          width: 40px;
          height: 40px;
          background: var(--ifrs17-gradient-primary);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }

        .user-details {
          display: flex;
          flex-direction: column;

          .user-name {
            font-weight: 600;
            color: var(--ifrs17-text-primary);
            font-size: 0.9rem;
          }

          .user-role {
            font-size: 0.8rem;
            color: var(--ifrs17-text-secondary);
          }
        }
      }

      .user-dropdown {
        position: absolute;
        top: 100%;
        right: 0;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        min-width: 220px;
        opacity: 0;
        visibility: hidden;
        transform: translateY(-10px);
        transition: all 0.3s ease;
        z-index: 1001;

        &.show {
          opacity: 1;
          visibility: visible;
          transform: translateY(0);
        }

        .dropdown-header {
          padding: 1rem;
          border-bottom: 1px solid var(--ifrs17-border);

          strong {
            display: block;
            color: var(--ifrs17-text-primary);
          }

          small {
            color: var(--ifrs17-text-secondary);
          }
        }

        .dropdown-item {
          padding: 0.75rem 1rem;
          cursor: pointer;
          transition: background 0.3s ease;
          color: var(--ifrs17-text-secondary);

          &:hover {
            background: var(--ifrs17-bg-secondary);
          }

          &.logout {
            color: var(--ifrs17-danger);
            border-top: 1px solid var(--ifrs17-border);
          }

          i {
            font-size: 0.85rem;
          }
        }

        .dropdown-divider {
          height: 1px;
          background: var(--ifrs17-border);
          margin: 0.5rem 0;
        }
      }
    }

    @media (max-width: 768px) {
      .main-nav {
        display: none;
      }
      
      .user-details {
        display: none;
      }
    }
  `],
  standalone: true,
  imports: [CommonModule, RouterModule]
})
export class HeaderComponent implements OnInit {
  currentUser: User | null = null;
  showUserMenu = false;
  showAnalyticsMenu = false;
  showErpMenu = false;

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.authService.currentUser$.subscribe(user => {
      this.currentUser = user;
    });

    // Fermer le menu utilisateur si on clique ailleurs
    document.addEventListener('click', (event) => {
      const target = event.target as HTMLElement;
      if (!target.closest('.user-menu')) {
        this.showUserMenu = false;
      }
    });
  }

  toggleUserMenu(): void {
    this.showUserMenu = !this.showUserMenu;
  }

  getRoleLabel(role: string): string {
    return role === 'actuaire' ? '👨‍💼 Actuaire' : '📊 Comptable';
  }

  logout(): void {
    this.authService.logout();
    this.router.navigate(['/auth/signin']);
  }
}
