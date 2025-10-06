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

          <!-- 🧭 Navigation -->
          <div class="col-md-6">
            <ul class="main-nav">
              <li>
                <a routerLink="/dashboard" routerLinkActive="active">
                  <i class="fas fa-chart-line me-2"></i>
                  Dashboard
                </a>
              </li>
              <li>
                <a routerLink="/ppna-analytics" routerLinkActive="active">
                  <i class="fas fa-calculator me-2"></i>
                  PPNA
                </a>
              </li>
              <li>
                <a routerLink="/ml-analytics-new" routerLinkActive="active">
                  <i class="fas fa-brain me-2"></i>
                  ML Analytics
                </a>
              </li>
              <li>
                <a routerLink="/ai-assistant" routerLinkActive="active">
                  <i class="fas fa-robot me-2"></i>
                  Assistant IA
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
      gap: 1rem;
      justify-content: center;

      li a {
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