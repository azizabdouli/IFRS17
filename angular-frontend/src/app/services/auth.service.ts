import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject, Observable, throwError, of } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export enum UserLevel {
  DEBUTANT = 'Débutant',
  INTERMEDIAIRE = 'Intermédiaire', 
  EXPERT = 'Expert',
  MAITRE_IFRS17 = 'Maître IFRS17'
}

export interface UserProgress {
  level: UserLevel;
  points: number;
  badges: string[];
  daily_tasks_completed: number;
  weekly_goals_achieved: number;
  monthly_reports_generated: number;
  accuracy_streak: number;
  progress_percentage: number;
}

export interface User {
  id: number;
  email: string;
  first_name: string;
  last_name: string;
  full_name: string;
  role: 'analyste_ifrs17';
  company: string;
  department?: string;
  level: UserLevel;
  points: number;
  progress: UserProgress;
  created_at: string;
  last_login?: string;
  login_count: number;
  phone?: string;
  employee_id?: string;
  is_active: boolean;
  is_verified: boolean;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  first_name: string;
  last_name: string;
  role?: string;
  company: string;
  phone?: string;
  department?: string;
  employee_id?: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface AuthResponse {
  success: boolean;
  message: string;
  user?: User;
  token?: string;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private readonly API_URL = environment.apiUrl || 'http://localhost:8001';
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();
  
  private readonly STORAGE_KEY = 'ifrs17_auth_token';
  private readonly USER_KEY = 'ifrs17_user_data';

  constructor(private http: HttpClient) {
    // Charger l'utilisateur depuis le localStorage au démarrage
    this.loadUserFromStorage();
  }

  private loadUserFromStorage(): void {
    const token = localStorage.getItem(this.STORAGE_KEY);
    const userData = localStorage.getItem(this.USER_KEY);
    
    if (token && userData) {
      try {
        const user = JSON.parse(userData);
        this.currentUserSubject.next(user);
        // Vérifier la validité du token avec le backend
        this.verifyToken().subscribe({
          next: (isValid) => {
            if (!isValid) {
              this.logout();
            }
          },
          error: () => {
            this.logout();
          }
        });
      } catch (error) {
        console.error('Erreur lors du chargement des données utilisateur:', error);
        this.logout();
      }
    }
  }

  private handleError(error: any): Observable<never> {
    let errorMessage = 'Une erreur inattendue s\'est produite';
    
    if (error.error) {
      // Erreur côté serveur
      if (error.status === 401) {
        errorMessage = 'Email ou mot de passe incorrect';
      } else if (error.status === 400) {
        errorMessage = error.error?.detail || 'Données invalides';
      } else if (error.status === 409) {
        errorMessage = 'Un utilisateur avec cet email existe déjà';
      } else if (error.status === 500) {
        errorMessage = 'Erreur serveur. Veuillez réessayer plus tard.';
      } else {
        errorMessage = error.error?.detail || errorMessage;
      }
    }
    
    console.error('Erreur AuthService:', error);
    return throwError(() => new Error(errorMessage));
  }

  /**
   * Connexion utilisateur avec API backend
   */
  login(loginData: LoginRequest): Observable<AuthResponse> {
    return this.http.post<TokenResponse>(`${this.API_URL}/auth/login`, loginData)
      .pipe(
        map(response => {
          // Stocker le token et les données utilisateur
          localStorage.setItem(this.STORAGE_KEY, response.access_token);
          localStorage.setItem(this.USER_KEY, JSON.stringify(response.user));
          
          // Mettre à jour le BehaviorSubject
          this.currentUserSubject.next(response.user);
          
          return {
            success: true,
            message: 'Connexion réussie',
            user: response.user,
            token: response.access_token
          };
        }),
        catchError(this.handleError.bind(this))
      );
  }

  /**
   * Inscription utilisateur avec API backend
   */
  register(registerData: RegisterRequest): Observable<AuthResponse> {
    return this.http.post<User>(`${this.API_URL}/auth/register`, registerData)
      .pipe(
        map(user => {
          return {
            success: true,
            message: 'Inscription réussie. Vous pouvez maintenant vous connecter.',
            user: user
          };
        }),
        catchError(this.handleError.bind(this))
      );
  }

  /**
   * Déconnexion avec invalidation de session côté serveur
   */
  logout(): Observable<any> {
    const token = localStorage.getItem(this.STORAGE_KEY);
    
    // Nettoyer le stockage local immédiatement
    localStorage.removeItem(this.STORAGE_KEY);
    localStorage.removeItem(this.USER_KEY);
    this.currentUserSubject.next(null);
    
    // Notifier le serveur si on a un token
    if (token) {
      const headers = { Authorization: `Bearer ${token}` };
      return this.http.post(`${this.API_URL}/auth/logout`, {}, { headers })
        .pipe(
          catchError(() => {
            // Même si la déconnexion serveur échoue, on considère comme réussie côté client
            return [];
          })
        );
    }
    
    return new Observable(observer => {
      observer.next(true);
      observer.complete();
    });
  }

  /**
   * Vérifier la validité du token actuel
   */
  verifyToken(): Observable<boolean> {
    const token = localStorage.getItem(this.STORAGE_KEY);
    if (!token) {
      return of(false);
    }

    const headers = { Authorization: `Bearer ${token}` };
    return this.http.get<any>(`${this.API_URL}/auth/verify`, { headers })
      .pipe(
        map(() => true),
        catchError(() => of(false))
      );
  }

  /**
   * Récupérer les informations de l'utilisateur actuel depuis le serveur
   */
  getCurrentUserInfo(): Observable<User> {
    const token = localStorage.getItem(this.STORAGE_KEY);
    if (!token) {
      return throwError(() => new Error('Aucun token trouvé'));
    }

    const headers = { Authorization: `Bearer ${token}` };
    return this.http.get<User>(`${this.API_URL}/auth/me`, { headers })
      .pipe(
        tap(user => {
          // Mettre à jour les données locales
          localStorage.setItem(this.USER_KEY, JSON.stringify(user));
          this.currentUserSubject.next(user);
        }),
        catchError(this.handleError.bind(this))
      );
  }

  /**
   * Mettre à jour les informations de l'utilisateur
   */
  updateUserInfo(updateData: Partial<User>): Observable<User> {
    const token = localStorage.getItem(this.STORAGE_KEY);
    if (!token) {
      return throwError(() => new Error('Aucun token trouvé'));
    }

    const headers = { Authorization: `Bearer ${token}` };
    return this.http.put<User>(`${this.API_URL}/auth/me`, updateData, { headers })
      .pipe(
        tap(user => {
          // Mettre à jour les données locales
          localStorage.setItem(this.USER_KEY, JSON.stringify(user));
          this.currentUserSubject.next(user);
        }),
        catchError(this.handleError.bind(this))
      );
  }

  /**
   * Changer le mot de passe
   */
  changePassword(currentPassword: string, newPassword: string): Observable<any> {
    const token = localStorage.getItem(this.STORAGE_KEY);
    if (!token) {
      return throwError(() => new Error('Aucun token trouvé'));
    }

    const headers = { Authorization: `Bearer ${token}` };
    const data = {
      current_password: currentPassword,
      new_password: newPassword
    };

    return this.http.post(`${this.API_URL}/auth/change-password`, data, { headers })
      .pipe(
        catchError(this.handleError.bind(this))
      );
  }

  /**
   * Vérifier si l'utilisateur est connecté
   */
  isAuthenticated(): boolean {
    return !!localStorage.getItem(this.STORAGE_KEY);
  }

  /**
   * Obtenir l'utilisateur actuel
   */
  getCurrentUser(): User | null {
    return this.currentUserSubject.value;
  }

  /**
   * Obtenir le token d'authentification
   */
  getToken(): string | null {
    return localStorage.getItem(this.STORAGE_KEY);
  }

  /**
   * Vérifier si l'utilisateur est un analyste IFRS17
   */
  isAnalysteIFRS17(): boolean {
    const user = this.getCurrentUser();
    return user?.role === 'analyste_ifrs17';
  }

  /**
   * Obtenir le nom complet de l'utilisateur
   */
  getFullName(): string {
    const user = this.getCurrentUser();
    return user ? user.full_name || `${user.first_name} ${user.last_name}` : '';
  }

  /**
   * Obtenir le niveau et progression de l'utilisateur
   */
  getUserProgress(): UserProgress | null {
    const user = this.getCurrentUser();
    return user?.progress || null;
  }

  /**
   * Attribuer des points à l'utilisateur 
   */
  awardPoints(points: number, action?: string): Observable<any> {
    const url = `${this.API_URL}/dashboard/award-points/${points}`;
    const params: any = {};
    if (action) {
      params.action = action;
    }
    
    return this.http.post(url, {}, { params }).pipe(
      tap(() => {
        // Recharger les données utilisateur pour mettre à jour la progression
        this.refreshUserData();
      }),
      catchError(this.handleError)
    );
  }

  /**
   * Recharger les données utilisateur depuis le serveur
   */
  refreshUserData(): void {
    if (this.isAuthenticated()) {
      this.http.get<User>(`${this.API_URL}/auth/me`).subscribe({
        next: (user) => {
          localStorage.setItem(this.USER_KEY, JSON.stringify(user));
          this.currentUserSubject.next(user);
        },
        error: (error) => {
          console.error('Erreur lors du rechargement des données utilisateur:', error);
        }
      });
    }
  }
}