// angular-frontend/src/app/interceptors/auth.interceptor.ts

import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const router = inject(Router);
  const token = localStorage.getItem('ifrs17_auth_token');
  
  // Cloner la requête et ajouter le token si disponible
  if (token) {
    req = req.clone({
      setHeaders: {
        Authorization: `Bearer ${token}`
      }
    });
  }
  
  return next(req).pipe(
    catchError((error) => {
      // Gérer les erreurs d'authentification
      if (error.status === 401 || error.status === 403) {
        // Token invalide ou expiré
        localStorage.removeItem('ifrs17_auth_token');
        localStorage.removeItem('ifrs17_user_data');
        router.navigate(['/auth/signin']);
      }
      
      return throwError(() => error);
    })
  );
};
