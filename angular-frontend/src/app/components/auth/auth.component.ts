import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators, AbstractControl } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService, LoginRequest, RegisterRequest } from '../../services/auth.service';

@Component({
  selector: 'app-auth',
  templateUrl: './auth.component.html',
  styleUrls: ['./auth.component.scss'],
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule]
})
export class AuthComponent implements OnInit {
  isSignUp = false;
  showPassword = false;
  isLoading = false;
  message = '';
  messageType: 'success' | 'error' = 'error';

  signinForm!: FormGroup;
  signupForm!: FormGroup;

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.initializeForms();
  }

  private initializeForms(): void {
    // Formulaire de connexion
    this.signinForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]]
    });

    // Formulaire d'inscription - Simplifié pour Analyste IFRS17
    this.signupForm = this.fb.group({
      firstName: ['', [Validators.required, Validators.minLength(2)]],
      lastName: ['', [Validators.required, Validators.minLength(2)]],
      email: ['', [Validators.required, Validators.email]],
      role: ['analyste_ifrs17'], // Valeur par défaut, pas de validation requise
      company: ['BNA'], // Par défaut BNA, pas de validation requise
      phone: [''],
      department: ['Assurance'], // Département d'assurance par défaut, pas de validation requise
      employeeId: [''],
      password: ['', [Validators.required, Validators.minLength(6)]], // Réduit à 6 caractères
      confirmPassword: ['', [Validators.required]],
      acceptTerms: [false, [Validators.requiredTrue]]
    }, { validators: this.passwordMatchValidator });
  }

  // Validateur personnalisé pour vérifier que les mots de passe correspondent
  private passwordMatchValidator(control: AbstractControl): {[key: string]: boolean} | null {
    const password = control.get('password');
    const confirmPassword = control.get('confirmPassword');
    
    if (!password || !confirmPassword) {
      return null;
    }
    
    if (password.value !== confirmPassword.value) {
      confirmPassword.setErrors({ passwordMismatch: true });
      return { passwordMismatch: true };
    } else {
      // Nettoyer l'erreur si les mots de passe correspondent
      if (confirmPassword.errors?.['passwordMismatch']) {
        delete confirmPassword.errors['passwordMismatch'];
        if (Object.keys(confirmPassword.errors).length === 0) {
          confirmPassword.setErrors(null);
        }
      }
      return null;
    }
  }

  switchToSignIn(): void {
    this.isSignUp = false;
    this.clearMessages();
    this.signinForm.reset();
  }

  switchToSignUp(): void {
    this.isSignUp = true;
    this.clearMessages();
    this.signupForm.reset();
  }

  togglePassword(): void {
    this.showPassword = !this.showPassword;
  }

  onSignIn(): void {
    if (this.signinForm.valid && !this.isLoading) {
      this.isLoading = true;
      this.clearMessages();

      const loginData: LoginRequest = {
        email: this.signinForm.value.email,
        password: this.signinForm.value.password
      };

      this.authService.login(loginData).subscribe({
        next: (response) => {
          this.isLoading = false;
          if (response.success) {
            this.showMessage('Connexion réussie ! Redirection...', 'success');
            setTimeout(() => {
              this.router.navigate(['/dashboard']);
            }, 1000);
          } else {
            this.showMessage(response.message, 'error');
          }
        },
        error: (error) => {
          this.isLoading = false;
          this.showMessage(error.message || 'Erreur de connexion. Veuillez réessayer.', 'error');
          console.error('Erreur login:', error);
        }
      });
    } else {
      this.markFormGroupTouched(this.signinForm);
    }
  }

  onSignUp(): void {
    console.log('📝 Tentative d\'inscription...');
    console.log('Formulaire valide:', this.signupForm.valid);
    console.log('Données formulaire:', this.signupForm.value);
    console.log('Erreurs formulaire:', this.signupForm.errors);
    
    if (this.signupForm.valid && !this.isLoading) {
      this.isLoading = true;
      this.clearMessages();

      const registerData: RegisterRequest = {
        email: this.signupForm.value.email,
        password: this.signupForm.value.password,
        first_name: this.signupForm.value.firstName,
        last_name: this.signupForm.value.lastName,
        role: this.signupForm.value.role,
        company: this.signupForm.value.company,
        phone: this.signupForm.value.phone,
        department: this.signupForm.value.department,
        employee_id: this.signupForm.value.employeeId
      };

      console.log('📤 Envoi des données d\'inscription:', registerData);

      this.authService.register(registerData).subscribe({
        next: (response) => {
          this.isLoading = false;
          console.log('✅ Réponse inscription:', response);
          if (response.success) {
            this.showMessage('✅ Compte créé avec succès ! Redirection vers la connexion...', 'success');
            // Nettoyer le formulaire
            this.signupForm.reset();
            // Retour au mode connexion après un délai
            setTimeout(() => {
              this.isSignUp = false;
              this.clearMessages();
              this.showMessage('Vous pouvez maintenant vous connecter avec vos identifiants.', 'success');
            }, 2000);
          } else {
            this.showMessage(response.message, 'error');
          }
        },
        error: (error) => {
          this.isLoading = false;
          console.error('❌ Erreur inscription:', error);
          this.showMessage(error.message || 'Erreur lors de la création du compte. Veuillez réessayer.', 'error');
        }
      });
    } else {
      console.log('❌ Formulaire invalide, marquage des champs touchés');
      this.markFormGroupTouched(this.signupForm);
      
      // Affichage détaillé des erreurs
      Object.keys(this.signupForm.controls).forEach(key => {
        const control = this.signupForm.get(key);
        if (control?.invalid) {
          console.log(`Champ ${key} invalide:`, control.errors);
        }
      });
    }
  }

  private showMessage(message: string, type: 'success' | 'error'): void {
    this.message = message;
    this.messageType = type;
    
    // Auto-clear message après 8 secondes seulement pour les erreurs
    if (type === 'error') {
      setTimeout(() => {
        this.clearMessages();
      }, 8000);
    }
  }

  private clearMessages(): void {
    this.message = '';
  }

  markFormGroupTouched(formGroup: FormGroup): void {
    Object.keys(formGroup.controls).forEach(key => {
      const control = formGroup.get(key);
      control?.markAsTouched();
    });
  }
}