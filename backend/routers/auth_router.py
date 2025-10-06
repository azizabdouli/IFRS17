from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.database.schemas import UserCreate, UserLogin, UserResponse, TokenResponse, UserUpdate, PasswordChange
from backend.auth.user_service import UserService
from backend.auth.security import create_user_token, extract_user_from_token
from typing import Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration du router
router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()

# Dependency pour récupérer l'utilisateur actuel
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), 
                          db: Session = Depends(get_db)) -> UserResponse:
    """Récupérer l'utilisateur actuel à partir du token JWT"""
    try:
        # Extraire le token
        token = credentials.credentials
        user_data = extract_user_from_token(token)
        
        if user_data is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token invalide ou expiré",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Récupérer l'utilisateur depuis la base de données
        user = UserService.get_user_by_id(db, user_data["user_id"])
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Utilisateur introuvable",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return UserService.to_user_response(user)
        
    except Exception as e:
        logger.error(f"Erreur d'authentification: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentification échouée",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, request: Request, db: Session = Depends(get_db)):
    """Inscription d'un nouvel utilisateur Analyste IFRS17"""
    try:
        # Créer l'utilisateur
        user = UserService.create_user(db, user_data)
        
        logger.info(f"Nouvel utilisateur Analyste IFRS17 enregistré: {user.email}")
        return UserService.to_user_response(user)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erreur lors de l'inscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur"
        )

@router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLogin, request: Request, db: Session = Depends(get_db)):
    """Connexion d'un utilisateur Analyste IFRS17"""
    try:
        # Authentifier l'utilisateur
        user = UserService.authenticate_user(db, login_data.email, login_data.password)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Email ou mot de passe incorrect"
            )
        
        # Créer le token JWT
        access_token = create_user_token(user.id, user.email, user.role)
        
        logger.info(f"Connexion réussie: {user.email}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserService.to_user_response(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la connexion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur"
        )

@router.post("/logout")
async def logout(request: Request, current_user: UserResponse = Depends(get_current_user), 
                credentials: HTTPAuthorizationCredentials = Depends(security),
                db: Session = Depends(get_db)):
    """Déconnexion d'un utilisateur"""
    try:
        logger.info(f"Déconnexion: {current_user.email}")
        return {"message": "Déconnexion réussie"}
        
    except Exception as e:
        logger.error(f"Erreur lors de la déconnexion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserResponse = Depends(get_current_user)):
    """Récupérer les informations de l'utilisateur actuel"""
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_current_user(user_update: UserUpdate, request: Request,
                             current_user: UserResponse = Depends(get_current_user),
                             db: Session = Depends(get_db)):
    """Mettre à jour les informations de l'utilisateur actuel"""
    try:
        updated_user = UserService.update_user(db, current_user.id, user_update)
        
        if updated_user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Utilisateur introuvable"
            )
        
        logger.info(f"Utilisateur mis à jour: {current_user.email}")
        return UserService.to_user_response(updated_user)
        
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur"
        )

@router.post("/change-password")
async def change_password(password_data: PasswordChange, request: Request,
                         current_user: UserResponse = Depends(get_current_user),
                         db: Session = Depends(get_db)):
    """Changer le mot de passe de l'utilisateur actuel"""
    try:
        success = UserService.change_password(
            db, current_user.id, 
            password_data.current_password, 
            password_data.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mot de passe actuel incorrect"
            )
        
        logger.info(f"Mot de passe modifié: {current_user.email}")
        return {"message": "Mot de passe modifié avec succès"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du changement de mot de passe: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur"
        )

@router.get("/verify")
async def verify_token(current_user: UserResponse = Depends(get_current_user)):
    """Vérifier la validité du token"""
    return {"valid": True, "user": current_user}