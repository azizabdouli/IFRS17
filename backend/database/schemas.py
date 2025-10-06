from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    ANALYSTE_IFRS17 = "analyste_ifrs17"
    ACTUAIRE = "actuaire"
    COMPTABLE = "comptable"

class UserLevel(str, Enum):
    DEBUTANT = "Débutant"
    INTERMEDIAIRE = "Intermédiaire" 
    EXPERT = "Expert"
    MAITRE = "Maître IFRS17"

# Schémas pour la création d'utilisateur
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    role: UserRole = UserRole.ANALYSTE_IFRS17
    company: str = "BNA"
    phone: Optional[str] = None
    department: Optional[str] = "Assurance"
    employee_id: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Le mot de passe doit contenir au moins 6 caractères')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProgress(BaseModel):
    level: UserLevel = UserLevel.INTERMEDIAIRE
    points: int = 0
    badges: List[str] = []
    daily_tasks_completed: int = 0
    weekly_goals_achieved: int = 0
    monthly_reports_generated: int = 0
    accuracy_streak: int = 0
    progress_percentage: float = 0.0

# Schémas pour la réponse
class UserResponse(BaseModel):
    id: int
    email: str
    first_name: str
    last_name: str
    full_name: str
    role: UserRole
    company: str
    department: str = "Assurance"
    level: UserLevel = UserLevel.INTERMEDIAIRE
    points: int = 0
    progress: Optional[UserProgress] = None
    created_at: datetime
    last_login: Optional[datetime]
    login_count: int
    phone: Optional[str]
    employee_id: Optional[str]
    is_active: bool = True
    is_verified: bool = True
    
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    company: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
    expires_in: int = 3600

class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 6:
            raise ValueError('Le nouveau mot de passe doit contenir au moins 6 caractères')
        return v

# === SCHEMAS DASHBOARD ===

class KPIMetrics(BaseModel):
    total_ppna: float = Field(default=0.0, description="Total PPNA en TND")
    csm_total: float = Field(default=0.0, description="CSM total en TND") 
    onerous_contracts_count: int = Field(default=0, description="Nombre de contrats onéreux")
    profitability_ratio: float = Field(default=0.0, description="Ratio de profitabilité en %")
    loss_component: float = Field(default=0.0, description="Composant de perte en TND")
    revenue_growth: float = Field(default=0.0, description="Croissance du chiffre d'affaires en %")
    risk_score: float = Field(default=0.0, description="Score de risque sur 5")

class Alert(BaseModel):
    id: int
    type: str = Field(..., description="Type d'alerte: critical, warning, info")
    title: str = Field(..., description="Titre de l'alerte")
    message: str = Field(..., description="Message détaillé")
    priority: str = Field(..., description="Priorité: high, medium, low")
    created_at: datetime
    actions: List[str] = Field(default=[], description="Actions possibles")
    is_read: bool = False

class DashboardResponse(BaseModel):
    kpis: KPIMetrics
    alerts: List[Alert] = []
    quick_actions: List[str] = []
    last_updated: datetime
    user_progress: UserProgress

# === SCHEMAS PPNA ===

class PPNAMetrics(BaseModel):
    total_ppna: float = 0.0
    total_dac: float = 0.0
    total_premium: float = 0.0
    onerous_contracts_count: int = 0
    profitability_ratio: float = 0.0
    contracts_by_cohorte: dict = {}
    monthly_projection: dict = {}

# === RESPONSES GÉNÉRIQUES ===

class MessageResponse(BaseModel):
    message: str
    success: bool = True
    data: Optional[dict] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    success: bool = False