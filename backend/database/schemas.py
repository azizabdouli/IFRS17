from pydantic import BaseModel, EmailStr, validator, Field, model_validator
from typing import Optional, List
from datetime import datetime, date
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

# === ERP Assurance - Enums ===

class ClientType(str, Enum):
    PARTICULIER = "particulier"
    ENTREPRISE = "entreprise"
    INTERMEDIAIRE = "intermediaire"

class ClientStatus(str, Enum):
    ACTIF = "actif"
    INACTIF = "inactif"

class PolicyStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    SUSPENDED = "suspended"

class MeasurementModel(str, Enum):
    PAA = "PAA"
    GMM = "GMM"
    VFA = "VFA"

class CoverageStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class ClaimStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    REJECTED = "rejected"

class InvoiceStatus(str, Enum):
    PENDING = "pending"
    PAID = "paid"
    OVERDUE = "overdue"

class LedgerEntryType(str, Enum):
    PREMIUM = "premium"
    CLAIM = "claim"
    COMMISSION = "commission"
    ADJUSTMENT = "adjustment"

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
    # 🔥 NOUVEAUX CHAMPS
    compliance_score: float = Field(default=0.0, description="Score de conformité IFRS17 en %")
    accuracy_rate: float = Field(default=0.0, description="Taux de précision ML en %")

class Alert(BaseModel):
    id: str
    type: str = Field(..., description="Type d'alerte: info, warning, error, success")
    title: str = Field(..., description="Titre de l'alerte")
    message: str = Field(..., description="Message détaillé")
    priority: str = Field(..., description="Priorité: low, medium, high, critical")
    created_at: str
    action_url: Optional[str] = None
    action_text: Optional[str] = None

class RecommendedAction(BaseModel):
    id: str
    title: str
    description: str
    category: str
    priority: str = Field(..., description="Priority: low, medium, high")
    estimated_time: int = Field(..., description="Estimated time in minutes")
    points_reward: int = Field(default=0, description="Points reward")
    action_url: str
    icon: str = Field(default="fas fa-tasks")

class WeeklySummary(BaseModel):
    tasks_completed: int = 0
    points_earned: int = 0
    badges_earned: List[str] = []
    accuracy_avg: float = 0.0

class Achievements(BaseModel):
    recent_badges: List[str] = []
    next_level_progress: float = 0.0
    total_achievements: int = 0

class DashboardResponse(BaseModel):
    user_id: int
    kpis: KPIMetrics
    alerts: List[Alert] = []
    recommended_actions: List[RecommendedAction] = []
    weekly_summary: WeeklySummary
    achievements: Achievements
    contextual_insights: List[str] = []

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

# === ERP Assurance - Schémas ===

class PortfolioCreate(BaseModel):
    name: str
    description: Optional[str] = None
    currency: str = "TND"
    manager: Optional[str] = None

class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    currency: Optional[str] = None
    manager: Optional[str] = None

class PortfolioResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    currency: str
    manager: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ClientCreate(BaseModel):
    name: str
    client_type: ClientType = ClientType.PARTICULIER
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    status: ClientStatus = ClientStatus.ACTIF

class ClientUpdate(BaseModel):
    name: Optional[str] = None
    client_type: Optional[ClientType] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    status: Optional[ClientStatus] = None

class ClientResponse(BaseModel):
    id: int
    name: str
    client_type: ClientType
    email: Optional[str]
    phone: Optional[str]
    address: Optional[str]
    status: ClientStatus
    created_at: datetime

    class Config:
        from_attributes = True

class PolicyCreate(BaseModel):
    policy_number: str
    client_id: int
    portfolio_id: Optional[int] = None
    effective_date: date
    expiry_date: Optional[date] = None
    premium_amount: float = 0.0
    currency: str = "TND"
    status: PolicyStatus = PolicyStatus.ACTIVE
    ifrs17_group: Optional[str] = None
    cohort_year: Optional[int] = None
    measurement_model: MeasurementModel = MeasurementModel.PAA

class PolicyUpdate(BaseModel):
    policy_number: Optional[str] = None
    client_id: Optional[int] = None
    portfolio_id: Optional[int] = None
    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None
    premium_amount: Optional[float] = None
    currency: Optional[str] = None
    status: Optional[PolicyStatus] = None
    ifrs17_group: Optional[str] = None
    cohort_year: Optional[int] = None
    measurement_model: Optional[MeasurementModel] = None

class PolicyResponse(BaseModel):
    id: int
    policy_number: str
    client_id: int
    portfolio_id: Optional[int]
    effective_date: date
    expiry_date: Optional[date]
    premium_amount: float
    currency: str
    status: PolicyStatus
    ifrs17_group: Optional[str]
    cohort_year: Optional[int]
    measurement_model: MeasurementModel
    created_at: datetime

    class Config:
        from_attributes = True

class CoverageCreate(BaseModel):
    policy_id: int
    name: str
    limit_amount: float = 0.0
    deductible: float = 0.0
    premium_amount: float = 0.0
    status: CoverageStatus = CoverageStatus.ACTIVE

class CoverageUpdate(BaseModel):
    name: Optional[str] = None
    limit_amount: Optional[float] = None
    deductible: Optional[float] = None
    premium_amount: Optional[float] = None
    status: Optional[CoverageStatus] = None

class CoverageResponse(BaseModel):
    id: int
    policy_id: int
    name: str
    limit_amount: float
    deductible: float
    premium_amount: float
    status: CoverageStatus
    created_at: datetime

    class Config:
        from_attributes = True

class ClaimCreate(BaseModel):
    policy_id: int
    claim_number: str
    reported_date: date
    occurrence_date: Optional[date] = None
    status: ClaimStatus = ClaimStatus.OPEN
    amount: float = 0.0
    paid_amount: float = 0.0
    currency: str = "TND"
    description: Optional[str] = None

    @model_validator(mode="after")
    def validate_paid_amount(self):
        if self.paid_amount > self.amount:
            raise ValueError('Le montant payé ne peut pas dépasser le montant déclaré')
        return self

class ClaimUpdate(BaseModel):
    claim_number: Optional[str] = None
    reported_date: Optional[date] = None
    occurrence_date: Optional[date] = None
    status: Optional[ClaimStatus] = None
    amount: Optional[float] = None
    paid_amount: Optional[float] = None
    currency: Optional[str] = None
    description: Optional[str] = None

class ClaimResponse(BaseModel):
    id: int
    policy_id: int
    claim_number: str
    reported_date: date
    occurrence_date: Optional[date]
    status: ClaimStatus
    amount: float
    paid_amount: float
    currency: str
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class InvoiceCreate(BaseModel):
    policy_id: int
    invoice_number: str
    issued_date: date
    due_date: Optional[date] = None
    amount: float = 0.0
    paid_amount: float = 0.0
    status: InvoiceStatus = InvoiceStatus.PENDING
    currency: str = "TND"

    @model_validator(mode="after")
    def validate_invoice_paid_amount(self):
        if self.paid_amount > self.amount:
            raise ValueError('Le montant payé ne peut pas dépasser le montant facturé')
        return self

class InvoiceUpdate(BaseModel):
    invoice_number: Optional[str] = None
    issued_date: Optional[date] = None
    due_date: Optional[date] = None
    amount: Optional[float] = None
    paid_amount: Optional[float] = None
    status: Optional[InvoiceStatus] = None
    currency: Optional[str] = None

class InvoiceResponse(BaseModel):
    id: int
    policy_id: int
    invoice_number: str
    issued_date: date
    due_date: Optional[date]
    amount: float
    paid_amount: float
    status: InvoiceStatus
    currency: str
    created_at: datetime

    class Config:
        from_attributes = True

class LedgerEntryCreate(BaseModel):
    policy_id: int
    entry_type: LedgerEntryType
    account_code: str
    description: Optional[str] = None
    amount: float = 0.0
    currency: str = "TND"
    entry_date: date
    reference: Optional[str] = None

class LedgerEntryUpdate(BaseModel):
    entry_type: Optional[LedgerEntryType] = None
    account_code: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    entry_date: Optional[date] = None
    reference: Optional[str] = None

class LedgerEntryResponse(BaseModel):
    id: int
    policy_id: int
    entry_type: LedgerEntryType
    account_code: str
    description: Optional[str]
    amount: float
    currency: str
    entry_date: date
    reference: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ERPDataQualityResponse(BaseModel):
    missing_policy_links: int = 0
    claims_paid_over_amount: int = 0
    invoices_paid_over_amount: int = 0
    inactive_clients: int = 0

class ERPSummaryResponse(BaseModel):
    portfolios: int = 0
    clients: int = 0
    policies: int = 0
    coverages: int = 0
    claims: int = 0
    invoices: int = 0
    ledger_entries: int = 0
