from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import csv
import io
from datetime import date
from typing import List

from backend.database.connection import get_db
from backend.database.models import (
    Portfolio,
    Client,
    Policy,
    Coverage,
    Claim,
    Invoice,
    LedgerEntry,
    AuditLog
)
from backend.database.schemas import (
    UserResponse,
    PortfolioCreate,
    PortfolioUpdate,
    PortfolioResponse,
    ClientCreate,
    ClientUpdate,
    ClientResponse,
    PolicyCreate,
    PolicyUpdate,
    PolicyResponse,
    CoverageCreate,
    CoverageUpdate,
    CoverageResponse,
    ClaimCreate,
    ClaimUpdate,
    ClaimResponse,
    InvoiceCreate,
    InvoiceUpdate,
    InvoiceResponse,
    LedgerEntryCreate,
    LedgerEntryUpdate,
    LedgerEntryResponse,
    ERPSummaryResponse,
    ERPDataQualityResponse
)
from backend.routers.auth_router import get_current_user

router = APIRouter(prefix="/erp", tags=["🏢 ERP Assurance"])

WRITE_ROLES = {"actuaire", "comptable"}


def require_write_role(current_user: UserResponse):
    if current_user.role not in WRITE_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé: rôle insuffisant pour modification ERP"
        )


def log_audit(db: Session, user: UserResponse, action: str, resource: str, details: str):
    audit = AuditLog(
        user_id=user.id,
        action=action,
        resource=resource,
        details=details
    )
    db.add(audit)


@router.get("/summary", response_model=ERPSummaryResponse)
def get_erp_summary(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    return ERPSummaryResponse(
        portfolios=db.query(Portfolio).count(),
        clients=db.query(Client).count(),
        policies=db.query(Policy).count(),
        coverages=db.query(Coverage).count(),
        claims=db.query(Claim).count(),
        invoices=db.query(Invoice).count(),
        ledger_entries=db.query(LedgerEntry).count()
    )


@router.get("/data-quality", response_model=ERPDataQualityResponse)
def get_data_quality(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    missing_policy_links = db.query(Policy).filter(Policy.portfolio_id.is_(None)).count()
    claims_paid_over = db.query(Claim).filter(Claim.paid_amount > Claim.amount).count()
    invoices_paid_over = db.query(Invoice).filter(Invoice.paid_amount > Invoice.amount).count()
    inactive_clients = db.query(Client).filter(Client.status == "inactif").count()

    return ERPDataQualityResponse(
        missing_policy_links=missing_policy_links,
        claims_paid_over_amount=claims_paid_over,
        invoices_paid_over_amount=invoices_paid_over,
        inactive_clients=inactive_clients
    )


# =========================
# Portfolios
# =========================

@router.get("/portfolios", response_model=List[PortfolioResponse])
def list_portfolios(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    return db.query(Portfolio).order_by(Portfolio.name).all()


@router.get("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
def get_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio introuvable")
    return portfolio


@router.post("/portfolios", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
def create_portfolio(
    payload: PortfolioCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    portfolio = Portfolio(**payload.dict())
    db.add(portfolio)
    log_audit(db, current_user, "create", "portfolio", f"Portfolio {payload.name}")
    db.commit()
    db.refresh(portfolio)
    return portfolio


@router.put("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
def update_portfolio(
    portfolio_id: int,
    payload: PortfolioUpdate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio introuvable")
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(portfolio, field, value)
    log_audit(db, current_user, "update", "portfolio", f"Portfolio {portfolio_id}")
    db.commit()
    db.refresh(portfolio)
    return portfolio


@router.delete("/portfolios/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio introuvable")
    db.delete(portfolio)
    log_audit(db, current_user, "delete", "portfolio", f"Portfolio {portfolio_id}")
    db.commit()
    return None


# =========================
# Clients
# =========================

@router.get("/clients", response_model=List[ClientResponse])
def list_clients(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    return db.query(Client).order_by(Client.name).all()


@router.get("/clients/{client_id}", response_model=ClientResponse)
def get_client(
    client_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client introuvable")
    return client


@router.post("/clients", response_model=ClientResponse, status_code=status.HTTP_201_CREATED)
def create_client(
    payload: ClientCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    client = Client(**payload.dict())
    db.add(client)
    log_audit(db, current_user, "create", "client", f"Client {payload.name}")
    db.commit()
    db.refresh(client)
    return client


@router.put("/clients/{client_id}", response_model=ClientResponse)
def update_client(
    client_id: int,
    payload: ClientUpdate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client introuvable")
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(client, field, value)
    log_audit(db, current_user, "update", "client", f"Client {client_id}")
    db.commit()
    db.refresh(client)
    return client


@router.delete("/clients/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_client(
    client_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client introuvable")
    db.delete(client)
    log_audit(db, current_user, "delete", "client", f"Client {client_id}")
    db.commit()
    return None


@router.post("/clients/import")
async def import_clients(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    content = await file.read()
    decoded = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))
    created = 0
    errors = []
    for idx, row in enumerate(reader, start=1):
        name = row.get("name") or row.get("nom")
        if not name:
            errors.append({"row": idx, "error": "Nom client manquant"})
            continue
        client = Client(
            name=name,
            client_type=row.get("client_type", "particulier"),
            email=row.get("email"),
            phone=row.get("phone"),
            address=row.get("address"),
            status=row.get("status", "actif")
        )
        db.add(client)
        created += 1
    log_audit(db, current_user, "import", "client", f"Import clients: {created}")
    db.commit()
    return {"created": created, "errors": errors}


@router.get("/clients/export")
def export_clients(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    clients = db.query(Client).order_by(Client.name).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "name", "client_type", "email", "phone", "address", "status"])
    for client in clients:
        writer.writerow([
            client.id,
            client.name,
            client.client_type,
            client.email or "",
            client.phone or "",
            client.address or "",
            client.status
        ])
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=clients.csv"}
    )


# =========================
# Policies
# =========================

@router.get("/policies", response_model=List[PolicyResponse])
def list_policies(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    return db.query(Policy).order_by(Policy.policy_number).all()


@router.get("/policies/{policy_id}", response_model=PolicyResponse)
def get_policy(
    policy_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    policy = db.query(Policy).filter(Policy.id == policy_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Police introuvable")
    return policy


@router.post("/policies", response_model=PolicyResponse, status_code=status.HTTP_201_CREATED)
def create_policy(
    payload: PolicyCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    client = db.query(Client).filter(Client.id == payload.client_id).first()
    if not client:
        raise HTTPException(status_code=400, detail="Client invalide")
    if payload.portfolio_id:
        portfolio = db.query(Portfolio).filter(Portfolio.id == payload.portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=400, detail="Portfolio invalide")
    policy = Policy(**payload.dict())
    db.add(policy)
    log_audit(db, current_user, "create", "policy", f"Policy {payload.policy_number}")
    db.commit()
    db.refresh(policy)
    return policy


@router.put("/policies/{policy_id}", response_model=PolicyResponse)
def update_policy(
    policy_id: int,
    payload: PolicyUpdate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    policy = db.query(Policy).filter(Policy.id == policy_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Police introuvable")
    data = payload.dict(exclude_unset=True)
    if "client_id" in data:
        if not db.query(Client).filter(Client.id == data["client_id"]).first():
            raise HTTPException(status_code=400, detail="Client invalide")
    if "portfolio_id" in data and data["portfolio_id"] is not None:
        if not db.query(Portfolio).filter(Portfolio.id == data["portfolio_id"]).first():
            raise HTTPException(status_code=400, detail="Portfolio invalide")
    for field, value in data.items():
        setattr(policy, field, value)
    log_audit(db, current_user, "update", "policy", f"Policy {policy_id}")
    db.commit()
    db.refresh(policy)
    return policy


@router.delete("/policies/{policy_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_policy(
    policy_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    policy = db.query(Policy).filter(Policy.id == policy_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Police introuvable")
    db.delete(policy)
    log_audit(db, current_user, "delete", "policy", f"Policy {policy_id}")
    db.commit()
    return None


@router.post("/policies/import")
async def import_policies(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    content = await file.read()
    decoded = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))
    created = 0
    errors = []
    for idx, row in enumerate(reader, start=1):
        policy_number = row.get("policy_number") or row.get("numero_police")
        client_id = row.get("client_id")
        effective_date = row.get("effective_date")
        if not policy_number or not client_id or not effective_date:
            errors.append({"row": idx, "error": "Champs obligatoires manquants"})
            continue
        if not db.query(Client).filter(Client.id == int(client_id)).first():
            errors.append({"row": idx, "error": "Client invalide"})
            continue
        try:
            effective_date_value = date.fromisoformat(effective_date)
            expiry_value = date.fromisoformat(row["expiry_date"]) if row.get("expiry_date") else None
        except ValueError:
            errors.append({"row": idx, "error": "Format de date invalide (YYYY-MM-DD requis)"})
            continue
        policy = Policy(
            policy_number=policy_number,
            client_id=int(client_id),
            portfolio_id=int(row["portfolio_id"]) if row.get("portfolio_id") else None,
            effective_date=effective_date_value,
            expiry_date=expiry_value,
            premium_amount=float(row.get("premium_amount", 0.0)),
            currency=row.get("currency", "TND"),
            status=row.get("status", "active"),
            ifrs17_group=row.get("ifrs17_group"),
            cohort_year=int(row["cohort_year"]) if row.get("cohort_year") else None,
            measurement_model=row.get("measurement_model", "PAA")
        )
        db.add(policy)
        created += 1
    log_audit(db, current_user, "import", "policy", f"Import policies: {created}")
    db.commit()
    return {"created": created, "errors": errors}


@router.get("/policies/export")
def export_policies(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    policies = db.query(Policy).order_by(Policy.policy_number).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "policy_number", "client_id", "portfolio_id",
        "effective_date", "expiry_date", "premium_amount",
        "currency", "status", "ifrs17_group", "cohort_year", "measurement_model"
    ])
    for policy in policies:
        writer.writerow([
            policy.id,
            policy.policy_number,
            policy.client_id,
            policy.portfolio_id or "",
            policy.effective_date,
            policy.expiry_date or "",
            policy.premium_amount,
            policy.currency,
            policy.status,
            policy.ifrs17_group or "",
            policy.cohort_year or "",
            policy.measurement_model
        ])
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=policies.csv"}
    )


# =========================
# Coverages
# =========================

@router.get("/coverages", response_model=List[CoverageResponse])
def list_coverages(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    return db.query(Coverage).order_by(Coverage.id.desc()).all()


@router.get("/coverages/{coverage_id}", response_model=CoverageResponse)
def get_coverage(
    coverage_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    coverage = db.query(Coverage).filter(Coverage.id == coverage_id).first()
    if not coverage:
        raise HTTPException(status_code=404, detail="Garantie introuvable")
    return coverage


@router.post("/coverages", response_model=CoverageResponse, status_code=status.HTTP_201_CREATED)
def create_coverage(
    payload: CoverageCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    if not db.query(Policy).filter(Policy.id == payload.policy_id).first():
        raise HTTPException(status_code=400, detail="Police invalide")
    coverage = Coverage(**payload.dict())
    db.add(coverage)
    log_audit(db, current_user, "create", "coverage", f"Coverage {payload.name}")
    db.commit()
    db.refresh(coverage)
    return coverage


@router.put("/coverages/{coverage_id}", response_model=CoverageResponse)
def update_coverage(
    coverage_id: int,
    payload: CoverageUpdate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    coverage = db.query(Coverage).filter(Coverage.id == coverage_id).first()
    if not coverage:
        raise HTTPException(status_code=404, detail="Garantie introuvable")
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(coverage, field, value)
    log_audit(db, current_user, "update", "coverage", f"Coverage {coverage_id}")
    db.commit()
    db.refresh(coverage)
    return coverage


@router.delete("/coverages/{coverage_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_coverage(
    coverage_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    coverage = db.query(Coverage).filter(Coverage.id == coverage_id).first()
    if not coverage:
        raise HTTPException(status_code=404, detail="Garantie introuvable")
    db.delete(coverage)
    log_audit(db, current_user, "delete", "coverage", f"Coverage {coverage_id}")
    db.commit()
    return None


# =========================
# Claims
# =========================

@router.get("/claims", response_model=List[ClaimResponse])
def list_claims(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    return db.query(Claim).order_by(Claim.reported_date.desc()).all()


@router.get("/claims/{claim_id}", response_model=ClaimResponse)
def get_claim(
    claim_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    claim = db.query(Claim).filter(Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Sinistre introuvable")
    return claim


@router.post("/claims", response_model=ClaimResponse, status_code=status.HTTP_201_CREATED)
def create_claim(
    payload: ClaimCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    if payload.paid_amount > payload.amount:
        raise HTTPException(status_code=400, detail="Montant payé supérieur au montant déclaré")
    if not db.query(Policy).filter(Policy.id == payload.policy_id).first():
        raise HTTPException(status_code=400, detail="Police invalide")
    claim = Claim(**payload.dict())
    db.add(claim)
    log_audit(db, current_user, "create", "claim", f"Claim {payload.claim_number}")
    db.commit()
    db.refresh(claim)
    return claim


@router.put("/claims/{claim_id}", response_model=ClaimResponse)
def update_claim(
    claim_id: int,
    payload: ClaimUpdate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    claim = db.query(Claim).filter(Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Sinistre introuvable")
    data = payload.dict(exclude_unset=True)
    if "paid_amount" in data:
        amount_value = data.get("amount", claim.amount)
        if data["paid_amount"] > amount_value:
            raise HTTPException(status_code=400, detail="Montant payé supérieur au montant déclaré")
    for field, value in data.items():
        setattr(claim, field, value)
    log_audit(db, current_user, "update", "claim", f"Claim {claim_id}")
    db.commit()
    db.refresh(claim)
    return claim


@router.delete("/claims/{claim_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_claim(
    claim_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    claim = db.query(Claim).filter(Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Sinistre introuvable")
    db.delete(claim)
    log_audit(db, current_user, "delete", "claim", f"Claim {claim_id}")
    db.commit()
    return None


# =========================
# Invoices
# =========================

@router.get("/invoices", response_model=List[InvoiceResponse])
def list_invoices(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    return db.query(Invoice).order_by(Invoice.issued_date.desc()).all()


@router.get("/invoices/{invoice_id}", response_model=InvoiceResponse)
def get_invoice(
    invoice_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Quittance introuvable")
    return invoice


@router.post("/invoices", response_model=InvoiceResponse, status_code=status.HTTP_201_CREATED)
def create_invoice(
    payload: InvoiceCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    if payload.paid_amount > payload.amount:
        raise HTTPException(status_code=400, detail="Montant payé supérieur au montant facturé")
    if not db.query(Policy).filter(Policy.id == payload.policy_id).first():
        raise HTTPException(status_code=400, detail="Police invalide")
    invoice = Invoice(**payload.dict())
    db.add(invoice)
    log_audit(db, current_user, "create", "invoice", f"Invoice {payload.invoice_number}")
    db.commit()
    db.refresh(invoice)
    return invoice


@router.put("/invoices/{invoice_id}", response_model=InvoiceResponse)
def update_invoice(
    invoice_id: int,
    payload: InvoiceUpdate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Quittance introuvable")
    data = payload.dict(exclude_unset=True)
    if "paid_amount" in data:
        amount_value = data.get("amount", invoice.amount)
        if data["paid_amount"] > amount_value:
            raise HTTPException(status_code=400, detail="Montant payé supérieur au montant facturé")
    for field, value in data.items():
        setattr(invoice, field, value)
    log_audit(db, current_user, "update", "invoice", f"Invoice {invoice_id}")
    db.commit()
    db.refresh(invoice)
    return invoice


@router.delete("/invoices/{invoice_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_invoice(
    invoice_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Quittance introuvable")
    db.delete(invoice)
    log_audit(db, current_user, "delete", "invoice", f"Invoice {invoice_id}")
    db.commit()
    return None


# =========================
# Ledger Entries
# =========================

@router.get("/ledger-entries", response_model=List[LedgerEntryResponse])
def list_ledger_entries(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    return db.query(LedgerEntry).order_by(LedgerEntry.entry_date.desc()).all()


@router.get("/ledger-entries/{entry_id}", response_model=LedgerEntryResponse)
def get_ledger_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    entry = db.query(LedgerEntry).filter(LedgerEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Écriture introuvable")
    return entry


@router.post("/ledger-entries", response_model=LedgerEntryResponse, status_code=status.HTTP_201_CREATED)
def create_ledger_entry(
    payload: LedgerEntryCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    if not db.query(Policy).filter(Policy.id == payload.policy_id).first():
        raise HTTPException(status_code=400, detail="Police invalide")
    entry = LedgerEntry(**payload.dict())
    db.add(entry)
    log_audit(db, current_user, "create", "ledger_entry", f"Entry {payload.entry_type}")
    db.commit()
    db.refresh(entry)
    return entry


@router.put("/ledger-entries/{entry_id}", response_model=LedgerEntryResponse)
def update_ledger_entry(
    entry_id: int,
    payload: LedgerEntryUpdate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    entry = db.query(LedgerEntry).filter(LedgerEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Écriture introuvable")
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(entry, field, value)
    log_audit(db, current_user, "update", "ledger_entry", f"Entry {entry_id}")
    db.commit()
    db.refresh(entry)
    return entry


@router.delete("/ledger-entries/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ledger_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    require_write_role(current_user)
    entry = db.query(LedgerEntry).filter(LedgerEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Écriture introuvable")
    db.delete(entry)
    log_audit(db, current_user, "delete", "ledger_entry", f"Entry {entry_id}")
    db.commit()
    return None
