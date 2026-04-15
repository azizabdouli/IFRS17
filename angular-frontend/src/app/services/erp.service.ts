import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export interface ERPSummary {
  portfolios: number;
  clients: number;
  policies: number;
  coverages: number;
  claims: number;
  invoices: number;
  ledger_entries: number;
}

export interface ERPDataQuality {
  missing_policy_links: number;
  claims_paid_over_amount: number;
  invoices_paid_over_amount: number;
  inactive_clients: number;
}

export interface Portfolio {
  id: number;
  name: string;
  description?: string;
  currency: string;
  manager?: string;
  created_at: string;
}

export interface Client {
  id: number;
  name: string;
  client_type: string;
  email?: string;
  phone?: string;
  address?: string;
  status: string;
  created_at: string;
}

export interface Policy {
  id: number;
  policy_number: string;
  client_id: number;
  portfolio_id?: number;
  effective_date: string;
  expiry_date?: string;
  premium_amount: number;
  currency: string;
  status: string;
  ifrs17_group?: string;
  cohort_year?: number;
  measurement_model: string;
  created_at: string;
}

export interface Coverage {
  id: number;
  policy_id: number;
  name: string;
  limit_amount: number;
  deductible: number;
  premium_amount: number;
  status: string;
  created_at: string;
}

export interface Claim {
  id: number;
  policy_id: number;
  claim_number: string;
  reported_date: string;
  occurrence_date?: string;
  status: string;
  amount: number;
  paid_amount: number;
  currency: string;
  description?: string;
  created_at: string;
}

export interface Invoice {
  id: number;
  policy_id: number;
  invoice_number: string;
  issued_date: string;
  due_date?: string;
  amount: number;
  paid_amount: number;
  status: string;
  currency: string;
  created_at: string;
}

export interface LedgerEntry {
  id: number;
  policy_id: number;
  entry_type: string;
  account_code: string;
  description?: string;
  amount: number;
  currency: string;
  entry_date: string;
  reference?: string;
  created_at: string;
}

@Injectable({
  providedIn: 'root'
})
export class ERPService {
  private readonly baseUrl = `${environment.apiUrl}/erp`;

  constructor(private http: HttpClient) {}

  private handleError(error: HttpErrorResponse) {
    const detail = error.error?.detail || error.message;
    return throwError(() => new Error(detail));
  }

  getSummary(): Observable<ERPSummary> {
    return this.http.get<ERPSummary>(`${this.baseUrl}/summary`).pipe(
      catchError(this.handleError)
    );
  }

  getDataQuality(): Observable<ERPDataQuality> {
    return this.http.get<ERPDataQuality>(`${this.baseUrl}/data-quality`).pipe(
      catchError(this.handleError)
    );
  }

  listPortfolios(): Observable<Portfolio[]> {
    return this.http.get<Portfolio[]>(`${this.baseUrl}/portfolios`).pipe(
      catchError(this.handleError)
    );
  }

  createPortfolio(payload: Partial<Portfolio>): Observable<Portfolio> {
    return this.http.post<Portfolio>(`${this.baseUrl}/portfolios`, payload).pipe(
      catchError(this.handleError)
    );
  }

  updatePortfolio(id: number, payload: Partial<Portfolio>): Observable<Portfolio> {
    return this.http.put<Portfolio>(`${this.baseUrl}/portfolios/${id}`, payload).pipe(
      catchError(this.handleError)
    );
  }

  deletePortfolio(id: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/portfolios/${id}`).pipe(
      catchError(this.handleError)
    );
  }

  listClients(): Observable<Client[]> {
    return this.http.get<Client[]>(`${this.baseUrl}/clients`).pipe(
      catchError(this.handleError)
    );
  }

  createClient(payload: Partial<Client>): Observable<Client> {
    return this.http.post<Client>(`${this.baseUrl}/clients`, payload).pipe(
      catchError(this.handleError)
    );
  }

  updateClient(id: number, payload: Partial<Client>): Observable<Client> {
    return this.http.put<Client>(`${this.baseUrl}/clients/${id}`, payload).pipe(
      catchError(this.handleError)
    );
  }

  deleteClient(id: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/clients/${id}`).pipe(
      catchError(this.handleError)
    );
  }

  listPolicies(): Observable<Policy[]> {
    return this.http.get<Policy[]>(`${this.baseUrl}/policies`).pipe(
      catchError(this.handleError)
    );
  }

  createPolicy(payload: Partial<Policy>): Observable<Policy> {
    return this.http.post<Policy>(`${this.baseUrl}/policies`, payload).pipe(
      catchError(this.handleError)
    );
  }

  updatePolicy(id: number, payload: Partial<Policy>): Observable<Policy> {
    return this.http.put<Policy>(`${this.baseUrl}/policies/${id}`, payload).pipe(
      catchError(this.handleError)
    );
  }

  deletePolicy(id: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/policies/${id}`).pipe(
      catchError(this.handleError)
    );
  }

  listCoverages(): Observable<Coverage[]> {
    return this.http.get<Coverage[]>(`${this.baseUrl}/coverages`).pipe(
      catchError(this.handleError)
    );
  }

  createCoverage(payload: Partial<Coverage>): Observable<Coverage> {
    return this.http.post<Coverage>(`${this.baseUrl}/coverages`, payload).pipe(
      catchError(this.handleError)
    );
  }

  updateCoverage(id: number, payload: Partial<Coverage>): Observable<Coverage> {
    return this.http.put<Coverage>(`${this.baseUrl}/coverages/${id}`, payload).pipe(
      catchError(this.handleError)
    );
  }

  deleteCoverage(id: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/coverages/${id}`).pipe(
      catchError(this.handleError)
    );
  }

  listClaims(): Observable<Claim[]> {
    return this.http.get<Claim[]>(`${this.baseUrl}/claims`).pipe(
      catchError(this.handleError)
    );
  }

  createClaim(payload: Partial<Claim>): Observable<Claim> {
    return this.http.post<Claim>(`${this.baseUrl}/claims`, payload).pipe(
      catchError(this.handleError)
    );
  }

  updateClaim(id: number, payload: Partial<Claim>): Observable<Claim> {
    return this.http.put<Claim>(`${this.baseUrl}/claims/${id}`, payload).pipe(
      catchError(this.handleError)
    );
  }

  deleteClaim(id: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/claims/${id}`).pipe(
      catchError(this.handleError)
    );
  }

  listInvoices(): Observable<Invoice[]> {
    return this.http.get<Invoice[]>(`${this.baseUrl}/invoices`).pipe(
      catchError(this.handleError)
    );
  }

  createInvoice(payload: Partial<Invoice>): Observable<Invoice> {
    return this.http.post<Invoice>(`${this.baseUrl}/invoices`, payload).pipe(
      catchError(this.handleError)
    );
  }

  updateInvoice(id: number, payload: Partial<Invoice>): Observable<Invoice> {
    return this.http.put<Invoice>(`${this.baseUrl}/invoices/${id}`, payload).pipe(
      catchError(this.handleError)
    );
  }

  deleteInvoice(id: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/invoices/${id}`).pipe(
      catchError(this.handleError)
    );
  }

  listLedgerEntries(): Observable<LedgerEntry[]> {
    return this.http.get<LedgerEntry[]>(`${this.baseUrl}/ledger-entries`).pipe(
      catchError(this.handleError)
    );
  }

  createLedgerEntry(payload: Partial<LedgerEntry>): Observable<LedgerEntry> {
    return this.http.post<LedgerEntry>(`${this.baseUrl}/ledger-entries`, payload).pipe(
      catchError(this.handleError)
    );
  }

  updateLedgerEntry(id: number, payload: Partial<LedgerEntry>): Observable<LedgerEntry> {
    return this.http.put<LedgerEntry>(`${this.baseUrl}/ledger-entries/${id}`, payload).pipe(
      catchError(this.handleError)
    );
  }

  deleteLedgerEntry(id: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/ledger-entries/${id}`).pipe(
      catchError(this.handleError)
    );
  }
}
