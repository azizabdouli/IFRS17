import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Observable, Subscription } from 'rxjs';
import {
  ERPService,
  Portfolio,
  Client,
  Policy,
  Coverage,
  Claim,
  Invoice,
  LedgerEntry
} from '../../services/erp.service';

type ERPItem = Portfolio | Client | Policy | Coverage | Claim | Invoice | LedgerEntry;

interface FieldConfig {
  name: string;
  label: string;
  type: 'text' | 'number' | 'date' | 'select';
  required?: boolean;
  options?: { value: string; label: string }[];
}

interface EntityConfig {
  label: string;
  list: () => Observable<ERPItem[]>;
  create: (payload: Record<string, unknown>) => Observable<ERPItem>;
  update: (id: number, payload: Record<string, unknown>) => Observable<ERPItem>;
  remove: (id: number) => Observable<void>;
  fields: FieldConfig[];
  columns: string[];
}

@Component({
  selector: 'app-erp-management',
  standalone: true,
  imports: [CommonModule, RouterModule, ReactiveFormsModule],
  templateUrl: './erp-management.component.html',
  styleUrls: ['./erp-management.component.scss']
})
export class ErpManagementComponent implements OnInit, OnDestroy {
  form: FormGroup;
  items: ERPItem[] = [];
  entityKey = '';
  entityConfig?: EntityConfig;
  error: string | null = null;
  loading = false;
  editingId: number | null = null;
  private subscriptions: Subscription[] = [];

  constructor(
    private route: ActivatedRoute,
    private fb: FormBuilder,
    private erpService: ERPService
  ) {
    this.form = this.fb.group({});
  }

  ngOnInit(): void {
    const sub = this.route.paramMap.subscribe(params => {
      this.entityKey = params.get('entity') || 'clients';
      this.configureEntity();
      this.loadItems();
    });
    this.subscriptions.push(sub);
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  configureEntity(): void {
    const configs: Record<string, EntityConfig> = {
      portfolios: {
        label: 'Portefeuilles',
        list: () => this.erpService.listPortfolios(),
        create: (payload) => this.erpService.createPortfolio(payload as Partial<Portfolio>),
        update: (id, payload) => this.erpService.updatePortfolio(id, payload as Partial<Portfolio>),
        remove: (id) => this.erpService.deletePortfolio(id),
        fields: [
          { name: 'name', label: 'Nom', type: 'text', required: true },
          { name: 'description', label: 'Description', type: 'text' },
          { name: 'currency', label: 'Devise', type: 'text' },
          { name: 'manager', label: 'Gestionnaire', type: 'text' }
        ],
        columns: ['name', 'currency', 'manager']
      },
      clients: {
        label: 'Clients',
        list: () => this.erpService.listClients(),
        create: (payload) => this.erpService.createClient(payload as Partial<Client>),
        update: (id, payload) => this.erpService.updateClient(id, payload as Partial<Client>),
        remove: (id) => this.erpService.deleteClient(id),
        fields: [
          { name: 'name', label: 'Nom', type: 'text', required: true },
          { name: 'client_type', label: 'Type', type: 'select', options: [
            { value: 'particulier', label: 'Particulier' },
            { value: 'entreprise', label: 'Entreprise' },
            { value: 'intermediaire', label: 'Intermédiaire' }
          ]},
          { name: 'email', label: 'Email', type: 'text' },
          { name: 'phone', label: 'Téléphone', type: 'text' },
          { name: 'status', label: 'Statut', type: 'select', options: [
            { value: 'actif', label: 'Actif' },
            { value: 'inactif', label: 'Inactif' }
          ]}
        ],
        columns: ['name', 'client_type', 'email', 'status']
      },
      policies: {
        label: 'Polices',
        list: () => this.erpService.listPolicies(),
        create: (payload) => this.erpService.createPolicy(payload as Partial<Policy>),
        update: (id, payload) => this.erpService.updatePolicy(id, payload as Partial<Policy>),
        remove: (id) => this.erpService.deletePolicy(id),
        fields: [
          { name: 'policy_number', label: 'Numéro', type: 'text', required: true },
          { name: 'client_id', label: 'Client ID', type: 'number', required: true },
          { name: 'portfolio_id', label: 'Portfolio ID', type: 'number' },
          { name: 'effective_date', label: 'Date effet', type: 'date', required: true },
          { name: 'expiry_date', label: 'Date expiration', type: 'date' },
          { name: 'premium_amount', label: 'Prime', type: 'number' },
          { name: 'status', label: 'Statut', type: 'select', options: [
            { value: 'active', label: 'Active' },
            { value: 'expired', label: 'Expirée' },
            { value: 'suspended', label: 'Suspendue' }
          ]},
          { name: 'measurement_model', label: 'Modèle IFRS17', type: 'select', options: [
            { value: 'PAA', label: 'PAA' },
            { value: 'GMM', label: 'GMM' },
            { value: 'VFA', label: 'VFA' }
          ]}
        ],
        columns: ['policy_number', 'client_id', 'premium_amount', 'status']
      },
      coverages: {
        label: 'Garanties',
        list: () => this.erpService.listCoverages(),
        create: (payload) => this.erpService.createCoverage(payload as Partial<Coverage>),
        update: (id, payload) => this.erpService.updateCoverage(id, payload as Partial<Coverage>),
        remove: (id) => this.erpService.deleteCoverage(id),
        fields: [
          { name: 'policy_id', label: 'Police ID', type: 'number', required: true },
          { name: 'name', label: 'Nom', type: 'text', required: true },
          { name: 'limit_amount', label: 'Plafond', type: 'number' },
          { name: 'deductible', label: 'Franchise', type: 'number' },
          { name: 'premium_amount', label: 'Prime', type: 'number' }
        ],
        columns: ['name', 'policy_id', 'limit_amount', 'premium_amount']
      },
      claims: {
        label: 'Sinistres',
        list: () => this.erpService.listClaims(),
        create: (payload) => this.erpService.createClaim(payload as Partial<Claim>),
        update: (id, payload) => this.erpService.updateClaim(id, payload as Partial<Claim>),
        remove: (id) => this.erpService.deleteClaim(id),
        fields: [
          { name: 'claim_number', label: 'Numéro', type: 'text', required: true },
          { name: 'policy_id', label: 'Police ID', type: 'number', required: true },
          { name: 'reported_date', label: 'Date déclaration', type: 'date', required: true },
          { name: 'status', label: 'Statut', type: 'select', options: [
            { value: 'open', label: 'Ouvert' },
            { value: 'closed', label: 'Clos' },
            { value: 'rejected', label: 'Rejeté' }
          ]},
          { name: 'amount', label: 'Montant', type: 'number' },
          { name: 'paid_amount', label: 'Payé', type: 'number' }
        ],
        columns: ['claim_number', 'policy_id', 'status', 'amount']
      },
      invoices: {
        label: 'Quittances',
        list: () => this.erpService.listInvoices(),
        create: (payload) => this.erpService.createInvoice(payload as Partial<Invoice>),
        update: (id, payload) => this.erpService.updateInvoice(id, payload as Partial<Invoice>),
        remove: (id) => this.erpService.deleteInvoice(id),
        fields: [
          { name: 'invoice_number', label: 'Numéro', type: 'text', required: true },
          { name: 'policy_id', label: 'Police ID', type: 'number', required: true },
          { name: 'issued_date', label: 'Date émission', type: 'date', required: true },
          { name: 'due_date', label: 'Date échéance', type: 'date' },
          { name: 'status', label: 'Statut', type: 'select', options: [
            { value: 'pending', label: 'En attente' },
            { value: 'paid', label: 'Payée' },
            { value: 'overdue', label: 'En retard' }
          ]},
          { name: 'amount', label: 'Montant', type: 'number' },
          { name: 'paid_amount', label: 'Payé', type: 'number' }
        ],
        columns: ['invoice_number', 'policy_id', 'status', 'amount']
      },
      'ledger-entries': {
        label: 'Écritures',
        list: () => this.erpService.listLedgerEntries(),
        create: (payload) => this.erpService.createLedgerEntry(payload as Partial<LedgerEntry>),
        update: (id, payload) => this.erpService.updateLedgerEntry(id, payload as Partial<LedgerEntry>),
        remove: (id) => this.erpService.deleteLedgerEntry(id),
        fields: [
          { name: 'policy_id', label: 'Police ID', type: 'number', required: true },
          { name: 'entry_type', label: 'Type', type: 'select', options: [
            { value: 'premium', label: 'Prime' },
            { value: 'claim', label: 'Sinistre' },
            { value: 'commission', label: 'Commission' },
            { value: 'adjustment', label: 'Ajustement' }
          ]},
          { name: 'account_code', label: 'Compte', type: 'text', required: true },
          { name: 'amount', label: 'Montant', type: 'number' },
          { name: 'entry_date', label: 'Date', type: 'date', required: true }
        ],
        columns: ['entry_type', 'account_code', 'amount', 'entry_date']
      }
    };

    this.entityConfig = configs[this.entityKey] || configs.clients;
    this.buildForm();
  }

  buildForm(): void {
    const group: Record<string, unknown> = {};
    this.entityConfig?.fields.forEach(field => {
      const validators = field.required ? [Validators.required] : [];
      group[field.name] = ['', validators];
    });
    this.form = this.fb.group(group);
    this.editingId = null;
  }

  loadItems(): void {
    if (!this.entityConfig) {
      return;
    }
    this.error = null;
    this.loading = true;
    this.loadFrom(this.entityConfig.list());
  }

  loadFrom(stream: Observable<ERPItem[]>): void {
    const sub = stream.subscribe({
      next: (items: ERPItem[]) => {
        this.items = items;
        this.loading = false;
      },
      error: (err: Error) => {
        this.error = err.message;
        this.loading = false;
      }
    });
    this.subscriptions.push(sub);
  }

  mutate(stream: Observable<ERPItem | void>): void {
    const sub = stream.subscribe({
      next: () => {
        this.loadItems();
        this.resetForm();
      },
      error: (err: Error) => {
        this.error = err.message;
      }
    });
    this.subscriptions.push(sub);
  }

  submit(): void {
    if (!this.entityConfig) {
      return;
    }
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const payload = this.normalizePayload(this.form.value);
    if (this.editingId) {
      this.mutate(this.entityConfig.update(this.editingId, payload));
    } else {
      this.mutate(this.entityConfig.create(payload));
    }
  }

  edit(item: ERPItem): void {
    this.editingId = item.id;
    this.form.patchValue(item as any);
  }

  remove(item: ERPItem): void {
    if (!this.entityConfig) {
      return;
    }
    if (confirm('Confirmer la suppression ?')) {
      this.mutate(this.entityConfig.remove(item.id));
    }
  }

  resetForm(): void {
    this.form.reset();
    this.editingId = null;
  }

  private normalizePayload(value: Record<string, any>): Record<string, unknown> {
    const payload: Record<string, unknown> = {};
    this.entityConfig?.fields.forEach(field => {
      const raw = value[field.name];
      if (raw === '' || raw === null || typeof raw === 'undefined') {
        return;
      }
      if (field.type === 'number') {
        payload[field.name] = Number(raw);
      } else {
        payload[field.name] = raw;
      }
    });
    return payload;
  }
}
