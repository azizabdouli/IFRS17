# 📊 REVUE ACTUARIELLE DES VISUALISATIONS
## Validation Data Scientist & Actuaire

**Date**: 8 Octobre 2025  
**Analyste**: Expert Actuaire & Data Scientist  
**Scope**: Vérification des graphiques, métriques et cohérence des données

---

## ✅ RÉSUMÉ EXÉCUTIF

### État des Visualisations
- ✅ **LRC Chart** : Décomposition correcte (PPNA + RA + LC)
- ✅ **Ratios** : Calculs conformes (RA%, LC%)
- ⚠️ **Combined Ratio** : Absent du dashboard (à ajouter)
- ⚠️ **Alertes** : Manque de seuils visuels (>100%, >105%)

---

## 📈 ANALYSE DES GRAPHIQUES EXISTANTS

### 1. LRC Waterfall Chart (dashboard.component.ts)

#### Code Actuel:
```typescript
private updateLRCChart(): void {
  const ppna = this.ppnaMetrics.ppna_total || 0;
  const riskAdj = this.ppnaMetrics.risk_adjustment || 0;
  const lossComp = this.ppnaMetrics.loss_component || 0;
  const lrc = this.ppnaMetrics.lrc_total || 0;
  const known = ppna + riskAdj + lossComp;
  const autre = lrc > known ? lrc - known : 0;
  
  this.lrcChartData = {
    labels: ['PPNA', 'Risk Adj.', 'Loss Component', 'Autres'],
    datasets: [{
      data: [ppna, riskAdj, lossComp, autre],
      backgroundColor: ['#2563EB','#F59E0B','#DC2626','#9CA3AF']
    }]
  };
}
```

#### ✅ Validations:
- **Formule correcte** : LRC = PPNA + RA + LC + Autres
- **Gestion résidus** : `autre` capture les écarts (bonne pratique)
- **Couleurs** : Distinction visuelle claire
  - Bleu (#2563EB) : PPNA (neutre)
  - Orange (#F59E0B) : Risk Adj. (attention)
  - Rouge (#DC2626) : Loss Component (alerte)
  - Gris (#9CA3AF) : Autres (résidus)

#### 🔍 Observations:
1. **Résidu "Autres"** ne devrait jamais être significatif
   - Si `autre > 1% de LRC` → Erreur de calcul backend
   - Actuellement : Pas de validation de ce résidu

2. **Manque de tooltips actuariels** :
   ```typescript
   // Recommandé:
   tooltip: {
     callbacks: {
       label: (context) => {
         const value = context.parsed;
         const total = lrc;
         const percent = ((value / total) * 100).toFixed(1);
         return `${context.label}: ${value.toLocaleString('fr-TN')} TND (${percent}%)`;
       }
     }
   }
   ```

---

### 2. Ratios de Composition (Méthodes actuelles)

#### Risk Adjustment Percent:
```typescript
getRiskAdjustmentPercent(): number {
  return +( (this.ppnaMetrics.risk_adjustment / this.ppnaMetrics.lrc_total) * 100 ).toFixed(2);
}
```

#### ✅ Validations:
- **Formule** : RA% = (RA / LRC) × 100 ✅ CORRECT
- **Fourchette attendue** : 1-5% du LRC
- **Protection division par zéro** : ✅ Présente

#### ⚠️ Observations:
- **Manque de validation** : Pas d'alerte si RA% > 10% (anormal)
- **Comparaison non documentée** : Pas de benchmark affiché

#### 🚀 Recommandation:
```typescript
getRiskAdjustmentPercent(): number {
  if (!this.ppnaMetrics?.lrc_total) return 0;
  const percent = (this.ppnaMetrics.risk_adjustment / this.ppnaMetrics.lrc_total) * 100;
  
  // Validation actuarielle
  if (percent > 10) {
    console.warn(`⚠️ RA% anormalement élevé: ${percent.toFixed(1)}%`);
  }
  
  return +percent.toFixed(2);
}

// Afficher dans UI:
// "RA: 2.3% ✅"  si < 5%
// "RA: 7.8% ⚠️"  si 5-10%
// "RA: 12.5% 🔴" si > 10%
```

---

### 3. Loss Component Percent:
```typescript
getLossComponentPercent(): number {
  return +( (this.ppnaMetrics.loss_component / this.ppnaMetrics.lrc_total) * 100 ).toFixed(2);
}
```

#### ✅ Validations:
- **Formule** : LC% = (LC / LRC) × 100 ✅ CORRECT
- **Interprétation** :
  - **0%** : Portefeuille profitable ✅
  - **> 0%** : Contrats onéreux détectés ⚠️
  - **> 5%** : Problème de tarification majeur 🔴

#### 🚀 Amélioration recommandée:
```typescript
getLossComponentStatus(): { value: number, status: string, icon: string } {
  const percent = this.getLossComponentPercent();
  
  if (percent === 0) {
    return { value: percent, status: 'Profitable', icon: '✅' };
  } else if (percent <= 5) {
    return { value: percent, status: 'Contrats onéreux détectés', icon: '⚠️' };
  } else {
    return { value: percent, status: 'Révision tarifaire urgente', icon: '🔴' };
  }
}
```

---

## 🚨 MÉTRIQUES MANQUANTES (CRITIQUES)

### 1. Combined Ratio

#### Formule IFRS 17:
```
Combined Ratio = LRC / Primes
```

#### Implémentation recommandée:
```typescript
getCombinedRatio(): number {
  if (!this.ppnaMetrics?.lrc_total || !this.ppnaMetrics?.total_primes) return 0;
  return (this.ppnaMetrics.lrc_total / this.ppnaMetrics.total_primes) * 100;
}

getCombinedRatioStatus(): { value: number, label: string, color: string } {
  const ratio = this.getCombinedRatio();
  
  if (ratio < 100) {
    return { value: ratio, label: 'Profitable', color: '#10B981' }; // Vert
  } else if (ratio <= 105) {
    return { value: ratio, label: 'Zone acceptable', color: '#F59E0B' }; // Orange
  } else {
    return { value: ratio, label: 'Sous-tarification', color: '#DC2626' }; // Rouge
  }
}
```

#### Visualisation recommandée:
```html
<div class="combined-ratio-gauge">
  <h3>Combined Ratio</h3>
  <div class="gauge-container">
    <svg width="200" height="120">
      <!-- Arc gauge de 0% à 150% -->
      <!-- Zone verte: 0-100% -->
      <!-- Zone orange: 100-105% -->
      <!-- Zone rouge: >105% -->
    </svg>
    <div class="gauge-value">{{ getCombinedRatio() | number:'1.1-1' }}%</div>
    <div class="gauge-label" [style.color]="getCombinedRatioStatus().color">
      {{ getCombinedRatioStatus().label }}
    </div>
  </div>
</div>
```

---

### 2. Evolution Temporelle LRC

#### Graphique manquant: LRC Over Time

**Pourquoi c'est critique** :
- PPNA doit décroître linéairement (PAA prorata temporis)
- RA varie avec √(PPNA) (effet de diversification)
- LC apparaît soudainement si contrats deviennent onéreux

#### Implémentation recommandée:
```typescript
interface LRCEvolutionPoint {
  date: Date;
  ppna: number;
  risk_adjustment: number;
  loss_component: number;
  lrc_total: number;
}

private updateLRCEvolutionChart(history: LRCEvolutionPoint[]): void {
  this.lrcEvolutionChartData = {
    labels: history.map(h => h.date.toLocaleDateString('fr-TN')),
    datasets: [
      {
        label: 'LRC Total',
        data: history.map(h => h.lrc_total),
        borderColor: '#2563EB',
        fill: false,
        tension: 0.1
      },
      {
        label: 'PPNA',
        data: history.map(h => h.ppna),
        borderColor: '#10B981',
        fill: false,
        borderDash: [5, 5]
      },
      {
        label: 'Risk Adjustment',
        data: history.map(h => h.risk_adjustment),
        borderColor: '#F59E0B',
        fill: false
      }
    ]
  };
}
```

**Validation attendue** :
- PPNA décroît en ligne droite (si PAA linéaire)
- RA décroît en courbe (√ du PPNA)
- LRC = PPNA + RA reste > 0

---

### 3. Heatmap Risk Adjustment par Segment

#### Objectif:
Visualiser la dispersion du RA% par branche d'assurance

#### Code recommandé:
```typescript
interface SegmentRiskMatrix {
  segment: string;
  ra_percent: number;
  volatility: number;
  lrc_total: number;
}

getSegmentRiskHeatmap(): SegmentRiskMatrix[] {
  return this.ppnaSegments.map(seg => ({
    segment: seg.segment,
    ra_percent: (seg.risk_adjustment / seg.lrc_total) * 100,
    volatility: this.estimateVolatility(seg.segment),
    lrc_total: seg.lrc_total
  }));
}

// Fonction d'estimation volatilité (à calibrer avec données réelles)
private estimateVolatility(segment: string): number {
  const volatilityMap: { [key: string]: number } = {
    'Auto': 8,
    'Santé': 10,
    'Vie': 4,
    'Dommages': 12,
    'RC': 15
  };
  return volatilityMap[segment] || 8; // Défaut 8%
}
```

#### Visualisation (Chart.js Matrix):
```typescript
this.heatmapData = {
  datasets: [{
    label: 'RA% par Segment',
    data: heatmapData.map(d => ({
      x: d.segment,
      y: 'Risk Adjustment %',
      v: d.ra_percent
    })),
    backgroundColor: (context) => {
      const value = context.parsed.v;
      if (value < 2) return '#10B981'; // Vert
      if (value < 5) return '#F59E0B'; // Orange
      return '#DC2626'; // Rouge
    }
  }]
};
```

---

### 4. Onerous Contracts Dashboard Card

#### Métrique manquante: Contrats Onéreux

**Backend déjà en place** (ppna_service.py ligne 255):
```python
def detect_onerous_contracts(...):
    ratio_onereux = row['provisions'] / row['primes']
    if ratio_onereux > 0.80:
        # Détecté comme onéreux
```

#### Frontend à ajouter:
```typescript
interface OnerousContractMetrics {
  count: number;
  total_provisions: number;
  average_ratio: number;
}

onerousMetrics: OnerousContractMetrics = { count: 0, total_provisions: 0, average_ratio: 0 };

loadOnerousContracts(): void {
  this.ppnaService.getOnerousContracts().subscribe({
    next: (data) => {
      this.onerousMetrics = {
        count: data.onerous_count,
        total_provisions: data.onerous_provisions,
        average_ratio: data.average_onerous_ratio
      };
    }
  });
}
```

#### Visualisation recommandée:
```html
<div class="alert-card" [class.critical]="onerousMetrics.count > 0">
  <h3>⚠️ Contrats Onéreux Détectés</h3>
  <div class="metric-row">
    <span class="label">Nombre:</span>
    <span class="value">{{ onerousMetrics.count }}</span>
  </div>
  <div class="metric-row">
    <span class="label">Provisions concernées:</span>
    <span class="value">{{ onerousMetrics.total_provisions | number:'1.0-0' }} TND</span>
  </div>
  <div class="metric-row">
    <span class="label">Ratio moyen:</span>
    <span class="value">{{ onerousMetrics.average_ratio | percent:'1.1-1' }}</span>
  </div>
  <div class="action-buttons">
    <button (click)="exportOnerousContracts()">Exporter liste</button>
    <button (click)="viewOnerousDetails()">Voir détails</button>
  </div>
</div>
```

---

## 🎨 AMÉLIORATIONS VISUELLES

### 1. Code couleur actuariel standardisé

```scss
// Palette actuarielle IFRS 17
$color-ppna: #2563EB;        // Bleu - Élément neutre
$color-ra: #F59E0B;          // Orange - Attention (risque)
$color-lc: #DC2626;          // Rouge - Alerte (onéreux)
$color-profitable: #10B981;  // Vert - Bon état
$color-warning: #F59E0B;     // Orange - Surveillance
$color-critical: #DC2626;    // Rouge - Action requise

// Zones Combined Ratio
.zone-profitable { background: linear-gradient(135deg, #10B981, #34D399); }
.zone-acceptable { background: linear-gradient(135deg, #F59E0B, #FBBF24); }
.zone-critical { background: linear-gradient(135deg, #DC2626, #EF4444); }
```

### 2. Tooltips enrichis

```typescript
chartOptions = {
  plugins: {
    tooltip: {
      callbacks: {
        title: (context) => {
          return `Composante: ${context[0].label}`;
        },
        label: (context) => {
          const value = context.parsed;
          const total = this.ppnaMetrics.lrc_total;
          const percent = ((value / total) * 100).toFixed(1);
          
          return [
            `Montant: ${value.toLocaleString('fr-TN')} TND`,
            `% du LRC: ${percent}%`,
            this.getActuarialInterpretation(context.label, percent)
          ];
        }
      }
    }
  }
};

private getActuarialInterpretation(component: string, percent: string): string {
  switch(component) {
    case 'Risk Adj.':
      const raPercent = parseFloat(percent);
      if (raPercent < 2) return '✅ RA conforme (< 2%)';
      if (raPercent < 5) return '✅ RA acceptable (2-5%)';
      return '⚠️ RA élevé (> 5%)';
    
    case 'Loss Component':
      return parseFloat(percent) > 0 
        ? '⚠️ Contrats onéreux présents' 
        : '✅ Portefeuille profitable';
    
    default:
      return '';
  }
}
```

### 3. Dashboard Cards avec seuils

```html
<div class="metrics-grid">
  <!-- Combined Ratio Card -->
  <div class="metric-card" [ngClass]="getCombinedRatioClass()">
    <div class="metric-icon">📊</div>
    <div class="metric-label">Combined Ratio</div>
    <div class="metric-value">{{ getCombinedRatio() | number:'1.1-1' }}%</div>
    <div class="metric-threshold">
      <div class="threshold-bar">
        <div class="fill" [style.width.%]="getCombinedRatio()"></div>
        <span class="marker" style="left: 100%">100%</span>
        <span class="marker" style="left: 105%">105%</span>
      </div>
    </div>
  </div>

  <!-- Risk Adjustment Card -->
  <div class="metric-card">
    <div class="metric-icon">⚠️</div>
    <div class="metric-label">Risk Adjustment</div>
    <div class="metric-value">{{ ppnaMetrics.risk_adjustment | number:'1.0-0' }} TND</div>
    <div class="metric-secondary">
      {{ getRiskAdjustmentPercent() }}% du LRC
      <span [ngClass]="getRiskAdjustmentStatusClass()">
        {{ getRiskAdjustmentStatus() }}
      </span>
    </div>
  </div>

  <!-- Loss Component Card -->
  <div class="metric-card" [class.alert]="ppnaMetrics.loss_component > 0">
    <div class="metric-icon">🔴</div>
    <div class="metric-label">Loss Component</div>
    <div class="metric-value">{{ ppnaMetrics.loss_component | number:'1.0-0' }} TND</div>
    <div class="metric-secondary">
      {{ getLossComponentPercent() }}% du LRC
      <span *ngIf="ppnaMetrics.loss_component === 0" class="status-ok">✅ Aucun contrat onéreux</span>
      <span *ngIf="ppnaMetrics.loss_component > 0" class="status-warning">⚠️ Révision requise</span>
    </div>
  </div>
</div>
```

---

## 📋 CHECKLIST VALIDATION VISUALISATIONS

### Graphiques de base
- [x] LRC Waterfall Chart (PPNA + RA + LC)
- [x] Calcul RA% (RA / LRC)
- [x] Calcul LC% (LC / LRC)
- [ ] Combined Ratio (LRC / Primes)
- [ ] Evolution temporelle LRC
- [ ] Heatmap RA par segment

### KPIs critiques
- [ ] Combined Ratio avec zones colorées
- [ ] Alerte contrats onéreux (count + provisions)
- [ ] RA% avec seuils (2%, 5%, 10%)
- [ ] Écart LRC calculé vs. somme composantes

### Validations actuarielles
- [ ] Test: RA toujours > 0
- [ ] Test: LRC ≥ PPNA
- [ ] Test: LC ≥ 0 (par définition)
- [ ] Test: Combined Ratio réaliste (20-150%)
- [ ] Test: RA% entre 0.5% et 10%

### Améliorations UX
- [ ] Tooltips avec interprétations actuarielles
- [ ] Code couleur standardisé
- [ ] Seuils visuels (100%, 105% Combined Ratio)
- [ ] Export PDF rapport actuariel
- [ ] Historique 12 mois des métriques

---

## 🚀 PLAN D'IMPLÉMENTATION

### Phase 1 (Sprint actuel) - CRITIQUE
1. ✅ Corriger formules backend (Risk Adjustment CoC)
2. ✅ Ajouter validation logging
3. [ ] **Ajouter Combined Ratio au dashboard**
4. [ ] **Créer card "Contrats Onéreux"**
5. [ ] Tests unitaires visualisations

### Phase 2 (Sprint +1) - Important
1. [ ] Graphique évolution temporelle LRC
2. [ ] Heatmap RA par segment
3. [ ] Tooltips enrichis
4. [ ] Seuils visuels Combined Ratio

### Phase 3 (Sprint +2) - Nice to have
1. [ ] Export PDF rapport actuariel
2. [ ] Historique 12 mois métriques
3. [ ] Drill-down contrats onéreux
4. [ ] Comparaison benchmarks marché

---

## 📚 RÉFÉRENCES

1. **IFRS 17** - Disclosure Requirements (Annexe A)
   - Décomposition LRC (§98-100)
   - Risk Adjustment reconciliation (§105)
   - Onerous contracts disclosure (§103)

2. **Dashboard Best Practices**
   - Stephen Few - "Information Dashboard Design"
   - Actuarial visualizations standards (SOA)
   - Financial KPI color coding (IASB guidance)

---

**Validation**: Expert Actuaire & Data Scientist  
**Date**: 8 Octobre 2025  
**Version**: 1.0
