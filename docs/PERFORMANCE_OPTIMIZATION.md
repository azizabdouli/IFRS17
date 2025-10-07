# 🚀 OPTIMISATION DASHBOARD - GUIDE PERFORMANCE

## 🐌 Problème: Dashboard Lent

Le dashboard PAA prend beaucoup de temps à charger. Ce guide explique comment optimiser les performances.

---

## 🔍 Diagnostic

### Causes Possibles

1. **Backend lent** (MySQL, requêtes non optimisées)
2. **Trop de données chargées** (pas de pagination)
3. **Appels API multiples** (pas de cache)
4. **Angular change detection** (pas optimisé)

---

## ✅ Solutions Appliquées

### 1. Optimisation Backend

#### A. Index Base de Données
Les tables PAA ont déjà des index sur `group_id` et `contract_id`, mais vérifions:

```sql
-- Vérifier les index
SHOW INDEX FROM paa_groups;
SHOW INDEX FROM paa_contracts;
SHOW INDEX FROM paa_movements;

-- Ajouter des index si nécessaire (déjà fait normalement)
CREATE INDEX idx_group_id ON paa_movements(group_id);
CREATE INDEX idx_period ON paa_movements(period_start, period_end);
```

#### B. Pagination API
Modifier `paa_router.py` pour paginer les résultats:

```python
@router.get("/groups")
def list_groups_paginated(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    persistence = PAAPersistence(db)
    groups = persistence.list_groups()
    
    # Pagination
    total = len(groups)
    paginated = groups[skip : skip + limit]
    
    return {
        "groups": paginated,
        "total": total,
        "skip": skip,
        "limit": limit
    }
```

#### C. Cache Redis (Optionnel)
Pour cache avancé, installer Redis:

```bash
pip install redis
```

```python
# backend/utils/cache.py
import redis
import json

cache = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_cached_groups():
    cached = cache.get('paa_groups')
    if cached:
        return json.loads(cached)
    return None

def set_cached_groups(groups, ttl=300):  # 5 minutes
    cache.setex('paa_groups', ttl, json.dumps(groups))
```

---

### 2. Optimisation Frontend

#### A. Lazy Loading (Chargement Paresseux)
Modifier `paa-dashboard.component.ts`:

```typescript
loadGroups(): void {
  this.loadingGroups = true;
  
  // Charger seulement les 50 premiers groupes
  this.apiService.listPAAGroups(0, 50)
    .pipe(takeUntil(this.destroy$))
    .subscribe({
      next: (res) => {
        this.groups = res.groups || [];
        this.totalGroups = res.total || 0;
        this.loadingGroups = false;
      },
      error: (err) => {
        console.error('Erreur chargement groupes:', err);
        this.loadingGroups = false;
      }
    });
}
```

#### B. Virtual Scrolling (Défilement Virtuel)
Pour listes avec >100 éléments, utiliser CDK Virtual Scroll:

```bash
cd angular-frontend
npm install @angular/cdk
```

```typescript
// paa-dashboard.component.ts
import { ScrollingModule } from '@angular/cdk/scrolling';

@Component({
  // ...
  imports: [CommonModule, FormsModule, ScrollingModule]
})
```

```html
<!-- paa-dashboard.component.html -->
<cdk-virtual-scroll-viewport itemSize="60" class="groups-viewport">
  <div *cdkVirtualFor="let group of groups" 
       (click)="selectGroup(group)"
       class="group-item">
    {{ group.group_id }}
  </div>
</cdk-virtual-scroll-viewport>
```

#### C. OnPush Change Detection
Optimiser la détection de changements:

```typescript
import { ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-paa-dashboard',
  changeDetection: ChangeDetectionStrategy.OnPush,  // ← Ajouter ici
  // ...
})
```

#### D. Debounce sur les Requêtes
Éviter les appels API multiples:

```typescript
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';

searchGroups(searchTerm: string): void {
  of(searchTerm).pipe(
    debounceTime(300),           // Attendre 300ms
    distinctUntilChanged(),       // Seulement si changé
    switchMap(term => this.apiService.searchGroups(term))
  ).subscribe(results => {
    this.groups = results;
  });
}
```

---

### 3. Optimisation MySQL

#### A. Configuration MySQL (my.ini)
```ini
[mysqld]
# Buffer pool (utiliser 50-70% de la RAM disponible)
innodb_buffer_pool_size = 1G

# Connexions
max_connections = 200

# Query cache
query_cache_type = 1
query_cache_size = 64M

# Index
key_buffer_size = 256M
```

#### B. Analyser les Requêtes Lentes
```sql
-- Activer le log des requêtes lentes
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;  -- Requêtes > 1 seconde

-- Voir les requêtes lentes
SELECT * FROM mysql.slow_log;
```

#### C. Optimiser les Tables
```sql
-- Optimiser périodiquement
OPTIMIZE TABLE paa_groups;
OPTIMIZE TABLE paa_contracts;
OPTIMIZE TABLE paa_movements;
OPTIMIZE TABLE paa_snapshots;

-- Analyser les statistiques
ANALYZE TABLE paa_groups;
```

---

### 4. Optimisation Network

#### A. Compression HTTP (Backend)
```python
# backend/main.py
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

#### B. HTTP/2 (Nginx en production)
```nginx
server {
    listen 443 ssl http2;
    # ...
}
```

---

## 🎯 Solution Rapide (Immédiate)

### Modification Simple - Backend

Modifier `backend/routers/paa_router.py`:

```python
@router.get("/groups")
def list_groups(
    limit: int = 50,  # Limiter à 50 par défaut
    db: Session = Depends(get_db)
):
    persistence = PAAPersistence(db)
    all_groups = persistence.list_groups()
    
    # Retourner seulement les N premiers
    limited_groups = all_groups[:limit]
    
    return {
        "groups": limited_groups,
        "total": len(all_groups),
        "showing": len(limited_groups)
    }
```

### Modification Simple - Frontend

Modifier `angular-frontend/src/app/components/paa-dashboard/paa-dashboard.component.ts`:

```typescript
loadGroups(): void {
  this.loadingGroups = true;
  
  // Ajouter un timeout pour éviter le blocage UI
  setTimeout(() => {
    this.apiService.listPAAGroups()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (res) => {
          this.groups = res.groups || [];
          this.loadingGroups = false;
        },
        error: (err) => {
          console.error('Erreur chargement groupes:', err);
          this.loadingGroups = false;
        }
      });
  }, 0);
}
```

---

## 🔧 Test de Performance

### Backend
```bash
# Tester temps de réponse API
curl -w "@curl-format.txt" -o /dev/null -s http://127.0.0.1:8001/paa/groups
```

Créer `curl-format.txt`:
```
time_namelookup:  %{time_namelookup}s\n
time_connect:     %{time_connect}s\n
time_appconnect:  %{time_appconnect}s\n
time_pretransfer: %{time_pretransfer}s\n
time_redirect:    %{time_redirect}s\n
time_starttransfer: %{time_starttransfer}s\n
----------\n
time_total:       %{time_total}s\n
```

### Frontend
```typescript
// Mesurer temps de chargement
console.time('loadGroups');
this.apiService.listPAAGroups().subscribe(res => {
  console.timeEnd('loadGroups');  // Affiche le temps écoulé
});
```

---

## 📊 Métriques Cibles

| Métrique | Avant | Cible | Amélioration |
|----------|-------|-------|--------------|
| Chargement dashboard | 5-10s | <2s | 80% |
| Requête API /groups | 2-3s | <500ms | 83% |
| Render liste groupes | 1-2s | <300ms | 85% |
| Sélection groupe | 1s | <200ms | 80% |

---

## 🎓 Meilleures Pratiques

### Backend
1. ✅ Toujours paginer les listes longues
2. ✅ Utiliser des index sur les colonnes fréquemment requêtées
3. ✅ Limiter les JOIN complexes
4. ✅ Cache pour données rarement modifiées
5. ✅ Compression HTTP (GZip)

### Frontend
1. ✅ OnPush change detection pour composants
2. ✅ Virtual scrolling pour listes >100 items
3. ✅ Lazy loading des modules
4. ✅ Debounce sur recherche/filtres
5. ✅ Afficher loader pendant chargement

### Base de Données
1. ✅ Index sur clés étrangères
2. ✅ OPTIMIZE TABLE régulièrement
3. ✅ Analyser requêtes lentes
4. ✅ Configurer buffer pool MySQL
5. ✅ Utiliser EXPLAIN sur requêtes complexes

---

## 🐛 Débogage Performance

### 1. Chrome DevTools
```
F12 → Network → XHR
Regarder temps de chargement API
```

### 2. Angular DevTools
```bash
ng serve --source-map
# Chrome: F12 → Angular → Profiler
```

### 3. Backend Profiling
```python
import time

@router.get("/groups")
def list_groups(db: Session = Depends(get_db)):
    start = time.time()
    
    # Votre code ici
    
    elapsed = time.time() - start
    print(f"⏱️ list_groups took {elapsed:.2f}s")
```

---

## ✅ Checklist Optimisation

### Immédiat (5 min)
- [ ] Limiter liste groupes à 50 éléments
- [ ] Ajouter loading spinner visible
- [ ] Vérifier que MySQL est démarré
- [ ] Redémarrer backend/frontend

### Court Terme (30 min)
- [ ] Pagination API backend
- [ ] OnPush change detection frontend
- [ ] Index base de données
- [ ] Compression GZip

### Moyen Terme (2h)
- [ ] Virtual scrolling listes
- [ ] Cache Redis
- [ ] Lazy loading modules Angular
- [ ] Optimiser requêtes SQL

### Long Terme (1 jour)
- [ ] CDN pour assets statiques
- [ ] Service Worker (PWA)
- [ ] SSR (Server-Side Rendering)
- [ ] WebSocket pour updates temps réel

---

## 📞 Support

Si le dashboard reste lent après ces optimisations:

1. Vérifier logs backend: `backend/main.py` (console)
2. Vérifier console navigateur: F12 → Console
3. Tester API directement: http://127.0.0.1:8001/docs
4. Vérifier MySQL: `SHOW PROCESSLIST;`

---

**Date**: 7 Octobre 2025  
**Version**: 1.0  
**Auteur**: Abdouli Aziz
