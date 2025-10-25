# 📘 README — Aspect théorique IFRS 17 (PAA)

Ce document synthétise la théorie IFRS 17 utile au projet, aligne les formules avec l’implémentation actuelle (approche PAA) et renvoie vers les pièces complètes pour approfondir.

---

## 1) Étude de l’existant (contexte BNA)

- Processus initial: calculs IFRS 17 réalisés sous Excel/exports, consolidation manuelle, délais de 3–4 h/portefeuille, risque d’erreurs et faible traçabilité.
- Objectif du projet: automatiser PAA, fiabiliser les calculs (PPNA, RA, LC, LRC), exposer des APIs et une UI temps réel, ajouter IA/ML pour support décisionnel et contrôle qualité.
- Cible: application full‑stack (Angular + FastAPI + MySQL), traçabilité/audit, exports et tableaux de bord, assistant IA et modèles ML.

Références utiles: docs/PROJECT_CLEAN_SUMMARY.md, docs/TRANSFORMATION_PAA_COMPLETE.md, docs/ACTUARIAL_VALIDATION_REPORT.md.

---

## 2) Panorama IFRS 17 (GMM • PAA • VFA)

- GMM (modèle général): Passif = BEL + RA + CSM.
- PAA (approche simplifiée, ≤ 12 mois ou approximation raisonnable du GMM): LRC = PPNA + RA + LC.
- VFA: contrats à participation aux bénéfices (non traité dans ce projet).

Critères PAA (IFRS 17.53): durée de couverture ≤ 1 an, ou PAA ≈ GMM (écart toléré faible).

---

## 3) PAA — définitions et formules alignées au projet

- PPNA (provisions pour primes non acquises)
  - Prorata temporis (contrat): PPNA = Prime écrite × (jours restants / jours de couverture)
  - Portefeuille: PPNA_portefeuille = Σ PPNA_i

- RA (Risk Adjustment) — méthode implémentée (Cost of Capital simplifiée)
  - RA = PPNA × σ × CoC × CL
  - Paramètres par défaut: σ = 8% (volatilité), CoC = 6% (coût du capital), CL ≈ 2.0 (niveau de confiance ~95%)
  - Taux RA ≈ 0.5% – 3% des primes selon portefeuille
  - Alternatives (documentées mais non activées par défaut): Percentile (VaR), diversification (√·), etc.

- LC (Loss Component) — test d’onérosité
  - Coûts futurs estimés = PPNA × (S/P) + PPNA × (F/P)
  - LC = max(0, Coûts futurs estimés − PPNA − RA)
  - Contrat onéreux si Coûts futurs > PPNA + RA (IFRS 17 §47–52)

- LRC Total
  - LRC = PPNA + RA + LC

Formules détaillées et exemples chiffrés: docs/ASPECT_THEORIQUE_PROJET.md (sections 4.1–4.4).

---

## 4) KPI et interprétation

- Loss ratio (LR) = Sinistres / Primes × 100%
- Expense ratio (ER) = Frais / Primes × 100%
- Combined ratio (CR) = LR + ER
- Lecture IFRS17 (vue passif): CR_IFRS17 ≈ LRC / Primes × 100%
  - < 100%: portefeuille profitable
  - 100–105%: zone de vigilance
  - > 105%: sous‑tarification potentielle

Recommandation UI: jauge CR et alerte « contrats onéreux ». Voir docs/EXECUTIVE_SUMMARY_ACTUARIAL.md et docs/VISUALIZATION_ACTUARIAL_REVIEW.md.

---

## 5) Projection et actualisation (rappel théorique)

- Actualisation (si > 12 mois ou exigé): VAN = Σ CF_t / (1 + r_t)^t
- Méthodes de projection sinistres: Chain‑Ladder, Bornhuetter‑Ferguson
- Provisionnement: PSP, IBNR

Note: pour PAA ≤ 12 mois, l’actualisation est souvent non exigée; le projet conserve l’option d’extension.

Détails: docs/ASPECT_THEORIQUE_PROJET.md (sections 4.6–4.8, 5).

---

## 6) Alignement avec l’implémentation

- Backend FastAPI expose PPNA, RA, LRC et le test d’onérosité; UI affiche LRC et métriques; ML prévoit LRC et risques.
- RA: méthode CoC simplifiée avec paramètres par défaut (σ=8%, CoC=6%, CL≈2.0) — modifiables par segment produit si besoin.
- Contrats onéreux: LC calculé et exposé; prévoir affichage/exports dans le dashboard.

Sources techniques: backend/measurement/paa, backend/routers/paa_router.py, docs/PAA_MODULE_README.md.

---

## 7) Validation actuarielle et seuils

- Seuils de qualité (observés sur données projet):
  - R² LRC ML ≈ 0.93+, RA ~ 0.5–3% des primes, CR < 100% en moyenne.
- Vérifications: réconciliation PPNA vs. primes acquises, stabilité RA, non‑détection excessive d’onéreux.

Voir: docs/ACTUARIAL_VALIDATION_REPORT.md, docs/EXECUTIVE_SUMMARY_ACTUARIAL.md.

---

## 8) Références et approfondissements

- Dossier complet: docs/ASPECT_THEORIQUE_PROJET.md (norme, math, exemples, ML)
- Module PAA: docs/PAA_MODULE_README.md, docs/TRANSFORMATION_PAA_COMPLETE.md
- Démarrage rapide PAA: docs/QUICK_START_PAA.md
- Scénario: README_SCENARIO_IFRS17.md
- ML: README_ML.md

---

## 9) Annexe — aide‑mémoire formules

- PPNA = PE × (n − t) / n
- RA = PPNA × σ × CoC × CL
- Coûts futurs = PPNA × (S/P + F/P)
- LC = max(0, Coûts futurs − PPNA − RA)
- LRC = PPNA + RA + LC
- CR ≈ LRC / Primes × 100%

Ces formules sont cohérentes avec l’implémentation actuelle et validées par les documents actuariels joints.
