// src/app/models/ifrs17.models.ts

/**
 * 📊 Modèles de données IFRS17 pour la comptabilité d'assurance
 * Définit les structures de données conformes à la norme IFRS17
 */

// =================================
// 🏢 CONTRAT D'ASSURANCE
// =================================

export interface ContratAssurance {
  id: string;
  numeroContrat: string;
  typeContrat: TypeContrat;
  statutContrat: StatutContrat;
  dateInception: Date;
  dateEcheance: Date;
  souscripteur: Souscripteur;
  
  // Données financières
  primeContractuelle: number;
  primeEmise: number;
  primeAcquise: number;
  
  // Métriques IFRS17
  lrc: number; // Liability for Remaining Coverage
  lic: number; // Liability for Incurred Claims
  csm: number; // Contractual Service Margin
  ra: number;  // Risk Adjustment
  
  // Indicateurs
  estOnereux: boolean;
  groupeCohorte: string;
  uniteGeneration: string;
}

export enum TypeContrat {
  VIE = 'VIE',
  NON_VIE = 'NON_VIE',
  SANTE = 'SANTE',
  RENTE = 'RENTE',
  MIXTE = 'MIXTE'
}

export enum StatutContrat {
  ACTIF = 'ACTIF',
  SUSPENDU = 'SUSPENDU',
  RESILIER = 'RESILIER',
  ECHU = 'ECHU'
}

// =================================
// 👤 SOUSCRIPTEUR
// =================================

export interface Souscripteur {
  id: string;
  nom: string;
  prenom: string;
  raisonSociale?: string;
  typePersonne: 'PHYSIQUE' | 'MORALE';
  dateNaissance?: Date;
  adresse: Adresse;
  categorieRisque: CategorieRisque;
}

export interface Adresse {
  rue: string;
  ville: string;
  codePostal: string;
  pays: string;
}

export enum CategorieRisque {
  FAIBLE = 'FAIBLE',
  MOYEN = 'MOYEN',
  ELEVE = 'ELEVE',
  TRES_ELEVE = 'TRES_ELEVE'
}

// =================================
// 💰 FLUX DE TRESORERIE
// =================================

export interface FluxTresorerie {
  id: string;
  contratId: string;
  periode: Date;
  typeFlux: TypeFlux;
  montant: number;
  devise: string;
  actualise: boolean;
  tauxActualisation?: number;
}

export enum TypeFlux {
  PRIME = 'PRIME',
  SINISTRE = 'SINISTRE',
  COMMISSION = 'COMMISSION',
  FRAIS_GESTION = 'FRAIS_GESTION',
  FRAIS_ACQUISITION = 'FRAIS_ACQUISITION'
}

// =================================
// 🎯 COHORTE ET GROUPEMENT
// =================================

export interface Cohorte {
  id: string;
  anneeOrigination: number;
  ligneMetier: LigneMetier;
  devisePresentation: string;
  nombreContrats: number;
  primesTotales: number;
  
  // Agrégats IFRS17
  lrcTotal: number;
  licTotal: number;
  csmTotal: number;
  raTotal: number;
  
  // Indicateurs de performance
  ratioSinistralite: number;
  ratioCommission: number;
  ratioFraisGestion: number;
  ratioSolvabilite: number;
}

export enum LigneMetier {
  AUTOMOBILE = 'AUTOMOBILE',
  HABITATION = 'HABITATION',
  RESPONSABILITE_CIVILE = 'RESPONSABILITE_CIVILE',
  VIE_INDIVIDUELLE = 'VIE_INDIVIDUELLE',
  VIE_GROUPE = 'VIE_GROUPE',
  SANTE_COMPLEMENTAIRE = 'SANTE_COMPLEMENTAIRE',
  PROTECTION_JURIDIQUE = 'PROTECTION_JURIDIQUE'
}

// =================================
// 🧮 CALCULS ACTUARIELS
// =================================

export interface CalculActuariel {
  id: string;
  contratId: string;
  dateCalcul: Date;
  methodeCalcul: MethodeActuarielle;
  
  // Hypothèses actuarielles
  tauxActualisation: number;
  tablesMortalite: string;
  hypothesesLapse: number;
  hypothesesFrais: number;
  
  // Résultats
  valeursActuellesPrimesNet: number;
  valeursActuellesPrestations: number;
  provisionMathematique: number;
  margeBeneficiaire: number;
}

export enum MethodeActuarielle {
  VAN = 'VAN', // Valeur Actuelle Nette
  BEL = 'BEL', // Best Estimate Liability
  GMM = 'GMM', // General Measurement Model
  PAA = 'PAA', // Premium Allocation Approach
  VFA = 'VFA'  // Variable Fee Approach
}

// =================================
// 📊 REPORTING IFRS17
// =================================

export interface RapportIFRS17 {
  id: string;
  periode: Date;
  typeRapport: TypeRapport;
  
  // États financiers
  bilanIFRS17: BilanIFRS17;
  compteResultat: CompteResultatIFRS17;
  tableauFluxTresorerie: TableauFluxTresorerie;
  
  // Annexes
  mouvementsLRC: MouvementLRC[];
  mouvementsLIC: MouvementLIC[];
  analyseSensibilite: AnalyseSensibilite;
  
  // Validation
  statut: StatutRapport;
  validePar?: string;
  dateValidation?: Date;
  commentaires?: string;
}

export enum TypeRapport {
  MENSUEL = 'MENSUEL',
  TRIMESTRIEL = 'TRIMESTRIEL',
  SEMESTRIEL = 'SEMESTRIEL',
  ANNUEL = 'ANNUEL'
}

export enum StatutRapport {
  BROUILLON = 'BROUILLON',
  EN_COURS = 'EN_COURS',
  VALIDE = 'VALIDE',
  PUBLIE = 'PUBLIE'
}

// =================================
// 🏦 BILAN IFRS17
// =================================

export interface BilanIFRS17 {
  // Actifs d'assurance
  actifsContrats: number;
  actifsReassurance: number;
  
  // Passifs d'assurance
  passifsContrats: number;
  lrcTotal: number;
  licTotal: number;
  
  // Capitaux propres
  capitauxPropres: number;
  resultatNet: number;
  autresElementsResultatGlobal: number;
}

export interface CompteResultatIFRS17 {
  // Revenus d'assurance
  revenus: number;
  revenusReassurance: number;
  
  // Charges d'assurance
  chargesAssurance: number;
  chargesAcquisition: number;
  
  // Résultat
  resultatTechnique: number;
  resultatFinancier: number;
  resultatNet: number;
}

export interface TableauFluxTresorerie {
  fluxActivitesOperationnelles: number;
  fluxActivitesInvestissement: number;
  fluxActivitesFinancement: number;
  variationTresorerie: number;
}

// =================================
// 🔄 MOUVEMENTS
// =================================

export interface MouvementLRC {
  periode: Date;
  soldeOuverture: number;
  nouveauxContrats: number;
  revenusContractuels: number;
  chargesIncurres: number;
  ajustementsFinanciers: number;
  soldeCloture: number;
}

export interface MouvementLIC {
  periode: Date;
  soldeOuverture: number;
  chargesIncurres: number;
  paiementsSinistres: number;
  ajustementsActuariels: number;
  soldeCloture: number;
}

// =================================
// 📈 ANALYSE DE SENSIBILITÉ
// =================================

export interface AnalyseSensibilite {
  scenarioBase: ScenarioActuariel;
  scenariosStress: ScenarioActuariel[];
  impactCapitauxPropres: number;
  impactResultat: number;
  indicateursRisque: IndicateurRisque[];
}

export interface ScenarioActuariel {
  nom: string;
  description: string;
  hypotheses: Map<string, number>;
  resultats: ResultatScenario;
}

export interface ResultatScenario {
  lrcTotal: number;
  licTotal: number;
  csmTotal: number;
  raTotal: number;
  ratioSolvabilite: number;
}

export interface IndicateurRisque {
  typeRisque: TypeRisque;
  valeur: number;
  seuil: number;
  statut: 'VERT' | 'ORANGE' | 'ROUGE';
}

export enum TypeRisque {
  SOUSCRIPTION = 'SOUSCRIPTION',
  MARCHE = 'MARCHE',
  CREDIT = 'CREDIT',
  OPERATIONNEL = 'OPERATIONNEL',
  LIQUIDITE = 'LIQUIDITE'
}

// =================================
// 🎛️ PARAMÉTRAGE
// =================================

export interface ParametrageIFRS17 {
  id: string;
  nom: string;
  version: string;
  dateApplication: Date;
  
  // Paramètres techniques
  tauxActualisationDefaut: number;
  seuilMaterialite: number;
  methodeMesure: MethodeActuarielle;
  frequenceRevision: FrequenceRevision;
  
  // Paramètres métier
  lignesMetierActives: LigneMetier[];
  devisesAutorizees: string[];
  territoiresGeographiques: string[];
  
  // Validation
  valide: boolean;
  validePar?: string;
  dateValidation?: Date;
}

export enum FrequenceRevision {
  MENSUELLE = 'MENSUELLE',
  TRIMESTRIELLE = 'TRIMESTRIELLE',
  SEMESTRIELLE = 'SEMESTRIELLE',
  ANNUELLE = 'ANNUELLE'
}

// =================================
// 🔄 ÉTATS DE TRANSITION
// =================================

export interface TransitionIFRS17 {
  dateTransition: Date;
  methodeTransition: MethodeTransition;
  ajustementsTransition: AjustementTransition[];
  impactCapitauxPropres: number;
  documentationTransition: string;
}

export enum MethodeTransition {
  RETROSPECTIVE_COMPLETE = 'RETROSPECTIVE_COMPLETE',
  RETROSPECTIVE_MODIFIEE = 'RETROSPECTIVE_MODIFIEE',
  JUSTE_VALEUR = 'JUSTE_VALEUR'
}

export interface AjustementTransition {
  compteComptable: string;
  libelle: string;
  montantAvant: number;
  montantApres: number;
  ecart: number;
  explication: string;
}