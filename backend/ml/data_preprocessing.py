# ml/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Classe pour le préprocessing des données IFRS17 PAA
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyage initial des données
        """
        df_clean = df.copy()
        
        # Conversion des colonnes numériques importantes
        numeric_columns = ['DUREE', 'MNTPRNET', 'MNTPPNA', 'MNTACCESS', 'MNTPRASSI', 
                          'NBPPNATOT', 'NBPPNAJ', 'NUMQUITT', 'CODFAM', 'CODPROD']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Conversion des dates
        date_columns = ['DATECREA', 'DATEEMISS', 'DEBEFFQUI', 'FINEFFQUI', 'DATEPAIEM']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], format='%Y%m%d', errors='coerce')
        
        # Création de features temporelles
        if 'DEBEFFQUI' in df_clean.columns:
            df_clean['annee_effet'] = df_clean['DEBEFFQUI'].dt.year
            df_clean['mois_effet'] = df_clean['DEBEFFQUI'].dt.month
            df_clean['jour_semaine_effet'] = df_clean['DEBEFFQUI'].dt.dayofweek
            
        if 'DATEEMISS' in df_clean.columns:
            df_clean['annee_emission'] = df_clean['DATEEMISS'].dt.year
            df_clean['mois_emission'] = df_clean['DATEEMISS'].dt.month
            
        # Calcul de durées
        if 'DEBEFFQUI' in df_clean.columns and 'FINEFFQUI' in df_clean.columns:
            df_clean['duree_contrat_jours'] = (df_clean['FINEFFQUI'] - df_clean['DEBEFFQUI']).dt.days
            df_clean['duree_contrat_annees'] = df_clean['duree_contrat_jours'] / 365.25
            
        # Ratios financiers
        if 'MNTPRNET' in df_clean.columns and 'MNTPRASSI' in df_clean.columns:
            df_clean['ratio_prime_assure'] = df_clean['MNTPRNET'] / (df_clean['MNTPRASSI'] + 1e-8)
            
        if 'MNTPPNA' in df_clean.columns and 'NBPPNATOT' in df_clean.columns:
            df_clean['ppna_par_unite'] = df_clean['MNTPPNA'] / (df_clean['NBPPNATOT'] + 1e-8)
            
        # Indicateurs de risque
        if 'DUREE' in df_clean.columns:
            # Convertir DUREE en numérique en gérant les erreurs
            df_clean['DUREE'] = pd.to_numeric(df_clean['DUREE'], errors='coerce')
            df_clean['contrat_long_terme'] = (df_clean['DUREE'] > 12).astype(int)
        else:
            df_clean['contrat_long_terme'] = 0
            
        if 'MNTPRNET' in df_clean.columns:
            df_clean['prime_elevee'] = (df_clean['MNTPRNET'] > df_clean['MNTPRNET'].quantile(0.75)).astype(int)
        else:
            df_clean['prime_elevee'] = 0
        
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: list = None) -> pd.DataFrame:
        """
        Encodage des variables catégorielles
        """
        df_encoded = df.copy()
        
        if categorical_columns is None:
            categorical_columns = ['CODFAM', 'CODPROD', 'CODFORMU', 'TYPEEMMIS', 'STATQUIT', 'STADECTX', 'STATU']
            categorical_columns = [col for col in categorical_columns if col in df.columns]
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df_encoded[col + '_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                # Pour de nouvelles données
                try:
                    df_encoded[col + '_encoded'] = self.encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Gérer les nouvelles catégories
                    df_encoded[col + '_encoded'] = 0
                    
        return df_encoded
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Gestion des valeurs manquantes
        """
        df_imputed = df.copy()
        
        # Colonnes numériques
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy=strategy)
                df_imputed[[col]] = self.imputers[col].fit_transform(df[[col]])
            else:
                df_imputed[[col]] = self.imputers[col].transform(df[[col]])
                
        return df_imputed
    
    def scale_features(self, df: pd.DataFrame, features_to_scale: list = None, scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Normalisation des features
        """
        df_scaled = df.copy()
        
        if features_to_scale is None:
            features_to_scale = ['MNTPRNET', 'MNTACCESS', 'MNTPRASSI', 'NBPPNATOT', 'NBPPNAJ', 'MNTPPNA', 'DUREE']
            features_to_scale = [col for col in features_to_scale if col in df.columns]
        
        scaler_key = f"{scaler_type}_scaler"
        if scaler_key not in self.scalers:
            if scaler_type == 'standard':
                self.scalers[scaler_key] = StandardScaler()
            elif scaler_type == 'minmax':
                self.scalers[scaler_key] = MinMaxScaler()
                
            df_scaled[features_to_scale] = self.scalers[scaler_key].fit_transform(df[features_to_scale])
        else:
            df_scaled[features_to_scale] = self.scalers[scaler_key].transform(df[features_to_scale])
            
        return df_scaled
    
    def create_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Création de features spécifiques pour le ML
        """
        df_features = df.copy()
        
        # Features d'agrégation par produit
        if 'CODPROD' in df.columns and 'MNTPRNET' in df.columns:
            prod_stats = df.groupby('CODPROD')['MNTPRNET'].agg(['mean', 'std', 'count']).reset_index()
            prod_stats.columns = ['CODPROD', 'avg_prime_produit', 'std_prime_produit', 'nb_contrats_produit']
            df_features = df_features.merge(prod_stats, on='CODPROD', how='left')
            
        # Features d'agrégation par famille
        if 'CODFAM' in df.columns and 'MNTPRNET' in df.columns:
            fam_stats = df.groupby('CODFAM')['MNTPRNET'].agg(['mean', 'median']).reset_index()
            fam_stats.columns = ['CODFAM', 'avg_prime_famille', 'median_prime_famille']
            df_features = df_features.merge(fam_stats, on='CODFAM', how='left')
            
        # Features temporelles avancées
        if 'annee_effet' in df_features.columns:
            df_features['anciennete_produit'] = 2024 - df_features['annee_effet']
            
        # Features de saisonnalité
        if 'mois_effet' in df_features.columns:
            df_features['saison'] = df_features['mois_effet'].map({
                12: 'hiver', 1: 'hiver', 2: 'hiver',
                3: 'printemps', 4: 'printemps', 5: 'printemps',
                6: 'ete', 7: 'ete', 8: 'ete',
                9: 'automne', 10: 'automne', 11: 'automne'
            })
            
        return df_features
    
    def prepare_data_for_training(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Préparation complète des données pour l'entraînement
        """
        logger.info("Début du preprocessing des données...")
        
        # 1. Nettoyage
        df_clean = self.clean_data(df)
        logger.info("✅ Nettoyage terminé")
        
        # 2. Création de features
        df_features = self.create_features_for_ml(df_clean)
        logger.info("✅ Création de features terminée")
        
        # 3. Encodage catégoriel
        df_encoded = self.encode_categorical_features(df_features)
        logger.info("✅ Encodage catégoriel terminé")
        
        # 4. Gestion des valeurs manquantes
        df_imputed = self.handle_missing_values(df_encoded)
        logger.info("✅ Gestion des valeurs manquantes terminée")
        
        # 5. Sélection des features pour ML
        ml_features = self._select_ml_features(df_imputed)
        
        # 6. Normalisation
        df_final = self.scale_features(ml_features)
        logger.info("✅ Normalisation terminée")
        
        # 7. Préparation du target si spécifié
        y = None
        if target_column and target_column in df_final.columns:
            y = df_final[target_column]
            df_final = df_final.drop(columns=[target_column])
        elif target_column:
            logger.warning(f"⚠️ Colonne target '{target_column}' introuvable dans les données")
            
        # 8. Vérifications finales
        if df_final.empty:
            raise ValueError("❌ Le DataFrame final est vide après preprocessing")
        if y is not None and len(y) == 0:
            raise ValueError("❌ La série target est vide")
        if y is not None and len(df_final) != len(y):
            raise ValueError(f"❌ Tailles incompatibles: features={len(df_final)}, target={len(y)}")
            
        logger.info(f"✅ Preprocessing terminé. Shape finale: {df_final.shape}")
        if y is not None:
            logger.info(f"✅ Target shape: {len(y)}, valeurs uniques: {y.nunique()}")
            
        return df_final, y
    
    def _select_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sélection des features pertinentes pour le ML
        """
        # Features numériques de base
        numeric_features = [
            'MNTPRNET', 'MNTACCESS', 'MNTPRASSI', 'NBPPNATOT', 'NBPPNAJ', 'MNTPPNA',
            'DUREE', 'FRACT', 'annee_effet', 'mois_effet', 'jour_semaine_effet',
            'duree_contrat_jours', 'ratio_prime_assure', 'ppna_par_unite',
            'contrat_long_terme', 'prime_elevee', 'anciennete_produit'
        ]
        
        # Features encodées
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]
        
        # Features d'agrégation
        agg_features = [
            'avg_prime_produit', 'std_prime_produit', 'nb_contrats_produit',
            'avg_prime_famille', 'median_prime_famille'
        ]
        
        # Sélectionner les features disponibles
        available_features = []
        for feature_list in [numeric_features, encoded_features, agg_features]:
            available_features.extend([f for f in feature_list if f in df.columns])
        
        return df[available_features]