"""
Script pour créer la base de données IFRS17 et les tables dans XAMPP MySQL
"""
import pymysql
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from backend.database.connection import DATABASE_TYPE, get_database_url
from backend.database.models import Base

# Charger les variables d'environnement depuis .env
load_dotenv()

def create_database_if_not_exists():
    """Crée la base de données IFRS17 si elle n'existe pas"""
    
    # Configuration pour XAMPP (MySQL par défaut)
    host = os.getenv("DATABASE_HOST", "localhost")
    port = int(os.getenv("DATABASE_PORT", "3306"))
    user = os.getenv("DATABASE_USER", "root")
    password = os.getenv("DATABASE_PASSWORD", "")
    database_name = os.getenv("DATABASE_NAME", "ifrs17_auth")
    
    print(f"🔗 Connexion à MySQL XAMPP...")
    print(f"Host: {host}:{port}")
    print(f"User: {user}")
    print(f"Database: {database_name}")
    
    try:
        # Connexion à MySQL sans spécifier de base de données
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Créer la base de données si elle n'existe pas
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ Base de données '{database_name}' créée/vérifiée avec succès")
            
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la création de la base de données: {e}")
        return False

def create_tables():
    """Crée les tables dans la base de données"""
    try:
        # Obtenir l'URL de la base de données
        database_url = get_database_url()
        print(f"🔗 Connexion à: {database_url.replace(database_url.split('@')[0].split('//')[1], '***')}")
        
        # Créer l'engine avec la base de données spécifiée
        engine = create_engine(database_url, echo=True)
        
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        print("✅ Tables créées avec succès!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des tables: {e}")
        return False

def main():
    """Script principal"""
    print("🚀 Initialisation de la base de données IFRS17 pour XAMPP")
    print("=" * 60)
    
    # Étape 1: Créer la base de données
    if create_database_if_not_exists():
        print("✅ Étape 1: Base de données créée")
    else:
        print("❌ Échec de l'étape 1")
        return
    
    # Étape 2: Créer les tables
    if create_tables():
        print("✅ Étape 2: Tables créées")
        print("\n🎉 Initialisation terminée avec succès!")
        print("\n📋 Tables créées:")
        print("   - users (utilisateurs)")
        print("   - user_sessions (sessions utilisateur)")
        print("\n🔗 Vous pouvez maintenant utiliser l'application avec XAMPP!")
    else:
        print("❌ Échec de l'étape 2")

if __name__ == "__main__":
    main()