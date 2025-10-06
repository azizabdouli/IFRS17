"""
Database Schema Creation Script
Exécutez ce script pour créer les tables nécessaires dans votre base de données PostgreSQL ou MySQL
"""

from backend.database.connection import engine, Base, get_database_url
from backend.database.models import User, UserSession, AuditLog

def create_tables():
    """Créer toutes les tables dans la base de données"""
    print("Création des tables de base de données...")
    print(f"URL de connexion: {get_database_url()}")
    
    try:
        # Créer toutes les tables définies dans les modèles
        Base.metadata.create_all(bind=engine)
        print("✅ Tables créées avec succès!")
        
        # Afficher les tables créées
        print("\nTables créées:")
        for table_name in Base.metadata.tables.keys():
            print(f"  - {table_name}")
            
    except Exception as e:
        print(f"❌ Erreur lors de la création des tables: {e}")
        raise

def drop_tables():
    """Supprimer toutes les tables (ATTENTION: Cela supprime toutes les données!)"""
    print("⚠️  ATTENTION: Suppression de toutes les tables!")
    confirm = input("Êtes-vous sûr? Tapez 'OUI' pour confirmer: ")
    
    if confirm == "OUI":
        try:
            Base.metadata.drop_all(bind=engine)
            print("✅ Tables supprimées avec succès!")
        except Exception as e:
            print(f"❌ Erreur lors de la suppression des tables: {e}")
            raise
    else:
        print("Opération annulée.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--drop":
        drop_tables()
    else:
        create_tables()