"""
Script pour recréer les tables MySQL avec le schéma complet
"""
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()

def recreate_tables():
    """Supprimer et recréer les tables avec le bon schéma"""
    try:
        host = "localhost"
        port = 3306
        user = "root"
        password = ""
        database_name = "ifrs17_auth"
        
        print("🔗 Connexion à MySQL...")
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database_name,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            print("🗑️ Suppression des anciennes tables...")
            cursor.execute("DROP TABLE IF EXISTS audit_logs")
            cursor.execute("DROP TABLE IF EXISTS user_sessions")
            cursor.execute("DROP TABLE IF EXISTS users")
            print("✅ Tables supprimées")
            
        connection.commit()
        connection.close()
        print("✅ Base de données nettoyée")
        print("\n📝 Redémarrez le backend pour créer les tables avec le bon schéma SQLAlchemy")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Nettoyage de la base de données MySQL")
    print("=" * 50)
    
    if recreate_tables():
        print("\n🎉 Terminé!")
        print("\n📌 Prochaine étape:")
        print("   Redémarrez le backend - les tables seront créées automatiquement")
    else:
        print("\n❌ Échec du nettoyage")
