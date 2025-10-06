"""
Script simple pour initialiser MySQL XAMPP pour IFRS17
"""
import pymysql
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Charger les variables d'environnement
load_dotenv()

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    full_name = Column(String(200), nullable=True)
    role = Column(String(50), default="analyste_ifrs17")
    company = Column(String(100), default="BNA")
    department = Column(String(100), nullable=True)
    employee_id = Column(String(50), nullable=True)
    phone = Column(String(20), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    level = Column(String(50), default="Débutant")
    points = Column(Integer, default=0)
    login_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

def main():
    print("🚀 Configuration MySQL pour XAMPP - IFRS17")
    print("=" * 50)
    
    # Configuration MySQL XAMPP
    host = "localhost"
    port = 3306
    user = "root"
    password = ""
    database_name = "ifrs17_auth"
    
    try:
        # Étape 1: Créer la base de données
        print("📝 Étape 1: Création de la base de données...")
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ Base de données '{database_name}' créée")
            
        connection.close()
        
        # Étape 2: Créer les tables
        print("📝 Étape 2: Création des tables...")
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database_name}")
        Base.metadata.create_all(bind=engine)
        print("✅ Tables créées avec succès!")
        
        # Étape 3: Vérification
        print("📝 Étape 3: Vérification...")
        with engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            print(f"📋 Tables disponibles: {tables}")
            
        print("\n🎉 Configuration terminée!")
        print("💡 Votre application peut maintenant utiliser MySQL XAMPP")
        print(f"🔗 Base de données: {database_name}")
        print("📊 Tables: users")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()