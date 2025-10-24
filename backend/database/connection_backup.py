import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Charger les variables d'environnement depuis .env
load_dotenv()

# Configuration de la base de données
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "sqlite")

if DATABASE_TYPE == "sqlite":
    # SQLite pour les tests et développement
    DATABASE_URL = "sqlite:///./ifrs17_auth.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
elif DATABASE_TYPE == "postgresql":
    # PostgreSQL pour la production
    DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
    DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "ifrs17_auth")
    DATABASE_USER = os.getenv("DATABASE_USER", "postgres")
    DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "password")
    
    DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    engine = create_engine(DATABASE_URL)
elif DATABASE_TYPE == "mysql":
    # MySQL/MariaDB avec XAMPP
    DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
    DATABASE_PORT = os.getenv("DATABASE_PORT", "3306")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "ifrs17_auth")
    DATABASE_USER = os.getenv("DATABASE_USER", "root")
    DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "")
    
    # Utilisation de PyMySQL comme driver avec gestion reconnexion
    DATABASE_URL = f"mysql+pymysql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    engine = create_engine(
        DATABASE_URL,
        echo=True,
        pool_pre_ping=True,  # Vérifie la connexion avant utilisation
        pool_recycle=3600,   # Recycle les connexions après 1h
        pool_size=10,        # Taille du pool de connexions
        max_overflow=20      # Connexions supplémentaires autorisées
    )
else:
    raise ValueError(f"Type de base de données non supporté: {DATABASE_TYPE}")

# SQLAlchemy
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()

# Dependency pour obtenir la session DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        raise
    finally:
        try:
            db.close()
        except Exception:
            # Ignorer les erreurs de fermeture si la connexion est déjà perdue
            pass

def get_database_url():
    """Retourne l'URL de la base de données configurée"""
    if DATABASE_TYPE == "sqlite":
        return "sqlite:///./ifrs17_auth.db"
    elif DATABASE_TYPE == "postgresql":
        DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
        DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
        DATABASE_NAME = os.getenv("DATABASE_NAME", "ifrs17_auth")
        DATABASE_USER = os.getenv("DATABASE_USER", "postgres")
        DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "password")
        return f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    elif DATABASE_TYPE == "mysql":
        DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
        DATABASE_PORT = os.getenv("DATABASE_PORT", "3306")
        DATABASE_NAME = os.getenv("DATABASE_NAME", "ifrs17_auth")
        DATABASE_USER = os.getenv("DATABASE_USER", "root")
        DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "")
        return f"mysql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    else:
        return DATABASE_URL

# Dependency pour obtenir la session DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()