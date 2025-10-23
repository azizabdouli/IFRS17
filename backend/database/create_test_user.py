"""
Script pour créer un utilisateur de test IFRS17
Usage: python -m backend.database.create_test_user
"""

from backend.database.connection import SessionLocal
from backend.auth.user_service import UserService
from backend.database.schemas import UserCreate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_users():
    """Créer des utilisateurs de test pour l'application IFRS17"""
    db = SessionLocal()
    
    try:
        # Utilisateur 1: Admin/Actuaire Senior
        test_user_1 = UserCreate(
            email="admin@bna.tn",
            password="Admin123!",
            first_name="Ahmed",
            last_name="Ben Ali",
            company="BNA",
            department="Assurance IARD",
            phone="+216 71 123 456",
            employee_id="BNA001"
        )
        
        # Utilisateur 2: Analyste Junior
        test_user_2 = UserCreate(
            email="analyste@bna.tn",
            password="Analyste123!",
            first_name="Fatma",
            last_name="Mansour",
            company="BNA",
            department="Actuariat",
            phone="+216 71 234 567",
            employee_id="BNA002"
        )
        
        # Utilisateur 3: Test simple
        test_user_3 = UserCreate(
            email="test@test.com",
            password="test123",
            first_name="Test",
            last_name="User",
            company="BNA",
            department="IT",
            phone="+216 71 345 678",
            employee_id="BNA003"
        )
        
        users_to_create = [
            (test_user_1, "admin@bna.tn"),
            (test_user_2, "analyste@bna.tn"),
            (test_user_3, "test@test.com")
        ]
        
        created_users = []
        
        for user_data, email in users_to_create:
            try:
                # Vérifier si l'utilisateur existe déjà
                existing_user = UserService.get_user_by_email(db, email)
                if existing_user:
                    logger.info(f"✓ Utilisateur {email} existe déjà (ID: {existing_user.id})")
                    created_users.append(existing_user)
                else:
                    # Créer le nouvel utilisateur
                    user = UserService.create_user(db, user_data)
                    logger.info(f"✓ Utilisateur créé: {email} (ID: {user.id})")
                    created_users.append(user)
                    
            except ValueError as e:
                logger.warning(f"⚠ {email}: {e}")
            except Exception as e:
                logger.error(f"✗ Erreur lors de la création de {email}: {e}")
        
        logger.info("\n" + "="*60)
        logger.info("UTILISATEURS DE TEST CRÉÉS/VÉRIFIÉS")
        logger.info("="*60)
        
        for user in created_users:
            logger.info(f"""
┌─────────────────────────────────────────────────────────┐
│ Email:        {user.email:<42} │
│ Nom:          {user.first_name} {user.last_name:<35} │
│ Rôle:         {user.role:<42} │
│ Département:  {user.department or 'N/A':<42} │
│ ID Employé:   {user.employee_id or 'N/A':<42} │
│ Status:       {'✓ Actif' if user.is_active else '✗ Inactif':<42} │
└─────────────────────────────────────────────────────────┘
            """)
        
        logger.info("\n" + "="*60)
        logger.info("INFORMATIONS DE CONNEXION")
        logger.info("="*60)
        logger.info("""
1. Administrateur/Actuaire Senior:
   Email:     admin@bna.tn
   Password:  Admin123!
   
2. Analyste Junior:
   Email:     analyste@bna.tn
   Password:  Analyste123!
   
3. Utilisateur Test:
   Email:     test@test.com
   Password:  test123
        """)
        logger.info("="*60)
        
        return created_users
        
    except Exception as e:
        logger.error(f"Erreur générale: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    logger.info("🚀 Création des utilisateurs de test IFRS17...")
    users = create_test_users()
    logger.info(f"✅ Terminé! {len(users)} utilisateur(s) disponible(s)")
