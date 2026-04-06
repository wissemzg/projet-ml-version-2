"""
User Management System for BVMT Trading Assistant.
Handles user profiles, authentication, preferences, and portfolio history.
Stores data in JSON files for simplicity.
"""
import json
import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "users"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _hash_password(password: str, salt: str = "") -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


class UserProfile:
    """Represents a user profile with investment preferences."""

    def __init__(self, user_id: str, username: str, data: dict = None):
        self.user_id = user_id
        self.username = username
        d = data or {}
        self.display_name = d.get('display_name', username)
        self.password_hash = d.get('password_hash', '')
        self.salt = d.get('salt', '')
        self.created_at = d.get('created_at', datetime.now().isoformat())
        self.last_login = d.get('last_login', '')

        # Investment profile (filled by chatbot onboarding)
        self.profile_completed = d.get('profile_completed', False)
        self.investment_experience = d.get('investment_experience', '')       # beginner/intermediate/expert
        self.risk_tolerance = d.get('risk_tolerance', 'moderate')             # conservative/moderate/aggressive
        self.investment_horizon = d.get('investment_horizon', '')             # short/medium/long
        self.monthly_budget = d.get('monthly_budget', 0)                     # TND per month
        self.total_capital = d.get('total_capital', 5000)                     # TND
        self.preferred_sectors = d.get('preferred_sectors', [])               # e.g. ['bancaire','industriel']
        self.investment_goals = d.get('investment_goals', '')                 # e.g. 'revenu passif'
        self.age_range = d.get('age_range', '')                              # e.g. '25-35'
        self.notes = d.get('notes', '')

        # RL & feedback
        self.feedback_history = d.get('feedback_history', [])                # list of {date, action, liked, stock}
        self.portfolio_id = d.get('portfolio_id', '')
        self.portfolio_history = d.get('portfolio_history', [])

    def to_dict(self) -> dict:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'display_name': self.display_name,
            'password_hash': self.password_hash,
            'salt': self.salt,
            'created_at': self.created_at,
            'last_login': self.last_login,
            'profile_completed': self.profile_completed,
            'investment_experience': self.investment_experience,
            'risk_tolerance': self.risk_tolerance,
            'investment_horizon': self.investment_horizon,
            'monthly_budget': self.monthly_budget,
            'total_capital': self.total_capital,
            'preferred_sectors': self.preferred_sectors,
            'investment_goals': self.investment_goals,
            'age_range': self.age_range,
            'notes': self.notes,
            'feedback_history': self.feedback_history,
            'portfolio_id': self.portfolio_id,
            'portfolio_history': self.portfolio_history,
        }

    def get_profile_summary(self) -> str:
        """Return a text summary for chatbot context."""
        if not self.profile_completed:
            return "Profil investisseur non complété."
        return (
            f"Investisseur: {self.display_name}\n"
            f"Expérience: {self.investment_experience}\n"
            f"Tolérance au risque: {self.risk_tolerance}\n"
            f"Horizon: {self.investment_horizon}\n"
            f"Capital: {self.total_capital} TND\n"
            f"Budget mensuel: {self.monthly_budget} TND\n"
            f"Secteurs préférés: {', '.join(self.preferred_sectors) if self.preferred_sectors else 'aucun'}\n"
            f"Objectifs: {self.investment_goals}\n"
            f"Tranche d'âge: {self.age_range}"
        )


class UserManager:
    """Manages user accounts and profiles."""

    def __init__(self):
        self._users: Dict[str, UserProfile] = {}
        self._load_all()

    def _user_file(self, user_id: str) -> Path:
        return DATA_DIR / f"{user_id}.json"

    def _load_all(self):
        for f in DATA_DIR.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                uid = data.get('user_id', f.stem)
                self._users[uid] = UserProfile(uid, data.get('username', uid), data)
            except Exception:
                pass

    def _save(self, user: UserProfile):
        with open(self._user_file(user.user_id), 'w', encoding='utf-8') as f:
            json.dump(user.to_dict(), f, ensure_ascii=False, indent=2)

    def register(self, username: str, password: str, display_name: str = "") -> dict:
        """Register a new user."""
        # Check if username exists
        for u in self._users.values():
            if u.username.lower() == username.lower():
                return {'success': False, 'error': 'Ce nom d\'utilisateur existe déjà.'}

        user_id = f"u_{uuid.uuid4().hex[:12]}"
        salt = uuid.uuid4().hex[:16]
        pw_hash = _hash_password(password, salt)

        user = UserProfile(user_id, username, {
            'display_name': display_name or username,
            'password_hash': pw_hash,
            'salt': salt,
            'created_at': datetime.now().isoformat(),
        })
        self._users[user_id] = user
        self._save(user)
        return {'success': True, 'user_id': user_id, 'username': username}

    def login(self, username: str, password: str) -> dict:
        """Authenticate a user."""
        for u in self._users.values():
            if u.username.lower() == username.lower():
                pw_hash = _hash_password(password, u.salt)
                if pw_hash == u.password_hash:
                    u.last_login = datetime.now().isoformat()
                    self._save(u)
                    return {'success': True, 'user_id': u.user_id, 'username': u.username,
                            'display_name': u.display_name, 'profile_completed': u.profile_completed}
                else:
                    return {'success': False, 'error': 'Mot de passe incorrect.'}
        return {'success': False, 'error': 'Utilisateur non trouvé.'}

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        return self._users.get(user_id)

    def update_profile(self, user_id: str, updates: dict) -> dict:
        """Update user investment profile."""
        user = self._users.get(user_id)
        if not user:
            return {'success': False, 'error': 'Utilisateur non trouvé.'}

        for key in ['investment_experience', 'risk_tolerance', 'investment_horizon',
                     'monthly_budget', 'total_capital', 'preferred_sectors',
                     'investment_goals', 'age_range', 'notes', 'display_name',
                     'profile_completed', 'portfolio_id']:
            if key in updates:
                setattr(user, key, updates[key])

        self._save(user)
        return {'success': True, 'profile': user.to_dict()}

    def add_feedback(self, user_id: str, feedback: dict) -> dict:
        """Add feedback for RL learning."""
        user = self._users.get(user_id)
        if not user:
            return {'success': False, 'error': 'Utilisateur non trouvé.'}

        feedback['date'] = datetime.now().isoformat()
        user.feedback_history.append(feedback)
        # Keep last 200 feedbacks
        if len(user.feedback_history) > 200:
            user.feedback_history = user.feedback_history[-200:]
        self._save(user)
        return {'success': True}

    def get_all_users_summary(self) -> list:
        """Get summary of all users (admin)."""
        return [{'user_id': u.user_id, 'username': u.username,
                 'display_name': u.display_name, 'profile_completed': u.profile_completed,
                 'last_login': u.last_login} for u in self._users.values()]
