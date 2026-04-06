"""
Reinforcement Learning Portfolio Correction Module.
Uses a simple contextual bandit approach to learn user preferences
and adjust portfolio recommendations based on feedback.

States: user profile features + market conditions
Actions: stock allocations (weight adjustments)
Rewards: user feedback (liked/disliked, profit/loss)
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "rl"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class PortfolioRL:
    """
    Contextual bandit-based portfolio optimization.
    Learns from user feedback to adjust stock allocation weights.
    
    Features:
    - Tracks which stocks user liked/disliked
    - Adjusts confidence scores of recommendations
    - Learns risk tolerance from actual behavior
    - Adapts sector preferences from feedback
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.model_file = DATA_DIR / f"rl_{user_id}.json"
        self.learning_rate = 0.1
        self.exploration_rate = 0.15
        self.discount = 0.95

        # Load or initialize model
        self.model = self._load_model()

    def _load_model(self) -> dict:
        if self.model_file.exists():
            try:
                with open(self.model_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'stock_scores': {},      # stock_code -> adjustment score [-1, 1]
            'sector_scores': {},     # sector -> preference weight
            'risk_learned': 0.0,     # learned risk adjustment
            'total_feedback': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'history': [],           # recent actions
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat(),
        }

    def _save_model(self):
        self.model['updated'] = datetime.now().isoformat()
        with open(self.model_file, 'w', encoding='utf-8') as f:
            json.dump(self.model, f, ensure_ascii=False, indent=2)

    def record_feedback(self, stock_code: str, stock_name: str, action: str,
                        liked: bool, profit_pct: float = 0.0,
                        sector: str = "", reason: str = "") -> dict:
        """
        Record user feedback on a recommendation.
        
        Args:
            stock_code: Stock ticker
            stock_name: Human-readable name
            action: 'buy'/'sell'/'hold' that was recommended
            liked: Whether user agreed with recommendation
            profit_pct: Actual profit/loss percentage (if known)
            sector: Stock sector
            reason: User's reason for feedback
        """
        reward = self._compute_reward(liked, profit_pct)

        # Update stock score
        scores = self.model['stock_scores']
        old_score = scores.get(stock_code, 0.0)
        scores[stock_code] = old_score + self.learning_rate * (reward - old_score)
        scores[stock_code] = max(-1.0, min(1.0, scores[stock_code]))

        # Update sector preference
        if sector:
            sec_scores = self.model['sector_scores']
            old_sec = sec_scores.get(sector, 0.0)
            sec_scores[sector] = old_sec + self.learning_rate * (reward - old_sec)
            sec_scores[sector] = max(-1.0, min(1.0, sec_scores[sector]))

        # Update risk from behavior
        if action == 'buy' and liked:
            self.model['risk_learned'] += self.learning_rate * 0.1
        elif action == 'buy' and not liked:
            self.model['risk_learned'] -= self.learning_rate * 0.1
        self.model['risk_learned'] = max(-0.5, min(0.5, self.model['risk_learned']))

        # Statistics
        self.model['total_feedback'] += 1
        if liked:
            self.model['positive_feedback'] += 1
        else:
            self.model['negative_feedback'] += 1

        # History (keep last 100)
        self.model['history'].append({
            'date': datetime.now().isoformat(),
            'stock_code': stock_code, 'stock_name': stock_name,
            'action': action, 'liked': liked,
            'profit_pct': profit_pct, 'reward': reward,
            'sector': sector, 'reason': reason
        })
        if len(self.model['history']) > 100:
            self.model['history'] = self.model['history'][-100:]

        self._save_model()

        return {
            'success': True,
            'stock_score': scores[stock_code],
            'total_feedback': self.model['total_feedback'],
            'accuracy': self.model['positive_feedback'] / max(1, self.model['total_feedback'])
        }

    def _compute_reward(self, liked: bool, profit_pct: float) -> float:
        """Compute reward signal from feedback."""
        base = 1.0 if liked else -1.0
        if profit_pct != 0:
            profit_signal = np.tanh(profit_pct / 10.0)  # Normalize profit
            return 0.6 * base + 0.4 * profit_signal
        return base

    def adjust_allocations(self, allocations: list, user_profile: dict = None) -> list:
        """
        Adjust portfolio allocations based on learned preferences.
        
        Args:
            allocations: List of dicts with stock_code, weight, confidence, etc.
            user_profile: User investment profile for context
            
        Returns:
            Adjusted allocations with modified weights
        """
        if not allocations:
            return allocations

        scores = self.model['stock_scores']
        sec_scores = self.model['sector_scores']
        risk_adj = self.model['risk_learned']

        adjusted = []
        total_weight = 0.0

        for alloc in allocations:
            a = dict(alloc)  # Copy
            code = a.get('stock_code', '')
            sector = a.get('sector', '')

            # Stock-specific adjustment
            stock_adj = scores.get(code, 0.0)
            # Sector adjustment
            sector_adj = sec_scores.get(sector, 0.0) if sector else 0.0

            # Combine adjustments
            combined_adj = 0.5 * stock_adj + 0.3 * sector_adj + 0.2 * risk_adj

            # Adjust weight (multiplicative)
            old_weight = a.get('allocation_pct', a.get('weight', 0))
            new_weight = old_weight * (1.0 + combined_adj * 0.3)
            new_weight = max(5.0, new_weight)  # Minimum 5%

            # Adjust confidence
            old_conf = a.get('confidence', 0.5)
            a['confidence'] = min(1.0, max(0.1, old_conf + stock_adj * 0.1))

            a['allocation_pct'] = new_weight
            a['weight'] = new_weight
            a['rl_adjustment'] = round(combined_adj, 3)
            a['rl_stock_score'] = round(stock_adj, 3)

            # Add exploration: occasionally boost low-scored stocks
            if np.random.random() < self.exploration_rate and stock_adj < 0:
                a['allocation_pct'] *= 1.1
                a['rl_explored'] = True

            adjusted.append(a)
            total_weight += a['allocation_pct']

        # Normalize weights to sum to 100%
        if total_weight > 0:
            for a in adjusted:
                a['allocation_pct'] = round(a['allocation_pct'] / total_weight * 100, 1)
                a['weight'] = a['allocation_pct']

        return adjusted

    def get_learned_risk_profile(self, base_profile: str = 'moderate') -> str:
        """Get RL-adjusted risk profile."""
        risk_adj = self.model['risk_learned']
        profiles = ['conservative', 'moderate', 'aggressive']
        base_idx = profiles.index(base_profile) if base_profile in profiles else 1

        # Shift based on learned preference
        if risk_adj > 0.2:
            adjusted_idx = min(2, base_idx + 1)
        elif risk_adj < -0.2:
            adjusted_idx = max(0, base_idx - 1)
        else:
            adjusted_idx = base_idx

        return profiles[adjusted_idx]

    def get_model_summary(self) -> dict:
        """Get summary of learned model for display."""
        m = self.model
        top_stocks = sorted(m['stock_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
        bottom_stocks = sorted(m['stock_scores'].items(), key=lambda x: x[1])[:5]
        top_sectors = sorted(m['sector_scores'].items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            'total_feedback': m['total_feedback'],
            'positive_rate': round(m['positive_feedback'] / max(1, m['total_feedback']) * 100, 1),
            'risk_adjustment': round(m['risk_learned'], 3),
            'top_stocks': [{'code': c, 'score': round(s, 3)} for c, s in top_stocks],
            'bottom_stocks': [{'code': c, 'score': round(s, 3)} for c, s in bottom_stocks],
            'top_sectors': [{'sector': s, 'score': round(v, 3)} for s, v in top_sectors],
            'recent_feedback': m['history'][-5:] if m['history'] else [],
        }
