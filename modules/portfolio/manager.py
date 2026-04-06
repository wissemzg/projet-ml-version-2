"""
Module 4 — Decision & Portfolio Management for BVMT.
Implements risk profiles, portfolio simulation, buy/sell/hold recommendations,
and mandatory explainability.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class RiskProfile:
    """User risk profile configuration."""
    CONSERVATIVE = 'conservative'
    MODERATE = 'moderate'
    AGGRESSIVE = 'aggressive'
    
    CONFIGS = {
        'conservative': {
            'max_allocation_pct': 15,
            'max_volatility': 2.0,
            'min_liquidity_score': 0.6,
            'preferred_sectors': ['banking', 'insurance'],
            'stop_loss_pct': 3,
            'take_profit_pct': 8,
            'description': 'Profil conservateur: priorité à la préservation du capital, faible exposition au risque'
        },
        'moderate': {
            'max_allocation_pct': 25,
            'max_volatility': 4.0,
            'min_liquidity_score': 0.4,
            'preferred_sectors': ['banking', 'industry', 'consumer'],
            'stop_loss_pct': 5,
            'take_profit_pct': 15,
            'description': 'Profil modéré: équilibre entre risque et rendement, diversification active'
        },
        'aggressive': {
            'max_allocation_pct': 40,
            'max_volatility': 8.0,
            'min_liquidity_score': 0.2,
            'preferred_sectors': ['all'],
            'stop_loss_pct': 10,
            'take_profit_pct': 30,
            'description': 'Profil agressif: recherche de rendement élevé, tolérance au risque importante'
        }
    }
    
    @classmethod
    def get_config(cls, profile: str) -> dict:
        return cls.CONFIGS.get(profile, cls.CONFIGS['moderate'])


class PortfolioSimulator:
    """Virtual portfolio simulator with performance metrics."""
    
    def __init__(self, initial_capital: float = 5000.0, risk_profile: str = 'moderate'):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.risk_profile = risk_profile
        self.config = RiskProfile.get_config(risk_profile)
        self.positions = {}  # {stock_code: {shares, avg_price, date_bought}}
        self.history = []
        self.trades = []
    
    def buy(self, stock_code: str, stock_name: str, price: float, 
            amount: float = None, shares: int = None, date: str = None) -> dict:
        """Execute buy order."""
        if shares is None and amount is not None:
            shares = int(amount / price)
        elif shares is None:
            # Auto-size based on risk profile
            max_spend = self.cash * (self.config['max_allocation_pct'] / 100)
            shares = int(max_spend / price)
        
        cost = shares * price
        if cost > self.cash or shares <= 0:
            return {"success": False, "reason": "Insufficient funds or invalid order"}
        
        self.cash -= cost
        
        if stock_code in self.positions:
            old = self.positions[stock_code]
            total_shares = old['shares'] + shares
            avg_price = (old['avg_price'] * old['shares'] + price * shares) / total_shares
            self.positions[stock_code]['shares'] = total_shares
            self.positions[stock_code]['avg_price'] = avg_price
        else:
            self.positions[stock_code] = {
                'shares': shares,
                'avg_price': price,
                'stock_name': stock_name,
                'date_bought': date or datetime.now().strftime('%Y-%m-%d')
            }
        
        trade = {
            'type': 'BUY', 'stock': stock_code, 'name': stock_name,
            'shares': shares, 'price': price, 'cost': cost,
            'date': date or datetime.now().strftime('%Y-%m-%d')
        }
        self.trades.append(trade)
        return {"success": True, **trade}
    
    def sell(self, stock_code: str, price: float, shares: int = None, date: str = None) -> dict:
        """Execute sell order."""
        if stock_code not in self.positions:
            return {"success": False, "reason": "No position in this stock"}
        
        pos = self.positions[stock_code]
        if shares is None:
            shares = pos['shares']
        
        shares = min(shares, pos['shares'])
        revenue = shares * price
        profit = (price - pos['avg_price']) * shares
        
        self.cash += revenue
        pos['shares'] -= shares
        
        if pos['shares'] <= 0:
            del self.positions[stock_code]
        
        trade = {
            'type': 'SELL', 'stock': stock_code, 'name': pos.get('stock_name', stock_code),
            'shares': shares, 'price': price, 'revenue': revenue, 'profit': round(profit, 2),
            'date': date or datetime.now().strftime('%Y-%m-%d')
        }
        self.trades.append(trade)
        return {"success": True, **trade}
    
    def get_portfolio_value(self, current_prices: dict) -> dict:
        """Calculate current portfolio value and metrics."""
        positions_value = 0
        position_details = []
        
        for code, pos in self.positions.items():
            current_price = current_prices.get(code, pos['avg_price'])
            value = pos['shares'] * current_price
            pnl = (current_price - pos['avg_price']) * pos['shares']
            pnl_pct = ((current_price / pos['avg_price']) - 1) * 100
            
            positions_value += value
            position_details.append({
                'code': code,
                'name': pos.get('stock_name', code),
                'shares': pos['shares'],
                'avg_price': round(pos['avg_price'], 3),
                'current_price': round(current_price, 3),
                'value': round(value, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'allocation_pct': 0  # filled below
            })
        
        total_value = self.cash + positions_value
        roi = ((total_value / self.initial_capital) - 1) * 100
        
        # Fill allocation percentages
        for p in position_details:
            p['allocation_pct'] = round((p['value'] / total_value) * 100, 1) if total_value > 0 else 0
        
        return {
            'total_value': round(total_value, 2),
            'cash': round(self.cash, 2),
            'invested': round(positions_value, 2),
            'roi_pct': round(roi, 2),
            'initial_capital': self.initial_capital,
            'num_positions': len(self.positions),
            'positions': sorted(position_details, key=lambda x: x['value'], reverse=True),
            'risk_profile': self.risk_profile
        }
    
    def compute_performance_metrics(self, daily_values: list) -> dict:
        """Compute Sharpe ratio, Max Drawdown from daily portfolio values."""
        if len(daily_values) < 2:
            return {}
        
        values = np.array(daily_values)
        returns = np.diff(values) / values[:-1]
        
        # Sharpe Ratio (annualized, risk-free rate ~7% for Tunisia)
        rf_daily = 0.07 / 252
        excess_returns = returns - rf_daily
        sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Max Drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Calmar Ratio
        annual_return = (values[-1] / values[0]) ** (252 / len(values)) - 1
        calmar = abs(annual_return / (max_drawdown / 100)) if max_drawdown != 0 else 0
        
        return {
            'sharpe_ratio': round(sharpe, 4),
            'max_drawdown_pct': round(max_drawdown, 2),
            'annual_return_pct': round(annual_return * 100, 2),
            'volatility_pct': round(np.std(returns) * np.sqrt(252) * 100, 2),
            'calmar_ratio': round(calmar, 4),
            'total_return_pct': round(((values[-1] / values[0]) - 1) * 100, 2)
        }


class DecisionEngine:
    """
    Rule-based decision engine with mandatory explainability.
    Generates buy/sell/hold recommendations with detailed reasoning.
    """
    
    def __init__(self, risk_profile: str = 'moderate'):
        self.risk_profile = risk_profile
        self.config = RiskProfile.get_config(risk_profile)
    
    def recommend(self, stock_data: dict) -> dict:
        """
        Generate recommendation with explainability.
        
        stock_data should contain:
        - forecast: dict with price predictions
        - sentiment: dict with sentiment score
        - anomaly: dict with anomaly flags
        - technicals: dict with RSI, MACD, etc.
        - current_price: float
        """
        signals = []
        explanations = []
        
        current_price = stock_data.get('current_price', 0)
        stock_name = stock_data.get('stock_name', 'Unknown')
        
        # 1. Forecast signal
        forecast = stock_data.get('forecast', {})
        if forecast and 'ensemble' in forecast:
            ensemble = forecast['ensemble']
            if 'forecast' in ensemble and len(ensemble['forecast']) > 0:
                pred_price = ensemble['forecast'][-1]
                expected_return = ((pred_price / current_price) - 1) * 100 if current_price > 0 else 0
                
                if expected_return > 2:
                    signals.append(('buy', 0.3))
                    explanations.append(f"📈 Prévision haussière: {expected_return:+.1f}% sur 5 jours (cible: {pred_price:.3f} TND)")
                elif expected_return < -2:
                    signals.append(('sell', 0.3))
                    explanations.append(f"📉 Prévision baissière: {expected_return:+.1f}% sur 5 jours")
                else:
                    signals.append(('hold', 0.1))
                    explanations.append(f"➡️ Prévision neutre: {expected_return:+.1f}% sur 5 jours")
        
        # 2. Sentiment signal
        sentiment = stock_data.get('sentiment', {})
        sent_score = sentiment.get('score', 0)
        if sent_score > 0.3:
            signals.append(('buy', 0.2))
            explanations.append(f"😊 Sentiment positif: score {sent_score:+.2f}")
        elif sent_score < -0.3:
            signals.append(('sell', 0.2))
            explanations.append(f"😟 Sentiment négatif: score {sent_score:+.2f}")
        else:
            signals.append(('hold', 0.05))
            explanations.append(f"😐 Sentiment neutre: score {sent_score:+.2f}")
        
        # 3. Technical signals
        technicals = stock_data.get('technicals', {})
        rsi = technicals.get('rsi')
        if rsi is not None:
            if rsi < 30:
                signals.append(('buy', 0.25))
                explanations.append(f"🔵 RSI survendu ({rsi:.0f}): opportunité d'achat technique")
            elif rsi > 70:
                signals.append(('sell', 0.25))
                explanations.append(f"🔴 RSI suracheté ({rsi:.0f}): signal de vente technique")
            else:
                signals.append(('hold', 0.05))
                explanations.append(f"⚪ RSI neutre ({rsi:.0f})")
        
        macd_hist = technicals.get('macd_hist')
        if macd_hist is not None:
            if macd_hist > 0:
                signals.append(('buy', 0.15))
                explanations.append(f"📊 MACD positif ({macd_hist:.3f}): momentum haussier")
            else:
                signals.append(('sell', 0.15))
                explanations.append(f"📊 MACD négatif ({macd_hist:.3f}): momentum baissier")
        
        # 4. Anomaly warning
        anomaly = stock_data.get('anomaly', {})
        if anomaly.get('has_recent_anomaly', False):
            signals.append(('hold', 0.2))
            explanations.append("⚠️ Anomalie détectée récemment: prudence recommandée")
        
        # 5. Aggregate decision
        recommendation = self._aggregate_signals(signals)
        
        # Risk adjustment
        if self.risk_profile == 'conservative':
            if recommendation['action'] == 'buy' and recommendation['confidence'] < 0.6:
                recommendation['action'] = 'hold'
                explanations.append("🛡️ Profil conservateur: signal d'achat insuffisant, recommandation ajustée à CONSERVER")
        
        recommendation['explanations'] = explanations
        recommendation['stock'] = stock_name
        recommendation['stock_code'] = stock_data.get('stock_code', '')
        recommendation['risk_profile'] = self.risk_profile
        recommendation['current_price'] = current_price
        
        return recommendation
    
    def _aggregate_signals(self, signals: list) -> dict:
        """Weighted aggregation of all signals."""
        if not signals:
            return {'action': 'hold', 'confidence': 0.0}
        
        buy_score = sum(w for a, w in signals if a == 'buy')
        sell_score = sum(w for a, w in signals if a == 'sell')
        hold_score = sum(w for a, w in signals if a == 'hold')
        total = buy_score + sell_score + hold_score
        
        if total == 0:
            return {'action': 'hold', 'confidence': 0.0}
        
        scores = {'buy': buy_score / total, 'sell': sell_score / total, 'hold': hold_score / total}
        action = max(scores, key=scores.get)
        confidence = scores[action]
        
        return {
            'action': action.upper() if action != 'hold' else 'HOLD',
            'action_fr': {'buy': 'ACHETER', 'sell': 'VENDRE', 'hold': 'CONSERVER'}[action],
            'confidence': round(confidence, 2),
            'scores': {k: round(v, 4) for k, v in scores.items()}
        }
    
    def generate_portfolio_suggestion(self, stocks_data: list, capital: float = 5000.0) -> dict:
        """
        Generate portfolio allocation suggestion.
        The "5000 TND" use case.
        """
        recommendations = []
        for sd in stocks_data:
            rec = self.recommend(sd)
            if rec['action'] == 'BUY' or (rec['action'] == 'ACHETER'):
                rec['expected_return'] = sd.get('expected_return', 0)
                recommendations.append(rec)
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Allocate capital
        max_positions = {'conservative': 5, 'moderate': 4, 'aggressive': 3}
        max_pos = max_positions.get(self.risk_profile, 4)
        
        selected = recommendations[:max_pos]
        if not selected:
            return {
                'suggestion': 'HOLD_CASH',
                'message': f"Aucune opportunité d'achat claire détectée. Recommandation: garder les {capital:.0f} TND en liquidités.",
                'allocations': [],
                'capital': capital,
                'risk_profile': self.risk_profile
            }
        
        # Equal weight allocation with risk adjustment
        total_confidence = sum(r['confidence'] for r in selected)
        allocations = []
        remaining = capital
        
        for i, rec in enumerate(selected):
            weight = rec['confidence'] / total_confidence if total_confidence > 0 else 1 / len(selected)
            amount = round(capital * weight, 2)
            price = rec.get('current_price', 1)
            shares = int(amount / price) if price > 0 else 0
            actual_cost = shares * price
            
            allocations.append({
                'stock': rec['stock'],
                'stock_code': rec.get('stock_code', ''),
                'action': rec['action_fr'],
                'confidence': rec['confidence'],
                'shares': shares,
                'price': price,
                'amount': round(actual_cost, 2),
                'weight_pct': round(weight * 100, 1),
                'explanations': rec['explanations'][:3]
            })
            remaining -= actual_cost
        
        return {
            'suggestion': 'DIVERSIFIED_BUY',
            'message': f"Portefeuille diversifié recommandé ({self.risk_profile}): {len(allocations)} positions",
            'allocations': allocations,
            'invested': round(capital - remaining, 2),
            'cash_remaining': round(remaining, 2),
            'capital': capital,
            'risk_profile': self.risk_profile
        }
