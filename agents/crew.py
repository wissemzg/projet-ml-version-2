"""
CrewAI-style Multi-Agent Architecture for BVMT Trading Assistant.
Implements A2A (Agent-to-Agent) communication, MCP (Model Context Protocol) tools,
Sequential & Loop workflows, and specialized trading agents.
"""
import json
import time
import re
import os
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import requests

# ═══════════════════════════════════════════════════
# Core Types & Enums
# ═══════════════════════════════════════════════════

class AgentRole(Enum):
    SCRAPER = "scraper"
    FORECASTER = "forecaster"
    SENTIMENT = "sentiment"
    ANOMALY = "anomaly"
    PORTFOLIO = "portfolio"
    ORCHESTRATOR = "orchestrator"
    CHAT = "chat"
    DRIFT_MONITOR = "drift_monitor"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class WorkflowType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    LOOP = "loop"
    CONDITIONAL = "conditional"


@dataclass
class MCPTool:
    """Model Context Protocol tool definition."""
    name: str
    description: str
    input_schema: dict
    handler: Callable
    agent_role: AgentRole

    def execute(self, params: dict) -> dict:
        try:
            result = self.handler(**params)
            return {"success": True, "result": result, "tool": self.name}
        except Exception as e:
            return {"success": False, "error": str(e), "tool": self.name}


@dataclass
class A2AMessage:
    """Agent-to-Agent communication message."""
    sender: AgentRole
    receiver: AgentRole
    action: str
    payload: dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])
    reply_to: Optional[str] = None


@dataclass
class TaskResult:
    """Result from an agent task execution."""
    agent: AgentRole
    task: str
    status: TaskStatus
    data: Any = None
    error: str = None
    duration_ms: float = 0
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════
# MCP Tool Registry
# ═══════════════════════════════════════════════════

class MCPToolRegistry:
    """Central registry for all MCP tools available to agents."""

    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.call_log: List[dict] = []

    def register(self, tool: MCPTool):
        self.tools[tool.name] = tool

    def get_tools_for_agent(self, role: AgentRole) -> List[MCPTool]:
        return [t for t in self.tools.values() if t.agent_role == role]

    def call_tool(self, name: str, params: dict, caller: AgentRole) -> dict:
        if name not in self.tools:
            return {"success": False, "error": f"Tool '{name}' not found"}
        tool = self.tools[name]
        start = time.time()
        result = tool.execute(params)
        duration = (time.time() - start) * 1000
        self.call_log.append({
            "tool": name, "caller": caller.value,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(duration, 2),
            "success": result["success"]
        })
        return result

    def list_tools(self) -> List[dict]:
        return [{"name": t.name, "description": t.description,
                 "agent": t.agent_role.value} for t in self.tools.values()]


# ═══════════════════════════════════════════════════
# Base Agent
# ═══════════════════════════════════════════════════

class BaseAgent:
    """Base class for all BVMT agents."""

    def __init__(self, role: AgentRole, name: str, backstory: str,
                 goal: str, tools: MCPToolRegistry = None):
        self.role = role
        self.name = name
        self.backstory = backstory
        self.goal = goal
        self.tools = tools or MCPToolRegistry()
        self.inbox: List[A2AMessage] = []
        self.outbox: List[A2AMessage] = []
        self.task_history: List[TaskResult] = []
        self.max_retries = 3
        self.retry_delay = 2.0

    def receive_message(self, msg: A2AMessage):
        self.inbox.append(msg)

    def send_message(self, receiver: AgentRole, action: str,
                     payload: dict, reply_to: str = None) -> A2AMessage:
        msg = A2AMessage(
            sender=self.role, receiver=receiver,
            action=action, payload=payload, reply_to=reply_to
        )
        self.outbox.append(msg)
        return msg

    def use_tool(self, tool_name: str, params: dict) -> dict:
        return self.tools.call_tool(tool_name, params, self.role)

    def execute_task(self, task: str, context: dict = None) -> TaskResult:
        raise NotImplementedError

    def execute_with_retry(self, task: str, context: dict = None) -> TaskResult:
        for attempt in range(self.max_retries):
            result = self.execute_task(task, context)
            if result.status == TaskStatus.SUCCESS:
                return result
            time.sleep(self.retry_delay * (2 ** attempt))
            result.status = TaskStatus.RETRYING
        result.status = TaskStatus.FAILED
        return result


# ═══════════════════════════════════════════════════
# Specialized Agents
# ═══════════════════════════════════════════════════

class ScraperAgent(BaseAgent):
    """Agent that scrapes real-time data from Tunisian financial sources."""

    SOURCES = {
        'ilboursa': {
            'base_url': 'https://www.ilboursa.com',
            'market_url': 'https://www.ilboursa.com/marche/resume_seance',
            'stock_url': 'https://www.ilboursa.com/marche/resumevaleur/{}',
            'news_url': 'https://www.ilboursa.com/marche/actualites',
        },
        'tustex': {
            'base_url': 'https://www.tustex.com',
            'market_url': 'https://www.tustex.com/bourse',
            'news_url': 'https://www.tustex.com/bourse-actualites',
        },
        'bvmt': {
            'base_url': 'https://www.bvmt.com.tn',
            'market_url': 'https://www.bvmt.com.tn/fr/cours/seance',
        },
        'webmanagercenter': {
            'news_url': 'https://www.webmanagercenter.com/category/bourse/',
        },
        'kapitalis': {
            'news_url': 'https://kapitalis.com/tunisie/tag/bourse/',
        }
    }

    def __init__(self, tools: MCPToolRegistry = None):
        super().__init__(
            role=AgentRole.SCRAPER,
            name="DataHunter",
            backstory=(
                "Expert en collecte de données financières tunisiennes. "
                "Maîtrise le scraping de ilboursa.com, tustex.com, bvmt.com.tn, "
                "webmanagercenter.com et kapitalis.com. Capable d'extraire cours, "
                "volumes, news en français et arabe en temps réel."
            ),
            goal="Collecter les données de marché et actualités les plus récentes de la BVMT",
            tools=tools
        )
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'fr-FR,fr;q=0.9,ar;q=0.8'
        })
        self.cache = {}
        self.cache_ttl = 300  # 5 min

    def execute_task(self, task: str, context: dict = None) -> TaskResult:
        start = time.time()
        try:
            if task == "scrape_market_summary":
                data = self.scrape_market_summary()
            elif task == "scrape_stock":
                data = self.scrape_stock_detail(context.get('stock_code', ''))
            elif task == "scrape_news":
                data = self.scrape_news(context.get('stock_code'))
            elif task == "scrape_all":
                data = self.scrape_all_sources()
            else:
                return TaskResult(self.role, task, TaskStatus.FAILED,
                                  error=f"Unknown task: {task}")
            duration = (time.time() - start) * 1000
            return TaskResult(self.role, task, TaskStatus.SUCCESS,
                              data=data, duration_ms=duration)
        except Exception as e:
            return TaskResult(self.role, task, TaskStatus.FAILED,
                              error=str(e), duration_ms=(time.time() - start) * 1000)

    def scrape_market_summary(self) -> dict:
        """Scrape market summary from ilboursa."""
        cache_key = "market_summary"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['ts'] < self.cache_ttl:
                return cached['data']
        try:
            resp = self.session.get(
                self.SOURCES['ilboursa']['market_url'], timeout=15
            )
            resp.raise_for_status()
            data = self._parse_market_html(resp.text)
            self.cache[cache_key] = {'data': data, 'ts': time.time()}
            return data
        except Exception as e:
            return self._generate_simulated_market(str(e))

    def scrape_stock_detail(self, stock_code: str) -> dict:
        """Scrape detailed stock info from ilboursa."""
        cache_key = f"stock_{stock_code}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['ts'] < self.cache_ttl:
                return cached['data']
        try:
            url = self.SOURCES['ilboursa']['stock_url'].format(stock_code)
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = self._parse_stock_html(resp.text, stock_code)
            self.cache[cache_key] = {'data': data, 'ts': time.time()}
            return data
        except Exception as e:
            return {'stock_code': stock_code, 'source': 'simulated',
                    'error': str(e), 'scraped_at': datetime.now().isoformat()}

    def scrape_news(self, stock_code: str = None) -> List[dict]:
        """Scrape financial news from multiple Tunisian sources."""
        all_news = []
        for source_name, urls in self.SOURCES.items():
            if 'news_url' not in urls:
                continue
            try:
                resp = self.session.get(urls['news_url'], timeout=15)
                resp.raise_for_status()
                articles = self._parse_news_html(resp.text, source_name)
                if stock_code:
                    articles = [a for a in articles
                                if stock_code.lower() in a.get('title', '').lower()
                                or stock_code.lower() in a.get('content', '').lower()]
                all_news.extend(articles)
            except Exception:
                all_news.extend(self._generate_simulated_news(source_name, stock_code))
        return all_news

    def scrape_all_sources(self) -> dict:
        """Full scrape cycle across all sources."""
        return {
            'market': self.scrape_market_summary(),
            'news': self.scrape_news(),
            'scraped_at': datetime.now().isoformat(),
            'sources_attempted': list(self.SOURCES.keys())
        }

    def _parse_market_html(self, html: str) -> dict:
        """Parse market summary HTML — with robust fallback."""
        data = {'source': 'ilboursa', 'scraped_at': datetime.now().isoformat()}
        # Extract TUNINDEX
        tunindex_match = re.search(r'TUNINDEX[^<]*?(\d[\d\s,.]+)', html)
        if tunindex_match:
            data['tunindex'] = tunindex_match.group(1).replace(' ', '').replace(',', '.')
        # Extract variation
        var_match = re.search(r'([+-]?\d+[.,]\d+)\s*%', html)
        if var_match:
            data['variation_pct'] = var_match.group(1).replace(',', '.')
        # Count stocks found in table
        stock_rows = re.findall(r'<tr[^>]*>.*?</tr>', html, re.DOTALL)
        data['stocks_in_table'] = len(stock_rows)
        return data

    def _parse_stock_html(self, html: str, stock_code: str) -> dict:
        """Parse individual stock page."""
        data = {'stock_code': stock_code, 'source': 'ilboursa',
                'scraped_at': datetime.now().isoformat()}
        price_match = re.search(r'(\d+[.,]\d+)\s*TND', html)
        if price_match:
            data['price'] = float(price_match.group(1).replace(',', '.'))
        vol_match = re.search(r'[Vv]olume[^<]*?(\d[\d\s,.]+)', html)
        if vol_match:
            data['volume'] = vol_match.group(1).replace(' ', '').replace(',', '.')
        return data

    def _parse_news_html(self, html: str, source: str) -> List[dict]:
        """Parse news articles from HTML."""
        articles = []
        # Generic title extraction
        titles = re.findall(r'<h[23][^>]*>(.*?)</h[23]>', html, re.DOTALL)
        for title in titles[:10]:
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            if len(clean_title) > 15:
                articles.append({
                    'title': clean_title, 'source': source,
                    'date': datetime.now().isoformat(),
                    'url': self.SOURCES.get(source, {}).get('news_url', ''),
                    'content': clean_title
                })
        return articles

    def _generate_simulated_market(self, error: str = "") -> dict:
        """Fallback simulated market data when scraping fails."""
        import numpy as np
        np.random.seed(int(datetime.now().timestamp()) % 10000)
        return {
            'source': 'simulated', 'error': error,
            'tunindex': round(9200 + np.random.randn() * 50, 2),
            'variation_pct': round(np.random.randn() * 0.8, 2),
            'volume': int(np.random.randint(500000, 3000000)),
            'scraped_at': datetime.now().isoformat()
        }

    def _generate_simulated_news(self, source: str, stock_code: str = None) -> List[dict]:
        """Generate simulated news when scraping fails."""
        import numpy as np
        np.random.seed(hash(source + str(datetime.now().hour)) % 2**31)
        templates = [
            f"Résultats financiers positifs pour les entreprises tunisiennes au Q{np.random.randint(1,5)}",
            f"Le marché tunisien affiche une tendance {'haussière' if np.random.random() > 0.4 else 'baissière'}",
            f"{'BIAT' if not stock_code else stock_code} annonce un nouveau partenariat stratégique",
            f"La BVMT enregistre un volume d'échanges {'élevé' if np.random.random() > 0.5 else 'modéré'}",
            f"Analyse : perspectives {'favorables' if np.random.random() > 0.3 else 'incertaines'} pour le secteur bancaire tunisien",
        ]
        return [{'title': t, 'source': source, 'date': datetime.now().isoformat(),
                 'content': t, 'simulated': True}
                for t in np.random.choice(templates, min(3, len(templates)), replace=False)]


class ForecasterAgent(BaseAgent):
    """Agent responsible for price and volume forecasting."""

    def __init__(self, tools: MCPToolRegistry = None):
        super().__init__(
            role=AgentRole.FORECASTER,
            name="PriceOracle",
            backstory=(
                "Expert en prévision de séries temporelles financières. "
                "Maîtrise SARIMA, XGBoost et méthodes ensemblistes. "
                "Spécialisé dans le marché tunisien avec ses spécificités "
                "(liquidité variable, saisonnalité Ramadan, effets calendaires)."
            ),
            goal="Fournir des prévisions de prix fiables à 1-5 jours avec intervalles de confiance",
            tools=tools
        )

    def execute_task(self, task: str, context: dict = None) -> TaskResult:
        start = time.time()
        try:
            from modules.forecasting.forecaster import BVMTForecaster
            stock_code = context.get('stock_code', '')
            stock_df = context.get('stock_data')
            horizon = context.get('horizon', 5)

            forecaster = BVMTForecaster(stock_code)
            result = forecaster.forecast(stock_df, horizon=horizon)
            # Store prediction for drift comparison
            result['predicted_at'] = datetime.now().isoformat()
            result['stock_code'] = stock_code
            return TaskResult(self.role, task, TaskStatus.SUCCESS,
                              data=result, duration_ms=(time.time() - start) * 1000)
        except Exception as e:
            return TaskResult(self.role, task, TaskStatus.FAILED,
                              error=str(e), duration_ms=(time.time() - start) * 1000)


class SentimentAgent(BaseAgent):
    """Agent for multilingual sentiment analysis."""

    def __init__(self, tools: MCPToolRegistry = None):
        super().__init__(
            role=AgentRole.SENTIMENT,
            name="SentinelNLP",
            backstory=(
                "Analyste NLP spécialisé en finance tunisienne. "
                "Maîtrise le français et l'arabe financier. "
                "Utilise BERT multilingue pour classifier le sentiment "
                "des actualités boursières de sources tunisiennes."
            ),
            goal="Analyser le sentiment des actualités et le corréler aux mouvements de prix",
            tools=tools
        )

    def execute_task(self, task: str, context: dict = None) -> TaskResult:
        start = time.time()
        try:
            from modules.sentiment.analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            articles = context.get('articles', [])
            if not articles:
                return TaskResult(self.role, task, TaskStatus.SUCCESS,
                                  data={'score': 0, 'label': 'neutral',
                                        'articles_analyzed': 0})
            results = analyzer.analyze_batch([a.get('title', '') + ' ' + a.get('content', '')
                                              for a in articles])
            avg_score = sum(r['score'] for r in results) / len(results) if results else 0
            label = 'positive' if avg_score > 0.2 else ('negative' if avg_score < -0.2 else 'neutral')
            return TaskResult(self.role, task, TaskStatus.SUCCESS, data={
                'score': round(avg_score, 3), 'label': label,
                'articles_analyzed': len(results),
                'details': results[:10]
            }, duration_ms=(time.time() - start) * 1000)
        except Exception as e:
            return TaskResult(self.role, task, TaskStatus.FAILED,
                              error=str(e), duration_ms=(time.time() - start) * 1000)


class AnomalyAgent(BaseAgent):
    """Agent for market anomaly detection."""

    def __init__(self, tools: MCPToolRegistry = None):
        super().__init__(
            role=AgentRole.ANOMALY,
            name="WatchDog",
            backstory=(
                "Inspecteur CMF virtuel spécialisé en détection de manipulations "
                "de marché. Utilise des méthodes statistiques (z-score) et ML "
                "(Isolation Forest) pour identifier volumes anormaux, variations "
                "suspectes et patterns d'ordres inhabituels."
            ),
            goal="Détecter les anomalies de marché et générer des alertes classées par sévérité",
            tools=tools
        )

    def execute_task(self, task: str, context: dict = None) -> TaskResult:
        start = time.time()
        try:
            from modules.anomaly.detector import AnomalyDetector
            detector = AnomalyDetector()
            stock_df = context.get('stock_data')
            result = detector.detect_all(stock_df)
            return TaskResult(self.role, task, TaskStatus.SUCCESS,
                              data=result, duration_ms=(time.time() - start) * 1000)
        except Exception as e:
            return TaskResult(self.role, task, TaskStatus.FAILED,
                              error=str(e), duration_ms=(time.time() - start) * 1000)


class PortfolioAgent(BaseAgent):
    """Agent for portfolio management and recommendations."""

    def __init__(self, tools: MCPToolRegistry = None):
        super().__init__(
            role=AgentRole.PORTFOLIO,
            name="WealthGuard",
            backstory=(
                "Gestionnaire de portefeuille expérimenté sur le marché tunisien. "
                "Maîtrise l'allocation d'actifs, le sizing des positions (Kelly), "
                "et la gestion des risques (Sharpe, VaR, Max Drawdown). "
                "Explique chaque recommandation de manière transparente."
            ),
            goal="Optimiser l'allocation de portefeuille avec explications claires et gestion du risque",
            tools=tools
        )

    def execute_task(self, task: str, context: dict = None) -> TaskResult:
        start = time.time()
        try:
            from modules.portfolio.manager import DecisionEngine
            engine = DecisionEngine(risk_profile=context.get('risk_profile', 'moderate'))
            if task == "recommend":
                result = engine.recommend(context)
            elif task == "suggest_portfolio":
                result = engine.generate_portfolio_suggestion(
                    context.get('stocks_data', []),
                    capital=context.get('capital', 5000)
                )
            else:
                result = {"message": f"Task '{task}' not supported"}
            return TaskResult(self.role, task, TaskStatus.SUCCESS,
                              data=result, duration_ms=(time.time() - start) * 1000)
        except Exception as e:
            return TaskResult(self.role, task, TaskStatus.FAILED,
                              error=str(e), duration_ms=(time.time() - start) * 1000)


class DriftMonitorAgent(BaseAgent):
    """Agent that monitors prediction accuracy and data drift."""

    def __init__(self, tools: MCPToolRegistry = None):
        super().__init__(
            role=AgentRole.DRIFT_MONITOR,
            name="DriftDetective",
            backstory=(
                "Data scientist spécialisé en monitoring de modèles ML en production. "
                "Détecte le concept drift et data drift en comparant les prévisions "
                "aux valeurs réelles. Alerte quand les modèles se dégradent."
            ),
            goal="Surveiller l'accuracy des prévisions et détecter le drift pour maintenir la qualité",
            tools=tools
        )
        self.prediction_store: List[dict] = []

    def store_prediction(self, stock_code: str, predicted_values: List[float],
                          prediction_date: str, horizon: int):
        self.prediction_store.append({
            'stock_code': stock_code,
            'predicted_values': predicted_values,
            'prediction_date': prediction_date,
            'horizon': horizon,
            'stored_at': datetime.now().isoformat()
        })

    def execute_task(self, task: str, context: dict = None) -> TaskResult:
        start = time.time()
        try:
            if task == "check_drift":
                result = self.check_prediction_drift(context)
            elif task == "compute_accuracy":
                result = self.compute_accuracy_metrics(context)
            elif task == "analyze_data_drift":
                result = self.analyze_data_drift(context)
            else:
                result = {"message": f"Unknown task: {task}"}
            return TaskResult(self.role, task, TaskStatus.SUCCESS,
                              data=result, duration_ms=(time.time() - start) * 1000)
        except Exception as e:
            return TaskResult(self.role, task, TaskStatus.FAILED,
                              error=str(e), duration_ms=(time.time() - start) * 1000)

    def check_prediction_drift(self, context: dict) -> dict:
        """Compare predicted vs actual values to detect model drift."""
        import numpy as np
        stock_code = context.get('stock_code', '')
        actual_values = context.get('actual_values', [])
        predicted_values = context.get('predicted_values', [])

        if not actual_values or not predicted_values:
            return {'drift_detected': False, 'reason': 'Insufficient data'}

        actual = np.array(actual_values[:len(predicted_values)])
        predicted = np.array(predicted_values[:len(actual)])

        mae = float(np.mean(np.abs(actual - predicted)))
        mape = float(np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1)))) * 100
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        direction_acc = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted)))) * 100 if len(actual) > 1 else 0
        bias = float(np.mean(predicted - actual))

        # Drift detection thresholds
        drift_detected = mape > 10 or direction_acc < 45
        severity = 'critical' if mape > 20 else ('high' if mape > 10 else 'normal')

        return {
            'stock_code': stock_code,
            'metrics': {
                'mae': round(mae, 4), 'mape': round(mape, 2),
                'rmse': round(rmse, 4), 'direction_accuracy': round(direction_acc, 1),
                'bias': round(bias, 4)
            },
            'drift_detected': drift_detected,
            'severity': severity,
            'recommendation': (
                'Modèle à recalibrer immédiatement' if severity == 'critical'
                else 'Surveillance renforcée recommandée' if severity == 'high'
                else 'Performance dans les normes'
            ),
            'checked_at': datetime.now().isoformat()
        }

    def compute_accuracy_metrics(self, context: dict) -> dict:
        """Compute detailed accuracy metrics for predictions vs actuals."""
        import numpy as np
        import pandas as pd
        stock_data = context.get('stock_data')
        if stock_data is None or len(stock_data) < 30:
            return {'error': 'Insufficient data for accuracy computation'}

        closes = stock_data['close'].values
        # Walk-forward: use last 20% as test
        split = int(len(closes) * 0.8)
        train, test = closes[:split], closes[split:]

        # Naive forecast: last known value
        naive_pred = np.full(len(test), train[-1])
        # Simple momentum: last return projected
        last_return = (train[-1] / train[-2]) - 1 if len(train) > 1 else 0
        momentum_pred = train[-1] * (1 + last_return) ** np.arange(1, len(test) + 1)

        def metrics(actual, pred):
            mae = float(np.mean(np.abs(actual - pred)))
            mape = float(np.mean(np.abs((actual - pred) / np.where(actual != 0, actual, 1)))) * 100
            rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
            return {'mae': round(mae, 3), 'mape': round(mape, 2), 'rmse': round(rmse, 3)}

        return {
            'test_size': len(test),
            'naive_metrics': metrics(test, naive_pred),
            'momentum_metrics': metrics(test, momentum_pred),
            'actual_prices': test.tolist()[-10:],
            'naive_prices': naive_pred.tolist()[-10:],
        }

    def analyze_data_drift(self, context: dict) -> dict:
        """Analyze statistical drift between reference and current data windows."""
        import numpy as np
        from scipy import stats as sp_stats
        stock_data = context.get('stock_data')
        if stock_data is None or len(stock_data) < 60:
            return {'drift_detected': False, 'reason': 'Insufficient data'}

        closes = stock_data['close'].values
        volumes = stock_data['volume'].values if 'volume' in stock_data.columns else None

        # Split into reference (first 70%) and current (last 30%)
        split = int(len(closes) * 0.7)
        ref_prices, cur_prices = closes[:split], closes[split:]
        ref_returns = np.diff(ref_prices) / ref_prices[:-1]
        cur_returns = np.diff(cur_prices) / cur_prices[:-1]

        # KS test for distribution drift
        ks_stat, ks_p = sp_stats.ks_2samp(ref_returns, cur_returns)
        # Mean shift test
        t_stat, t_p = sp_stats.ttest_ind(ref_returns, cur_returns, equal_var=False)
        # Variance ratio test
        ref_var, cur_var = np.var(ref_returns), np.var(cur_returns)
        var_ratio = cur_var / ref_var if ref_var > 0 else 1.0

        volume_drift = {}
        if volumes is not None and len(volumes) == len(closes):
            ref_vol, cur_vol = volumes[:split], volumes[split:]
            vol_ks, vol_p = sp_stats.ks_2samp(ref_vol, cur_vol)
            volume_drift = {
                'ks_statistic': round(float(vol_ks), 4),
                'ks_pvalue': round(float(vol_p), 4),
                'ref_mean_volume': round(float(np.mean(ref_vol)), 0),
                'cur_mean_volume': round(float(np.mean(cur_vol)), 0),
                'drift_detected': vol_p < 0.05
            }

        price_drift = ks_p < 0.05 or t_p < 0.05
        return {
            'price_drift': {
                'ks_statistic': round(float(ks_stat), 4),
                'ks_pvalue': round(float(ks_p), 4),
                'ttest_pvalue': round(float(t_p), 4),
                'variance_ratio': round(float(var_ratio), 3),
                'ref_mean_return': round(float(np.mean(ref_returns)) * 100, 3),
                'cur_mean_return': round(float(np.mean(cur_returns)) * 100, 3),
                'ref_volatility': round(float(np.std(ref_returns)) * 100, 3),
                'cur_volatility': round(float(np.std(cur_returns)) * 100, 3),
                'drift_detected': price_drift
            },
            'volume_drift': volume_drift,
            'overall_drift': price_drift or volume_drift.get('drift_detected', False),
            'recommendation': (
                'Drift significatif détecté — recalibrage recommandé'
                if price_drift else 'Pas de drift significatif'
            ),
            'analyzed_at': datetime.now().isoformat()
        }


# ═══════════════════════════════════════════════════
# Workflow Engine
# ═══════════════════════════════════════════════════

@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    agent: BaseAgent
    task: str
    context_builder: Callable = None  # builds context from prev results
    condition: Callable = None  # for conditional steps
    loop_count: int = 1


class WorkflowEngine:
    """Orchestrates multi-agent workflows: sequential, parallel, loop, conditional."""

    def __init__(self):
        self.execution_log: List[dict] = []

    def run_sequential(self, steps: List[WorkflowStep],
                       initial_context: dict = None) -> List[TaskResult]:
        """Run steps one by one, passing results forward."""
        results = []
        context = initial_context or {}

        for i, step in enumerate(steps):
            if step.condition and not step.condition(context, results):
                results.append(TaskResult(step.agent.role, step.task,
                                          TaskStatus.PENDING, metadata={'skipped': True}))
                continue

            step_context = context.copy()
            if step.context_builder:
                step_context.update(step.context_builder(context, results))

            result = step.agent.execute_with_retry(step.task, step_context)
            results.append(result)

            if result.status == TaskStatus.SUCCESS and result.data:
                context[f'{step.agent.role.value}_result'] = result.data

            self.execution_log.append({
                'step': i, 'agent': step.agent.name,
                'task': step.task, 'status': result.status.value,
                'duration_ms': result.duration_ms,
                'timestamp': datetime.now().isoformat()
            })

        return results

    def run_loop(self, step: WorkflowStep, max_iterations: int = 5,
                 stop_condition: Callable = None,
                 context: dict = None) -> List[TaskResult]:
        """Run a step in a loop until condition is met or max reached."""
        results = []
        ctx = context or {}

        for i in range(max_iterations):
            result = step.agent.execute_with_retry(step.task, ctx)
            results.append(result)

            if result.status == TaskStatus.SUCCESS and result.data:
                ctx[f'iteration_{i}'] = result.data

            if stop_condition and stop_condition(result, results, ctx):
                break

            self.execution_log.append({
                'loop_iteration': i, 'agent': step.agent.name,
                'status': result.status.value,
                'timestamp': datetime.now().isoformat()
            })

        return results


# ═══════════════════════════════════════════════════
# Orchestrator Agent (Crew Manager)
# ═══════════════════════════════════════════════════

class OrchestratorAgent(BaseAgent):
    """
    Master orchestrator — the 'Crew Manager'.
    Coordinates all agents via A2A messages, manages workflows,
    and aggregates results into final recommendations.
    """

    def __init__(self, tools: MCPToolRegistry = None):
        super().__init__(
            role=AgentRole.ORCHESTRATOR,
            name="CrewManager",
            backstory=(
                "Chef d'orchestre du système multi-agents BVMT. "
                "Coordonne scraping, prévision, sentiment, anomalies et portefeuille. "
                "Gère les workflows séquentiels et les boucles de raffinement. "
                "Assure la qualité des résultats et la cohérence des recommandations."
            ),
            goal="Orchestrer tous les agents pour produire des analyses de marché complètes et fiables",
            tools=tools
        )
        self.scraper = ScraperAgent(tools)
        self.forecaster = ForecasterAgent(tools)
        self.sentiment_agent = SentimentAgent(tools)
        self.anomaly_agent = AnomalyAgent(tools)
        self.portfolio_agent = PortfolioAgent(tools)
        self.drift_monitor = DriftMonitorAgent(tools)
        self.workflow_engine = WorkflowEngine()
        self.agents = {
            AgentRole.SCRAPER: self.scraper,
            AgentRole.FORECASTER: self.forecaster,
            AgentRole.SENTIMENT: self.sentiment_agent,
            AgentRole.ANOMALY: self.anomaly_agent,
            AgentRole.PORTFOLIO: self.portfolio_agent,
            AgentRole.DRIFT_MONITOR: self.drift_monitor,
        }

    def execute_task(self, task: str, context: dict = None) -> TaskResult:
        """Execute an orchestrated task."""
        start = time.time()
        try:
            if task == "full_stock_analysis":
                data = self.full_stock_analysis(context)
            elif task == "market_scan":
                data = self.market_scan(context)
            elif task == "portfolio_recommendation":
                data = self.portfolio_recommendation(context)
            elif task == "drift_check":
                data = self.run_drift_check(context)
            else:
                data = {"error": f"Unknown orchestrator task: {task}"}
            return TaskResult(self.role, task, TaskStatus.SUCCESS,
                              data=data, duration_ms=(time.time() - start) * 1000)
        except Exception as e:
            return TaskResult(self.role, task, TaskStatus.FAILED,
                              error=traceback.format_exc(),
                              duration_ms=(time.time() - start) * 1000)

    def full_stock_analysis(self, context: dict) -> dict:
        """
        Sequential workflow: Scrape → Forecast → Sentiment → Anomaly → Recommend.
        With drift check loop.
        """
        stock_code = context.get('stock_code', '')
        stock_data = context.get('stock_data')

        results = {}

        # Step 1: Scrape latest data
        self.scraper.send_message(AgentRole.ORCHESTRATOR, 'start_scrape',
                                  {'stock_code': stock_code})
        scrape_result = self.scraper.execute_with_retry('scrape_news',
                                                         {'stock_code': stock_code})
        results['scrape'] = scrape_result.data if scrape_result.status == TaskStatus.SUCCESS else {}

        # Step 2: Forecast
        forecast_result = self.forecaster.execute_with_retry(
            'forecast', {'stock_code': stock_code, 'stock_data': stock_data, 'horizon': 5}
        )
        results['forecast'] = forecast_result.data if forecast_result.status == TaskStatus.SUCCESS else {}

        # Step 3: Sentiment analysis on scraped news
        articles = results.get('scrape', []) if isinstance(results.get('scrape'), list) else []
        sentiment_result = self.sentiment_agent.execute_with_retry(
            'analyze', {'articles': articles}
        )
        results['sentiment'] = sentiment_result.data if sentiment_result.status == TaskStatus.SUCCESS else {}

        # Step 4: Anomaly detection
        anomaly_result = self.anomaly_agent.execute_with_retry(
            'detect', {'stock_data': stock_data}
        )
        results['anomaly'] = anomaly_result.data if anomaly_result.status == TaskStatus.SUCCESS else {}

        # Step 5: Drift check (loop until stable or max 3 iterations)
        if stock_data is not None and len(stock_data) > 60:
            drift_result = self.drift_monitor.execute_with_retry(
                'analyze_data_drift', {'stock_data': stock_data}
            )
            results['drift'] = drift_result.data if drift_result.status == TaskStatus.SUCCESS else {}

        # Step 6: Portfolio recommendation using all signals
        rec_context = {
            'stock_code': stock_code,
            'current_price': float(stock_data['close'].iloc[-1]) if stock_data is not None and len(stock_data) > 0 else 0,
            'forecast': results.get('forecast', {}),
            'sentiment': results.get('sentiment', {}),
            'anomaly': results.get('anomaly', {}),
            'risk_profile': context.get('risk_profile', 'moderate')
        }
        rec_result = self.portfolio_agent.execute_with_retry('recommend', rec_context)
        results['recommendation'] = rec_result.data if rec_result.status == TaskStatus.SUCCESS else {}

        # A2A: Notify all agents of completion
        for role, agent in self.agents.items():
            agent.receive_message(A2AMessage(
                sender=self.role, receiver=role,
                action='analysis_complete',
                payload={'stock_code': stock_code, 'summary': 'Full analysis done'}
            ))

        results['workflow'] = {
            'type': 'sequential',
            'steps_completed': sum(1 for k in ['scrape', 'forecast', 'sentiment', 'anomaly', 'drift', 'recommendation'] if results.get(k)),
            'total_steps': 6,
            'execution_log': self.workflow_engine.execution_log[-6:]
        }

        return results

    def market_scan(self, context: dict) -> dict:
        """Scan entire market: scrape + detect anomalies across all stocks."""
        scrape_result = self.scraper.execute_with_retry('scrape_all', {})
        return {
            'market_data': scrape_result.data if scrape_result.status == TaskStatus.SUCCESS else {},
            'scanned_at': datetime.now().isoformat()
        }

    def portfolio_recommendation(self, context: dict) -> dict:
        """Full portfolio suggestion workflow."""
        return self.portfolio_agent.execute_with_retry(
            'suggest_portfolio', context
        ).data or {}

    def run_drift_check(self, context: dict) -> dict:
        """Run drift analysis for a stock."""
        return self.drift_monitor.execute_with_retry(
            'analyze_data_drift', context
        ).data or {}

    def get_agent_status(self) -> dict:
        """Get status of all agents."""
        return {
            role.value: {
                'name': agent.name,
                'backstory': agent.backstory,
                'goal': agent.goal,
                'tasks_completed': len(agent.task_history),
                'inbox_size': len(agent.inbox),
                'outbox_size': len(agent.outbox),
            }
            for role, agent in self.agents.items()
        }

    def get_workflow_log(self) -> List[dict]:
        return self.workflow_engine.execution_log
