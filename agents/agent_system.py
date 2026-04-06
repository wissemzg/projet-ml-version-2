"""
Developer-facing Agent System for BVMT Trading Assistant.
Implements constrained autonomous agents that can:
- Read terminal outputs/logs
- Detect errors
- Propose/apply fixes safely
- Execute with retries and guardrails
"""
import json
import subprocess
import time
import re
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Dangerous command patterns that the agent must NEVER execute
DANGEROUS_PATTERNS = [
    r'rm\s+-rf\s+/', r'rmdir\s+/s', r'format\s+', r'del\s+/f\s+/q\s+C:',
    r'shutdown', r'reboot', r'mkfs', r'dd\s+if=', r'>\s*/dev/',
    r'curl.*\|\s*sh', r'wget.*\|\s*sh', r'pip\s+install.*--user.*sudo',
    r'DROP\s+TABLE', r'DROP\s+DATABASE', r'TRUNCATE',
    r'os\.system\(.*rm', r'shutil\.rmtree\(\s*["\']/',
]


class SafetyGuard:
    """Ensures agent actions are safe and constrained."""
    
    @staticmethod
    def is_command_safe(command: str) -> tuple:
        """Check if a command is safe to execute. Returns (is_safe, reason)."""
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Blocked dangerous pattern: {pattern}"
        return True, "Command appears safe"
    
    @staticmethod
    def is_patch_safe(filepath: str, patch: str) -> tuple:
        """Validate a code patch before applying."""
        # Don't modify system files
        blocked_paths = ['/etc/', '/usr/', '/bin/', 'C:\\Windows\\', 'C:\\Program Files\\']
        for bp in blocked_paths:
            if filepath.startswith(bp):
                return False, f"Cannot modify system path: {bp}"
        
        # Check patch doesn't introduce dangerous patterns
        dangerous_code = ['os.system', 'subprocess.call.*shell=True', 'eval(', 'exec(']
        for dc in dangerous_code:
            if re.search(dc, patch):
                return False, f"Patch contains potentially dangerous code: {dc}"
        
        return True, "Patch appears safe"


class AgentLogger:
    """Structured logging for agent activities."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.log_file = LOGS_DIR / f"{agent_name}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.entries = []
    
    def log(self, level: str, action: str, details: dict = None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.agent_name,
            'level': level,
            'action': action,
            'details': details or {}
        }
        self.entries.append(entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        return entry


class ErrorDetectorAgent:
    """Agent that reads logs/outputs and detects errors."""
    
    def __init__(self):
        self.logger = AgentLogger('error_detector')
        self.known_fixes = {
            'ModuleNotFoundError': self._fix_missing_module,
            'ImportError': self._fix_import_error,
            'FileNotFoundError': self._fix_file_not_found,
            'SyntaxError': self._fix_syntax_error,
            'ConnectionError': self._fix_connection_error,
            'ValueError': self._fix_value_error,
        }
    
    def analyze_output(self, output: str) -> List[Dict]:
        """Parse output for errors and propose fixes."""
        errors = []
        lines = output.split('\n')
        
        for i, line in enumerate(lines):
            for error_type in self.known_fixes:
                if error_type in line:
                    context = '\n'.join(lines[max(0, i-3):min(len(lines), i+3)])
                    fix = self.known_fixes[error_type](line, context)
                    errors.append({
                        'type': error_type,
                        'line': line.strip(),
                        'line_number': i + 1,
                        'context': context,
                        'proposed_fix': fix,
                        'severity': 'ERROR'
                    })
                    self.logger.log('ERROR', f'Detected {error_type}', {'line': line})
        
        # Check for warnings
        for i, line in enumerate(lines):
            if 'Warning' in line or 'DeprecationWarning' in line:
                errors.append({
                    'type': 'Warning',
                    'line': line.strip(),
                    'line_number': i + 1,
                    'proposed_fix': {'action': 'review', 'description': 'Consider addressing this warning'},
                    'severity': 'WARNING'
                })
        return errors
    
    def _fix_missing_module(self, line: str, context: str) -> dict:
        match = re.search(r"No module named '(\S+)'", line)
        module = match.group(1) if match else 'unknown'
        return {
            'action': 'install_package',
            'command': f'pip install {module}',
            'description': f'Install missing module: {module}'
        }
    
    def _fix_import_error(self, line: str, context: str) -> dict:
        return {
            'action': 'check_import',
            'description': 'Verify import path and module availability'
        }
    
    def _fix_file_not_found(self, line: str, context: str) -> dict:
        match = re.search(r"No such file.*'(.+)'", line)
        filepath = match.group(1) if match else 'unknown'
        return {
            'action': 'create_file_or_fix_path',
            'description': f'File not found: {filepath}. Check path or create file.'
        }
    
    def _fix_syntax_error(self, line: str, context: str) -> dict:
        return {
            'action': 'fix_syntax',
            'description': 'Review syntax near the indicated line'
        }
    
    def _fix_connection_error(self, line: str, context: str) -> dict:
        return {
            'action': 'retry_with_backoff',
            'description': 'Network issue detected. Will retry with exponential backoff.'
        }
    
    def _fix_value_error(self, line: str, context: str) -> dict:
        return {
            'action': 'validate_input',
            'description': 'Check input data types and values'
        }


class ExecutionAgent:
    """Agent that executes tasks with retries, backoff, and guardrails."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = AgentLogger('executor')
        self.safety = SafetyGuard()
        self.error_detector = ErrorDetectorAgent()
    
    def execute_with_retry(self, command: str, description: str = "") -> dict:
        """Execute a command with retries and exponential backoff."""
        is_safe, reason = self.safety.is_command_safe(command)
        if not is_safe:
            self.logger.log('BLOCKED', 'Command blocked', {'command': command, 'reason': reason})
            return {'success': False, 'reason': reason, 'blocked': True}
        
        for attempt in range(self.max_retries):
            try:
                self.logger.log('INFO', f'Executing (attempt {attempt + 1})', 
                              {'command': command, 'description': description})
                
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    timeout=120, cwd=str(Path(__file__).resolve().parent.parent)
                )
                
                if result.returncode == 0:
                    self.logger.log('SUCCESS', 'Command succeeded', 
                                  {'output_length': len(result.stdout)})
                    return {
                        'success': True,
                        'output': result.stdout,
                        'attempt': attempt + 1
                    }
                else:
                    errors = self.error_detector.analyze_output(result.stderr)
                    self.logger.log('RETRY', f'Command failed (attempt {attempt + 1})',
                                  {'stderr': result.stderr[:500], 'errors': errors})
                    
                    # Try to auto-fix
                    for err in errors:
                        fix = err.get('proposed_fix', {})
                        if fix.get('action') == 'install_package':
                            fix_cmd = fix.get('command', '')
                            safe, _ = self.safety.is_command_safe(fix_cmd)
                            if safe:
                                subprocess.run(fix_cmd, shell=True, capture_output=True, timeout=60)
                    
                    # Backoff
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
            
            except subprocess.TimeoutExpired:
                self.logger.log('TIMEOUT', f'Command timed out (attempt {attempt + 1})')
            except Exception as e:
                self.logger.log('ERROR', f'Execution error', {'error': str(e)})
        
        return {'success': False, 'reason': 'All retries exhausted'}
    
    def apply_patch(self, filepath: str, old_content: str, new_content: str, 
                    reason: str = "") -> dict:
        """Apply a code patch safely with backup."""
        is_safe, safety_reason = self.safety.is_patch_safe(filepath, new_content)
        if not is_safe:
            return {'success': False, 'reason': safety_reason}
        
        try:
            # Create backup
            backup_path = filepath + '.bak'
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    original = f.read()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original)
                
                # Apply patch
                updated = original.replace(old_content, new_content)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(updated)
                
                self.logger.log('PATCH', 'Applied patch', {
                    'file': filepath,
                    'reason': reason,
                    'backup': backup_path
                })
                
                return {
                    'success': True,
                    'file': filepath,
                    'backup': backup_path,
                    'reason': reason,
                    'changes': f'Replaced {len(old_content)} chars with {len(new_content)} chars'
                }
            else:
                return {'success': False, 'reason': 'File not found'}
        except Exception as e:
            return {'success': False, 'reason': str(e)}


class ChatAgent:
    """
    Chat agent using OpenAI Chat Completions for interactive assistance.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        self.model = os.environ.get('OPENAI_MODEL', 'gpt-4o')
        self.logger = AgentLogger('chat_agent')
        self.conversation_history = []
        self.system_prompt = """Tu es un assistant intelligent de trading pour la BVMT (Bourse de Tunis).
Tu aides les investisseurs tunisiens à:
- Comprendre les prévisions de prix et de liquidité
- Interpréter le sentiment du marché
- Analyser les anomalies détectées
- Prendre des décisions d'investissement éclairées
- Donner des suggestions d'achat/vente basées sur leur portefeuille réel

Tu dois toujours:
- Utiliser les données du portefeuille de l'utilisateur (capital liquide, positions, PnL) quand elles sont disponibles dans le contexte
- Donner des suggestions d'investissement personnalisées basées sur le profil de risque et le capital disponible
- Expliquer tes recommandations clairement
- Mentionner les risques associés
- Rappeler que le trading comporte des risques
- Répondre en français par défaut, ou en arabe si demandé

Quand l'utilisateur demande des suggestions IA ou des recommandations d'investissement, utilise les données de son portefeuille
(capital_liquide, valeur_titres, valeur_totale, positions, roi_pct) pour proposer des actions concrètes.

Contexte: Marché BVMT, cadre réglementaire CMF, devise TND."""
    
    def chat(self, user_message: str, context: dict = None) -> str:
        """Send message to OpenAI and get response."""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Build messages with context
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if context:
            context_msg = f"Contexte actuel du marché:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
            messages.append({"role": "system", "content": context_msg})
        
        messages.extend(self.conversation_history[-10:])  # Keep last 10 messages
        
        try:
            print(f"[CHAT] Calling OpenAI API with {len(messages)} messages, key={self.api_key[:20]}...", flush=True)
            response = self._call_api(messages)
            self.conversation_history.append({"role": "assistant", "content": response})
            self.logger.log('CHAT', 'Response generated', {'length': len(response)})
            print(f"[CHAT] OpenAI response OK ({len(response)} chars)", flush=True)
            return response
        except Exception as e:
            print(f"[CHAT ERROR] API call failed: {type(e).__name__}: {e}", flush=True)
            self.logger.log('ERROR', 'Chat failed', {'error': str(e)})
            return self._fallback_response(user_message, context)
    
    def _call_api(self, messages: list) -> str:
        """Call OpenAI Chat Completions API."""
        if not self.api_key:
            raise ValueError("No API key configured")
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        return data.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
    
    def _fallback_response(self, message: str, context: dict = None) -> str:
        """Provide a useful response when API is unavailable."""
        msg_lower = message.lower()
        
        if any(w in msg_lower for w in ['investir', 'acheter', 'recommand', 'portefeuille', '5000']):
            return """📊 **Recommandation de portefeuille (mode hors ligne)**

Pour un investissement de 5000 TND, voici une approche prudente :

1. **Diversification** : Répartir sur 3-5 valeurs de secteurs différents
2. **Secteur bancaire** (30-40%) : BIAT, Attijari Bank - piliers du marché
3. **Secteur industriel** (20-30%) : SFBT - valeur de fond de portefeuille
4. **Liquidités** (20-30%) : Garder une réserve pour les opportunités

⚠️ Ceci est une suggestion générique. Consultez les prévisions et le sentiment actuels dans l'onglet détail pour chaque valeur.

💡 Le trading comporte des risques. Ne jamais investir plus que ce que vous pouvez vous permettre de perdre."""
        
        if any(w in msg_lower for w in ['anomalie', 'alerte', 'suspect']):
            return """🔔 **Système d'alertes (mode hors ligne)**

Les anomalies détectées sont classées par sévérité :
- 🔴 **CRITIQUE** : Combinaison de plusieurs signaux inhabituels
- 🟠 **HAUTE** : Volume ou prix anormal significatif
- 🟡 **MOYENNE** : Déviation modérée des patterns habituels

Consultez l'onglet Surveillance pour les détails en temps réel."""
        
        return f"""Je suis l'assistant de trading BVMT. Je peux vous aider avec :
- 📈 Prévisions de prix et liquidité
- 📰 Analyse du sentiment du marché
- 🔔 Alertes d'anomalies
- 💼 Gestion de portefeuille

Comment puis-je vous aider ?"""
    
    def clear_history(self):
        self.conversation_history = []
