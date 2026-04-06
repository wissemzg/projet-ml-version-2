# Agent Architecture — BVMT Trading Assistant

## Overview
The agent system provides developer-facing autonomous capabilities constrained by strict safety guardrails.

## Architecture

```
┌─────────────────────────────────────────────┐
│              Agent System                    │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ ChatAgent   │  │ ExecutionAgent       │  │
│  │ (GPT-4o-mini)│  │ (retry+backoff)     │  │
│  └──────┬──────┘  └──────────┬───────────┘  │
│         │                    │               │
│  ┌──────┴──────┐  ┌─────────┴────────────┐  │
│  │ Safety      │  │ ErrorDetectorAgent   │  │
│  │ Guard       │  │ (log parser)         │  │
│  └─────────────┘  └──────────────────────┘  │
│         │                    │               │
│  ┌──────┴────────────────────┴───────────┐  │
│  │         AgentLogger (JSONL)           │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Components

### 1. ChatAgent
- **Purpose**: Interactive assistant for investors
- **Model**: GPT-4o via OpenAI API
- **Capabilities**: Answer trading questions, provide market context, portfolio advice
- **Fallback**: Keyword-based response when API unavailable
- **Constraints**: Never provides specific financial guarantees; always includes risk warnings

### 2. ExecutionAgent
- **Purpose**: Execute development/maintenance commands with retry logic
- **Features**:
  - Exponential backoff (base 1s, max 3 retries)
  - Auto-detection of missing packages → install
  - Safe patch application with file backup
  - 120s timeout per command
- **Constraints**: All commands pass through SafetyGuard before execution

### 3. ErrorDetectorAgent
- **Purpose**: Parse terminal output and log files for errors
- **Detects**: ModuleNotFoundError, ImportError, FileNotFoundError, SyntaxError, ConnectionError, ValueError
- **Output**: Structured error reports with proposed fixes

### 4. SafetyGuard
- **Purpose**: Prevent dangerous operations
- **Blocked patterns**:
  - `rm -rf /`, `format`, `shutdown`, `DROP TABLE`
  - Pipe to shell execution (`curl | sh`)
  - System file modifications
  - Code injection (`eval()`, `exec()`)
- **Principle**: Whitelist approach — only known-safe operations proceed

### 5. AgentLogger
- **Format**: JSONL (one JSON object per line)
- **Fields**: timestamp, agent name, level, action, details
- **Location**: `logs/` directory
- **Rotation**: Daily files

## Safety Constraints (Non-Negotiable)
1. **Never execute dangerous commands** — blocked at pattern level
2. **Patch format only** — all code changes via controlled replacement with backup
3. **Explain all changes** — every action logged with reason
4. **No system file access** — restricted to project directory
5. **Timeout enforcement** — 120s max per command
6. **Retry limits** — max 3 attempts with backoff
