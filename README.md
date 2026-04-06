# Tradeily — Intelligent Trading Assistant
Group members : - Wissem ZAOUGA
                - Ghada KHCHIMI
---              
## Project Overview

Tradeily is a full-stack, AI-powered machine learning project built to analyze and exploit data from the BVMT (Bourse des Valeurs Mobilières de Tunis). Its objective is to explore financial datasets, carry out data preprocessing and visualization, and develop predictive models for informed decision-making in the Tunisian stock market. The platform combines forecasting, anomaly detection, portfolio management, and multi-agent orchestration in a single intelligent environment tailored to BVMT trading and market analysis.

###Tradeili is a comprehensive intelligent trading assistant built for the BVMT, offering:
---
-Forecasting — predictive modeling based on EMA extrapolation, weighted regression, and optional XGBoost integration, with AIC/BIC-based model selection and ADF stationarity checks
-Sentiment Analysis — AI-powered multilingual sentiment evaluation of market news and trends in French and Arabic via GPT-4o
-Anomaly Detection — hybrid market monitoring that combines statistical indicators with machine learning-based anomaly identification
-Portfolio Management — smart portfolio support with explainable recommendations, configurable risk profiles, and Sharpe ratio-based simulation
-Multi-Agent System — an orchestrated 5-agent analysis pipeline with workflow traceability and safety guardrails
-Real-Time Scraping — live market data acquisition from ilboursa and BVMT sources, stored through persistent snapshots and structured tick logs
-Reinforcement Learning — adaptive portfolio optimization driven by user behavior and feedback
-GPT-4o Chat — an intelligent conversational interface for contextual financial assistance and market Q&A
SARIMA Dashboard — advanced statistical visualization of model quality and stationarity indicators across stocks

## 🗂️ Dataset structure

Source: BVMT Dataset
Dataset location and structure:
/data 
 /_converted_csv 
 /_raw_csv
 /merged_csv
 /report
