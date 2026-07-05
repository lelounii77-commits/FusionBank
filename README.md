<div align="center">

# FusionBanKAI

### AI-Powered Credit Risk Assessment Platform

**Built for the AMD Hackathon 2026 — Alinma Bank**

[Live Demo](https://fusionbank-r9xnexp8.manus.space/)

</div>

---

## Overview

FusionBanKAI is an AI-powered credit risk assessment platform designed to support bank officers in evaluating customer creditworthiness. It combines a machine learning risk-scoring engine with a unified customer financial profile, Open Banking integration, and an explainable AI layer, all presented through a modern banking dashboard.

The platform was developed for Alinma Bank's AMD Hackathon 2026 to demonstrate how AI can accelerate and standardize credit decisions while keeping the final judgment with a human officer.

## Live Demo

The deployed application is available at:

**https://fusionbank-r9xnexp8.manus.space/**

## Key Features

- **Customer Search** — look up customers by national ID, customer ID, phone number, or name
- **Unified Financial Profile** — a consolidated view of a customer's accounts, obligations, income, and expenses from multiple sources
- **Open Banking Integration** — customer-consented account syncing across Saudi banks, in line with SAMA's Open Banking framework
- **AI Credit Analysis** — an automated risk assessment engine that returns a risk level and probability score for each customer
- **Explainable AI** — a breakdown of the key factors driving each prediction (salary stability, debt-to-income ratio, and others), so officers understand *why* a decision was suggested
- **Credit Report Generation** — a structured, signable credit report combining the AI analysis with officer notes and a final decision
- **Reports Dashboard** — aggregated statistics on approvals, rejections, and manual reviews over time
- **Notifications** — real-time updates on financing requests, customer consent, and system events

## Machine Learning Component

### Problem

The core predictive task is a credit risk classification problem: given a customer's financial and demographic profile, predict their credit risk level (Low / Medium / High) so that officers can prioritize review effort and support lending decisions.

### Data

The models were trained on a synthetic banking dataset generated for the hackathon, covering customers, accounts, obligations, and AI-derived financial summaries. Synthetic data was used to avoid exposing real customer information while still reflecting realistic financial patterns.

### Pipeline

- Data cleaning: duplicate removal, missing-value imputation (grouped by relevant categories such as employment sector or account type), and outlier detection using the IQR method
- Table merging: customer, account, and obligation tables were aggregated and merged into a single master dataset
- Leakage prevention: after an initial baseline model revealed target leakage from the AI-summary features, a second iteration removed all fields derived from the risk label (such as default probability, financial stability score, and debt-to-income ratio) to keep the model realistic
- Class imbalance handling: SMOTE was applied to the training data to address the imbalance between Low, Medium, and High risk customers
- Model comparison: Random Forest and XGBoost were trained and compared, including a binary reformulation (Low vs. Not Low) to address weak performance on the minority Medium-risk class
- Evaluation: classification report, confusion matrix, weighted ROC-AUC, and 5-fold cross-validation were used to validate results and check for overfitting

### Result

XGBoost, retrained as a binary classifier (Low vs. Not Low risk), was selected as the best-performing configuration and used as the deployed risk-scoring model.

## System Architecture

FusionBanKAI consists of a machine learning pipeline, a FastAPI backend, and a React frontend:

```
Officer → React Dashboard → FastAPI Backend → ML Model (XGBoost) → Risk Score + Explanation
```

The trained model, label encoder, and feature list are serialized with joblib and loaded by the backend at startup, which keeps preprocessing and inference consistent between training and deployment.

## Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn (SMOTE), Pandas, NumPy |
| Backend | FastAPI, joblib |
| Frontend | React, TypeScript, Vite, pnpm, Tailwind CSS, shadcn/ui, Recharts |
| Routing | Wouter |
| API layer | tRPC |
| Deployment | Manus |

## Project Structure

```
fusionbankai/
├── fusionbankai.py        # Full ML pipeline: cleaning, preprocessing, model training and evaluation
├── client/                # React + TypeScript frontend (dashboard, pages, UI components)
│   └── src/
│       ├── pages/         # Dashboard, CustomerSearch, CreditReport, ExplainableAI, OpenBanking, ...
│       ├── components/    # Shared UI components
│       └── contexts/      # Application state (selected customer, consent, theme)
├── drizzle/                # Database schema and migrations
└── references/             # Supporting project documentation
```

## Ethical and Privacy Considerations

- The model was trained entirely on synthetic data to avoid exposing real customer financial information
- Fields that directly encode the target (such as pre-computed risk scores or default probability) were identified and removed to prevent data leakage
- The platform is designed as a decision-support tool: the AI provides a risk score and explanation, but the final credit decision is made and signed off by a human bank officer

## Limitations

- The dataset is synthetic; performance on real institutional banking data has not been validated
- The Medium-risk class remains the hardest to classify reliably, which is why a binary (Low vs. Not Low) reformulation was introduced as the deployed model
- The Open Banking integration in the demo simulates the consent and sync flow rather than connecting to live bank APIs

## Team

| Name |
|---|
| Layan Alfares |
| Shmokh Aljomah |
| Dana Alsubaie |
| Deem Alrashoud |

*AMD Hackathon 2026 — Alinma Bank*
