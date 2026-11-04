# Protecting Health Data: k-Anonymity vs Differential Privacy under GDPR and NIS2

A Master's Level Software Architecture and Empirical Evaluation Prototype
Course: Data Privacy (Privacidade de Dados) – Academic Year 2025/2026
Institution: ISCTE – Instituto Universitário de Lisboa | School of Technology and Architecture (ISTA)
Professor: Valderi Leithardt

Authors (Group Final Project):
- Mattia Malservigi (145863)
- Matteo Calloni (145860)
- Manuel Mulloni (145866)

## Overview

Sharing electronic health records for scientific research involves an inherent tension between data utility and fundamental rights to privacy. Under GDPR Article 9, health data constitutes a special category of personal data whose processing is prohibited in general, but authorized for scientific research (Art. 9(2)(j) and Art. 89(1)) provided technical and organizational safeguards are implemented. Furthermore, the NIS2 Directive ((EU) 2022/2555) classifies healthcare providers and health platforms as sectors of high criticality (Annex I), imposing mandatory cybersecurity risk-management (Art. 21) and rapid incident reporting (Art. 23).

This project implements a Secure Privacy Gateway that mediates all researcher queries and data exports, comparing two privacy-preserving paradigms:
1. Syntactic Anonymization (k-Anonymity via Mondrian Partitioning): Protects microdata releases once by generalizing quasi-identifiers so that every patient hides within an equivalence class of at least k individuals (|EC| >= k).
2. Mathematical Differential Privacy (epsilon-DP via Laplace Mechanism): Protects individual query responses by adding calibrated noise (b = Delta f / epsilon), providing formal mathematical bounds independent of adversary background knowledge.

The gateway incorporates cybersecurity controls modeled directly on NIS2 Art. 21(2):
- Tamper-Evident SHA-256 Hash Chained Audit Log (Art. 21(2)(h))
- Abuse Detection and Automatic Token Revocation (Art. 21(2)(b))
- Automated Regulatory Incident Dossier Generation:
  - NIS2 Art. 23: Early warning JSON (< 24h), incident notification (< 72h), and final report template (< 1 month).
  - GDPR Art. 33: Personal data breach notification JSON (< 72h) directed to the Portuguese DPA (CNPD).

## Commands to Run the Project and Generate PNG Figures

Here are the 5 terminal commands to run the tests, execute the end-to-end demonstration, recalculate benchmarks, generate the publication PNG figures, and start the interactive web application:

1. Install dependencies:
python -m pip install -r requirements.txt

2. Run automated test suite:
python -m pytest -v

3. Execute end-to-end demo and verify all security and regulatory rules:
python run_demo.py

4. Run all benchmarks and generate/download PNG plots into data/figures/:
python -m src.experiments.run_all_benchmarks

5. Launch the interactive Web Dashboard and REST API:
python run_server.py

## Generated Artifacts and Plots

After running command 4, the high-resolution publication PNG figures are saved in the data/figures/ directory:
- data/figures/figure_2_combined.png: (a) Error of DP Mean vs Privacy Budget (epsilon) for all patients (N=1000) and small oncology group (N=64); (b) k-Anonymity information loss (NCP) and maximum linkage risk vs k.
- data/figures/figure_3_range_queries.png: Error of range-count queries comparing Differential Privacy budget decay vs k-Anonymity constant error.
- data/figures/figure_4_nis2_timeline.png: NIS2 Art. 23 incident reporting steps and GDPR Art. 33 reporting alignment.

CSV benchmark outputs are available in data/:
- data/benchmark_dp_table12.csv: Differential Privacy Mean Blood Pressure Error (Table 12).
- data/benchmark_kanon_table13.csv: k-Anonymity Results for Quasi-Identifiers (Table 13).
- data/benchmark_range_queries.csv: Sequence range query error comparison.

## System Architecture and Rules

The Privacy Gateway enforces the following Table 8 rules on every interaction:
- Rule 1 (Authentication): Bearer tokens verified against identity registry.
- Rule 2 (Revocation): Revoked tokens rejected immediately.
- Rule 3 (Budget Enforcement): Requested query epsilon checked against remaining budget epsilon_rem.
- Rule 4 (Rate Limiting and Reconstruction Abuse Detection): Limits queries to <= 10 per minute; detects repeated adversarial query shifts. Triggers immediate token revocation and incident dossier generation.
- Rule 5 (Small-Cell Suppression): Blocks exact microdata filters yielding < 5 records to prevent singling out.
- Rule 6 (Sensitive Leakage Check): Researchers cannot access unanonymized raw clinical attributes.
- Rule 7 (Tamper-Evident Chaining): Every event is hashed and chained into audit_log.jsonl using SHA-256.
- Rule 8 (Automated Regulatory Incident Dossiers): Generates structured JSON files compliant with NIS2 Art. 23 and GDPR Art. 33.
