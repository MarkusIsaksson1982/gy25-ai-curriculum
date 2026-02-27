# Gy25 AI Curriculum – v1.2 (Nivå 2 COMPLETE)

A complete, ready-to-use Python codebase for teaching Artificial Intelligence in Swedish gymnasium (Nivå 1 & Nivå 2), aligned with Skolverket's curriculum.

## Overview

This project provides **10 self-contained micro-projects** (each 45-240 min) that map 1-to-1 to the Skolverket centralt innehåll for Artificiell intelligens. Projects supplement each other: Level 1 builds foundations (data, simple ML, ethics); Level 2 reuses the same code/datasets and extends them (neural nets, full pipelines, domain applications, explainability, risk assessment).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/MarkusIsaksson1982/gy25-ai-curriculum.git
cd gy25-ai-curriculum

# Install dependencies
pip install -r requirements.txt

# Launch the AI Explorer dashboard
streamlit run apps/ai_explorer.py
```

## Project Structure

```
gy25-ai-curriculum/
├── README.md
├── requirements.txt
├── setup.md
├── curriculum-mapping.md     # Master table: micro-projects ↔ Skolverket content
├── teacher-guide.md
├── student-guide.md
├── grading-rubrics.md        # E/C/A aligned per project
├── comparison-matrix-template.md
├── ethics-framework.md
├── shared/
│   ├── datasets/             # Titanic, Iris, MNIST, SCB-proxy, Swedish traffic
│   ├── utils.py              # metrics, viz, reflection templates
│   └── templates/
├── micro-projects/           # 10 self-contained folders
│   ├── 01-data-detective/
│   ├── 02-decision-tree-detective/
│   ├── 03-search-game-agent/
│   ├── 04-sentiment-nlp-lab/
│   ├── 05-mnist-vision-basics/
│   ├── 06-simple-rl-robot/
│   ├── 07-bias-fairness-simulator/
│   ├── 08-neural-net-scratch/
│   ├── 09-end-to-end-pipeline/
│   └── 10-domain-capstone/
├── apps/                     # Overarching Streamlit dashboard
│   └── ai_explorer.py
└── assessments/              # Quizzes, rubrics, portfolio template
```

## Micro-Projects (ALL 10 COMPLETE)

| # | Project | Level | Focus |
|---|---------|-------|-------|
| 01 | Data Detective | 1 | Data quality, bias, cleaning |
| 02 | Decision Tree Detective | 1 | Classification, transparency |
| 03 | Search & Game Agent | 1 | Search algorithms, game AI |
| 04 | Sentiment NLP Lab | 1 | NLP, generative AI overview |
| 05 | MNIST Vision Basics | 1-2 | Object recognition, CNN intro |
| 06 | Simple RL Robot | 1-2 | Reinforcement learning, CartPole |
| 07 | Bias & Fairness Simulator | 1-2 | EU AI Act, GDPR, fairness metrics |
| 08 | Neural Net from Scratch | 2 | MLP, SHAP, explainability |
| 09 | End-to-End Pipeline | 2 | Deployment, risk assessment |
| 10 | Domain Capstone | 2 | Swedish traffic/healthcare/finance/education |

## Features

- **Standalone micro-projects**: Clone → `pip install` → run
- **Streamlit dashboards**: Interactive training sliders, decision-tree visualizations
- **Swedish context**: EU AI Act, GDPR, Trafikverket/SCB examples
- **Built-in reflection**: Comparison-matrix exercises, human-AI comparisons
- **Assessment-ready**: E/C/A rubrics, quizzes, portfolio templates

## Nivå 2 (ARTI2000X) - Project Details

### 05 MNIST Vision Basics
- **Focus**: Object recognition, CNN introduction
- **Skills**: Image classification, convolutional neural networks
- **Swedish context**: Trafikverket vehicle detection, BankID

### 06 Simple RL Robot
- **Focus**: Reinforcement learning, Q-learning
- **Skills**: Training agents, reward functions
- **Swedish context**: Volvo self-driving, traffic optimization

### 07 Bias & Fairness Simulator
- **Focus**: EU AI Act, GDPR, fairness metrics
- **Skills**: Bias detection, demographic parity
- **Swedish context**: SCB data, Skolverket admissions

### 08 Neural Net from Scratch
- **Focus**: MLP architecture, SHAP explainability
- **Skills**: Building NNs from scratch
- **Swedish context**: Healthcare diagnostics, banking

### 09 End-to-End Pipeline
- **Focus**: ML pipeline, deployment, risk assessment
- **Skills**: Full ML lifecycle
- **Swedish context**: Production AI systems

### 10 Domain Capstone
- **Focus**: Self-selected domain application
- **Skills**: Complete AI solution
- **Swedish context**: Trafikverket, Healthcare, Finance, Education

---

# Gy25 AI Curriculum – v1.2 (Nivå 2 COMPLETE)
**FULL CURRICULUM READY** – 10 micro-projects for Swedish gymnasium

## Status

- ✅ Nivå 1: Projects 01-04 COMPLETE
- ✅ Nivå 2a: Projects 05-08 COMPLETE
- ✅ Nivå 2b: Projects 09-10 COMPLETE
- ✅ FULL CURRICULUM: 10/10 PROJECTS READY

---

*For questions or contributions, please contact the maintainer or open an issue.*
