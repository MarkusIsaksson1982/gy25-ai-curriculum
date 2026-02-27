# Teacher Guide - Gy25 AI Curriculum

## Getting Started

### Prerequisites
- Python 3.8+ installed
- Basic familiarity with Jupyter notebooks and Streamlit
- Understanding of fundamental AI/ML concepts

### Installation
```bash
git clone https://github.com/MarkusIsaksson1982/gy25-ai-curriculum.git
cd gy25-ai-curriculum
pip install -r requirements.txt
```

### Recommended Lesson Flow

#### Week 1-2: Foundations (Data Detective)
- Introduce AI and machine learning concepts
- Start with 01-data-detective project
- Focus on data quality and bias

#### Week 3-4: Basic ML (Decision Trees)
- Move to 02-decision-tree-detective
- Compare human vs AI decision-making
- Introduce transparency concepts

#### Week 5-6: Problem Solving (Search & Game Agent)
- Move to **03-search-game-agent**
- Let students play against the AI
- Discuss: "När är AI överlägsen människan?" (perfect information games)

#### Week 7-8: Vision and NLP
- **05 MNIST Vision Basics** - Introduction to CNNs and image classification
- **04 Sentiment NLP Lab** - Text analysis and generative AI overview

#### Week 9-10: Advanced Topics
- **06 Simple RL Robot** - Reinforcement learning with Q-learning
- **07 Bias & Fairness Simulator** - EU AI Act, GDPR, fairness metrics

#### Week 11-12: Deep Learning & Explainability
- **08 Neural Net from Scratch** - Build MLP from ground up + SHAP

#### Week 13-14: Deployment & Capstone
- **09 End-to-End Pipeline** - Model deployment and risk assessment
- **10 Domain Capstone** - Student-selected domain application

## Lesson Templates (45 min each)

### Typical Lesson Structure
1. **Intro (5 min)**: Connect to previous lesson, present today's goal
2. **Demonstration (10 min)**: Live demo of concept
3. **Hands-on (25 min)**: Students work on micro-project
4. **Reflection (5 min)**: Discussion, comparison matrix updates

## Nivå 2 (ARTI2000X) - Special Focus

### Project 05: MNIST Vision Basics
- **Duration**: 90-120 minutes
- **Core concepts**: CNN, image classification, computer vision
- **Swedish context**: Trafikverket vehicle detection, BankID
- **Assessment**: Build working CNN, achieve >95% accuracy

### Project 06: Simple RL Robot
- **Duration**: 90-120 minutes
- **Core concepts**: Q-learning, reward functions, exploration vs exploitation
- **Swedish context**: Volvo self-driving research, traffic optimization
- **Assessment**: Train agent to solve CartPole

### Project 07: Bias & Fairness Simulator
- **Duration**: 90-120 minutes
- **Core concepts**: EU AI Act, GDPR, fairness metrics, bias detection
- **Swedish context**: SCB data protection, Skolverket admissions
- **Assessment**: Analyze dataset for bias, propose mitigations

### Project 08: Neural Net from Scratch + SHAP
- **Duration**: 120-180 minutes
- **Core concepts**: MLP architecture, forward/backward propagation, SHAP
- **Swedish context**: Healthcare diagnostics, banking decisions
- **Assessment**: Build working NN from scratch, explain predictions

## Differentiation

### For Struggling Students (L1 focus)
- Provide more scaffolding in notebooks
- Focus on E-level criteria
- Use pre-filled comparison matrices

### For Advanced Students (L2 extension)
- Challenge extensions in each project
- Encourage A-level portfolio work
- Extra: SHAP explanations, model deployment

## Assessment

### Formative
- In-project reflections
- Comparison matrix completion
- Streamlit demo observations

### Summative
- Use grading-rubrics.md for E/C/A alignment
- Portfolio presentation (capstone project)
- Written reflection on ethics

## Swedish Context

### EU AI Act Integration
- High-risk AI systems in education (Chapter 2, Article 6)
- Transparency requirements
- Risk assessment in capstone

### GDPR Considerations
- Automated decision-making (Article 22)
- Student data protection
- Fairness in grading AI

### Swedish Examples
- Trafikverket: Traffic prediction, self-driving vehicles
- Healthcare: Medical diagnosis AI (Karolinska)
- SCB: Statistical data analysis
- Volvo: Autonomous vehicles research
- BankID: Signature recognition

---

*For curriculum alignment details, see curriculum-mapping.md*
