# 09 End-to-End Pipeline – Komplett ML-process

**Nivå:** 2  
**Tid:** 120-180 minuter  
**Fokus:** Kvalitetssäkring, Idrifttagande, Underhåll, Riskbedömning, Deployment  
**Dataset:** Swedish Student Data, Iris, Titanic

## Lärandemål

Efter detta projekt kan du:
- Bygga en komplett ML-pipeline från datainsamling till deployment
- Implementera kvalitetssäkring och testning av AI-system
- Skapa riskbedömning enligt EU AI Act
- Deploya en modell som Streamlit-app
- Förklara underhåll och övervakning av AI-system

## Koppling till Skolverket Gy25

| Centralt innehåll | Täckning |
|------------------|----------|
| Processen för att utveckla och kvalitetssäkra AI | ✅ Huvudfokus |
| Utvärdering och optimering av AI-system | ✅ Pipeline-steg |
| Lagar och regler om AI (EU AI Act, GDPR) | ✅ Riskbedömning |

## 🇸🇪 Svenska sammanhang

- **Skolverket**: Antagningssystem med fullständig dokumentation
- **Trafikverket**: AI deployment för trafikprediktion
- **Sjukvård**: Diagnostiksystem med regulatoriskt godkännande

## 🇪🇺 EU AI Act-koppling

- **Artikel 9**: Riskhanteringssystem
- **Artikel 10**: Datahantering och datastyrning
- **Artikel 11**: Teknisk dokumentation
- **Artikel 14**: Transparens och förklarbarhet

---

## Snabbstart

```bash
pip install scikit-learn pandas numpy streamlit joblib

streamlit run micro-projects/09-end-to-end-pipeline/app.py
```

---

## Introduktion

### Vad är en End-to-End Pipeline?

En ML-pipeline är en serie steg som transformerar rådata till en produktionsklar modell:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE OVERVIEW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  Data    │───▶│  Prepro- │───▶│   Train   │───▶│ Evaluate │         │
│  │ Ingestion│    │  cessing │    │  Model   │    │  Model   │         │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│                                                        │                │
│                                                        ▼                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │ Monitor  │◀───│ Deploy   │◀───│  Risk    │◀───│ Document │         │
│  │ System   │    │  Model   │    │Assessment│    │  ation   │         │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Steg 1: Datahantering

### Ladda och validera data

```python
import pandas as pd
import numpy as np

def load_data(filepath):
    """Ladda data med validering."""
    df = pd.read_csv(filepath)
    
    # Validera datakvalitet
    assert df.isnull().sum().sum() == 0, "Finns saknade värden!"
    assert len(df) > 100, "För få datapunkter!"
    
    return df

def split_data(df, target_column, test_size=0.2):
    """Dela data i träning och test."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=42)
```

### Datalagring och spårbarhet

```python
import json
from datetime import datetime

def log_data_info(df, filepath="data_log.json"):
    """Logga datainformation för spårbarhet."""
    log = {
        "timestamp": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    
    with open(filepath, 'w') as f:
        json.dump(log, f, indent=2)
    
    return log
```

---

## Steg 2: Förbehandling

### Pipeline med sklearn

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Definiera kolumner
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'country']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Kombinera
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

---

## Steg 3: Modellträning

### Modellval och hyperparameteroptimering

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

def train_and_optimize(X_train, y_train):
    """Träna och optimera flera modeller."""
    
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Hyperparameters
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        if name in param_grids:
            grid = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
            grid.fit(X_train, y_train)
            results[name] = {
                'score': grid.best_score_,
                'params': grid.best_params_,
                'model': grid.best_estimator_
            }
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            results[name] = {'score': score, 'model': model}
    
    return results, best_model
```

---

## Steg 4: Utvärdering

### Kvalitetsmetrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score,
                             classification_report)

def evaluate_model(model, X_test, y_test):
    """Utvärdera modell med flera metrics."""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    if y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    # Klassificeringsrapport
    results['classification_report'] = classification_report(y_test, y_pred)
    
    return results
```

### Cross-validation

```python
from sklearn.model_selection import cross_val_score

def cross_validate_model(model, X, y, cv=5):
    """Korsvalidera modellen."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return {
        'cv_scores': scores.tolist(),
        'cv_mean': scores.mean(),
        'cv_std': scores.std()
    }
```

---

## Steg 5: Riskbedömning (EU AI Act)

### Riskbedömningsramverk

```python
def eu_ai_act_risk_assessment(system_type, domain, impact_level):
    """
    Riskbedömning enligt EU AI Act.
    
    Args:
        system_type: Typ av AI-system
        domain: Användningsdomän (utbildning, sjukvård, etc.)
        impact_level: Hög/Medel/Låg påverkan
    """
    
    high_risk_domains = [
        'utbildning',      # Education
        'arbetsmarknad',   # Employment
        'sjukvård',        # Healthcare
        'rättsväsende',   # Justice
        'migration'        # Migration
    ]
    
    is_high_risk = domain.lower() in high_risk_domains or impact_level == 'Hög'
    
    assessment = {
        'system_type': system_type,
        'domain': domain,
        'impact_level': impact_level,
        'high_risk': is_high_risk,
        'required_actions': []
    }
    
    if is_high_risk:
        assessment['required_actions'] = [
            "Riskbedömning före idrifttagande",
            "Datahanteringsplan",
            "Teknisk dokumentation",
            "CE-märkning",
            "Kvalitetshanteringssystem",
            "Mänsklig översyn",
            "Loggning och övervakning",
            "Incidentrapportering"
        ]
    
    return assessment
```

### GDPR-checklist

```python
def gdpr_compliance_check(uses_personal_data, automated_decisions, profiling):
    """GDPR-efterlevnadscheck."""
    
    checklist = []
    
    if uses_personal_data:
        checklist.append({
            'requirement': 'Artikel 5 - Princip för laglighet',
            'status': 'Måste dokumenteras',
            'completed': False
        })
    
    if automated_decisions:
        checklist.append({
            'requirement': 'Artikel 22 - Automatiserat beslutsfattande',
            'status': 'Rätt till manuell granskning krävs',
            'completed': False
        })
    
    if profiling:
        checklist.append({
            'requirement': 'Artikel 13 - Rätt till information',
            'status': 'Transparenskrav',
            'completed': False
        })
    
    return checklist
```

---

## Steg 6: Deployment

### Spara modellen

```python
import joblib

def save_model(model, preprocessor, filepath_prefix="model"):
    """Spara modell och preprocessor."""
    
    # Spara modell
    joblib.dump(model, f"{filepath_prefix}.joblib")
    
    # Spara preprocessor
    joblib.dump(preprocessor, f"{filepath_prefix}_preprocessor.joblib")
    
    # Spara metadata
    metadata = {
        'model_type': type(model).__name__,
        'created_at': datetime.now().isoformat(),
        'sklearn_version': sklearn.__version__
    }
    
    with open(f"{filepath_prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Modell sparad: {filepath_prefix}.joblib")
```

### Streamlit Deployment App

```python
# Se app.py för komplett kod
import streamlit as st

st.title("🚀 ML Model Deployment")
st.write("Ladda upp din tränade modell och gör förutsägelser!")

uploaded_file = st.file_uploader("Ladda upp modell (.joblib)", type="joblib")

if uploaded_file:
    model = joblib.load(uploaded_file)
    st.success("Modell laddad!")
    
    # Input form
    features = st.text_input("Ange features (kommaseparerat)")
    
    if st.button("Förutsäg"):
        # Gör prediction
        result = model.predict([features])
        st.write(f"Resultat: {result}")
```

---

## Steg 7: Övervakning och Underhåll

### Monitoring Dashboard

```python
def log_prediction(input_data, prediction, timestamp):
    """Logga varje prediction för övervakning."""
    
    log_entry = {
        'timestamp': timestamp,
        'input': input_data,
        'prediction': prediction
    }
    
    # Append to log file
    with open('predictions_log.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def calculate_drift(baseline_data, current_data, threshold=0.1):
    """Beräkna data drift."""
    
    # Jämför fördelningar
    baseline_mean = baseline_data.mean()
    current_mean = current_data.mean()
    
    drift = abs(current_mean - baseline_mean) / baseline_mean
    
    return {
        'drift_detected': drift > threshold,
        'drift_percentage': drift,
        'threshold': threshold
    }
```

---

## Reflektionsfrågor

### Grundläggande (E-nivå)
1. Vilka är de viktigaste stegen i en ML-pipeline?
2. Varför är det viktigt att dokumentera varje steg?
3. Vad är skillnaden mellan träningsdata och produktionsdata?

### Utmanande (C-nivå)
1. Hur säkerställer du att modellen fungerar lika bra i produktion som i träning?
2. Vad innebär det att ett AI-system är "högrisk" enligt EU AI Act?
3. Hur ofta bör en modell omtränas?

### Avancerat (A-nivå)
1. Designa en komplett riskbedömning för ett antagningssystem.
2. Hur implementerar du kontinuerlig övervakning av en produktionsmodell?
3. Vilka etiska överväganden finns vid deployment av AI i samhället?

---

## 🇸🇪 Svenska facktermer

| Engelska | Svenska |
|----------|---------|
| Pipeline | Rörledning / Flöde |
| Deployment | Idrifttagande |
| Monitoring | Övervakning |
| Risk assessment | Riskbedömning |
| Documentation | Dokumentation |
| Quality assurance | Kvalitetssäkring |

---

## Nästa steg

- **Projekt 10**: Domain Capstone – tillämpa i valfri domän

---

*Projektet är en del av Gy25 AI Curriculum – Artificiell Intelligens för Svenska Gymnasium*
