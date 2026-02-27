# 07 Bias & Fairness Simulator – AI, Etik och EU AI Act

**Nivå:** 1-2  
**Tid:** 90-120 minuter  
**Fokus:** Bias-detektering, Fairness metrics, EU AI Act, GDPR  
**Dataset:** Swedish Student Data (simulerad)

## Lärandemål

Efter detta projekt kan du:
- Identifiera bias i dataset och modeller
- Använda fairness metrics (demographic parity, equalized odds)
- Förklara EU AI Act och dess krav
- Diskutera GDPR och integritetsskydd

## Koppling till Skolverket Gy25

| Centralt innehåll | Täckning |
|------------------|----------|
| Normer, lagar och bestämmelser (EU AI Act, GDPR) | ✅ Huvudfokus |
| Etiska dilemman (transparens, rättvisa) | ✅ Centralt |
| Demokratiska, sociala, ekonomiska risker | ✅ Diskuteras |

## 🇸🇪 Svenska sammanhang

- **SCB**: Statistiska centralbyrån, dataskydd
- **Skolverket**: Betygsbedömning, meritvärden
- **Trafikverket**: AI för trafficprediction

## 🇪🇺 EU AI Act-koppling

- **Kapitel 2**: Högrisk-AI-system (utbildning, arbetsmarknad)
- **Artikel 6**: Klassificering av hög risk
- **Bilaga III**: Specifika högrisktillämpningar

---

## Snabbstart

```bash
# Installera extra beroenden
pip install aif360 seaborn

# Starta appen
streamlit run micro-projects/07-bias-fairness-simulator/app.py
```

---

## Introduktion

### Vad är AI-bias?

**AI-bias** uppstår när AI-system behandlar olika grupper ojämlikt, ofta på grund av 
historisk data som reflekterar befintliga ojämlikheter.

### Varför är det viktigt?

```
┌────────────────────────────────────────────────────────────┐
│                    EXEMPEL PÅ AI-BIAS                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  📱 Rekrytering: AI som diskriminerar kvinnor             │
│     (tränad på historiskt mansdominerade data)            │
│                                                            │
│  🏦 Krediter: Minoriteter får sämre lånevillkor          │
│     (bias i inkomstdata)                                  │
│                                                            │
│  🏥 Sjukvård: Sämre diagnoser för vissa grupper          │
│     (underrepresenterade i träningsdata)                  │
│                                                            │
│  👮 Rättsväsende: Rasprofileringsrisk                      │
│     (historisk polisdata)                                 │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Steg 1: Installera och ladda data

### Installera AI Fairness 360

```bash
pip install aif360
```

### Ladda Swedish Student Data (simulerad)

```python
import pandas as pd
import numpy as np

# Simulerad dataset med svenska elever
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'student_id': range(1, n+1),
    'gender': np.random.choice(['M', 'F'], n, p=[0.48, 0.52]),
    'birth_country': np.random.choice(['Sweden', 'Other'], n, p=[0.75, 0.25]),
    'parent_education': np.random.choice(['High', 'Medium', 'Low'], n, p=[0.3, 0.4, 0.3]),
    'study_hours': np.random.normal(15, 5, n).clip(0, 40),
    'math_grade': np.random.normal(75, 15, n).clip(0, 100),
    'english_grade': np.random.normal(72, 15, n).clip(0, 100),
    'admitted': np.zeros(n)
})

# Skapa bias: grupp med lägre förälderutbildning får sämre betyg
data.loc[data['parent_education'] == 'Low', 'math_grade'] -= 5
data.loc[data['parent_education'] == 'Low', 'english_grade'] -= 5

# Admission baserat på betyg + hidden bias
threshold = 60
data['admitted'] = ((data['math_grade'] + data['english_grade']) / 2 > threshold).astype(int)

print(f"Dataset: {len(data)} elever")
print(data.head())
```

---

## Steg 2: Träna en modell

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Förbered features
le_gender = LabelEncoder()
le_country = LabelEncoder()
le_edu = LabelEncoder()

data['gender_enc'] = le_gender.fit_transform(data['gender'])
data['country_enc'] = le_country.fit_transform(data['birth_country'])
data['edu_enc'] = le_edu.fit_transform(data['parent_education'])

X = data[['study_hours', 'math_grade', 'english_grade', 'gender_enc', 'country_enc', 'edu_enc']]
y = data['admitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Träna modell
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Model accuracy: {np.mean(y_pred == y_test):.2%}")
```

---

## Steg 3: Mäta Bias

### Definiera skyddade grupper

```python
# Skyddad grupp: birth_country == 'Other'
protected = 'birth_country'
privileged = 'Sweden'
unprivileged = 'Other'

# Lägg till prediction till dataframe
data['prediction'] = model.predict(X)
```

### Beräkna Fairness Metrics

```python
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset

# Skapa aif360 dataset
df_aif = data.copy()
df_aif['label'] = df_aif['admitted']

dataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df_aif,
    label_names=['label'],
    protected_attribute_names=['birth_country']
)

# Beräkna metrics
metric = BinaryLabelDatasetMetric(
    dataset,
    unprivileged_groups=[{'birth_country': 1}],  # Other
    privileged_groups=[{'birth_country': 0}]     # Sweden
)

print("=== Fairness Metrics ===")
print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.3f}")
print(f"Disparate Impact: {metric.disparate_impact():.3f}")
print(f"Equalized Odds Difference: {metric.equalized_odds_difference():.3f}")
```

---

## Steg 4: Förklara Fairness Metrics

### Vad betyder de?

| Metric | Beskrivning | Ideal värde |
|--------|-------------|-------------|
| **Statistical Parity Difference** | Skillnad i positive rate mellan grupper | 0.0 |
| **Disparate Impact** | Ratio av positive rate (privilegierad / icke-privilegierad) | 1.0 |
| **Equalized Odds** | Samma accuracy för alla grupper | 0.0 |

### Visualisera bias

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Jämför antagningsgrad per grupp
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Per födelseland
admission_by_country = data.groupby('birth_country')['admitted'].mean()
axes[0].bar(admission_by_country.index, admission_by_country.values)
axes[0].set_title('Antagningsgrad per födelseland')
axes[0].set_ylabel('Andel antagna')

# Per kön
admission_by_gender = data.groupby('gender')['admitted'].mean()
axes[1].bar(admission_by_gender.index, admission_by_gender.values)
axes[1].set_title('Antagningsgrad per kön')
axes[1].set_ylabel('Andel antagna')

plt.tight_layout()
plt.show()
```

---

## Steg 5: Mitigera Bias

### Technique 1: Pre-processing

```python
# Reweighing - ändra vikterna i träningsdata
from aif360.algorithms.preprocessing import Reweighing

# Definiera privileged/unprivileged
privileged_groups = [{'birth_country': 0}]
unprivileged_groups = [{'birth_country': 1}]

rw = Reweighing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

dataset_transf = rw.fit_transform(dataset)

# Träna om med transformerad data
# ... (implementation details)
```

### Technique 2: In-processing

```python
# Adversarial debiasing - träna modell som är "blind" för skyddad attr
# Se aif360.algorithms.inprocessing
```

### Technique 3: Post-processing

```python
# Threshold adjustment - justera beslutsgränsen
# Ge mer fördelaktiga thresholds till underrepresenterade grupper
```

---

## 🇪🇺 EU AI Act – Det du behöver veta

### Högrisk-system (Kapitel 2, Artikel 6)

AI-system som används inom följande områden är **högrisk**:

| Område | Exempel |
|--------|---------|
| **Utbildning** | Antagningssystem, betygsbedömning |
| **Arbetsmarknad** | Rekrytering, befordran |
| **Grundläggande tjänster** | Krediter, försäkringar |
| **Rättsväsende** | Riskbedömning |
| **Migration** | Asylbedömning |

### Krav för högrisk-system

```
┌────────────────────────────────────────────────────────────┐
│                  EU AI ACT KRAV                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Riskbedömning innan deployment                         │
│  2. Datahantering (kvalitet, representativitet)          │
│  3. Teknisk dokumentation                                 │
│  4. Loggning och spårbarhet                              │
│  5. Transparens (till användare)                         │
│  6. Mänsklig översyn                                      │
│  7. Noggrannhet, robusthet, säkerhet                      │
│  8. CE-märkning                                           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🇸🇪 GDPR och AI

### Artikel 22 – Automatiserat beslutsfattande

> *"Den registrerade har rätt att inte bli föremål för ett beslut 
> som enbart grundas på automatiserad behandling ... och som 
> har rättsliga följder eller på liknande sätt betydande påverkan."*

### Praktiska konsekvenser

| Situation | Krav |
|-----------|------|
| Antagning till gymnasiet | Måste finnas mänsklig granskning |
| Kreditbedömning | Måste kunna begära manuell översyn |
| AI-granskning av arbeten | Elever måste få veta hur det går till |

---

## Reflektionsfrågor

### Grundläggande (E-nivå)

1. **Vad är AI-bias?** Ge ett eget exempel.
2. **Varför är det viktigt att titta på vem som INTE är med i datasetet?
3. **Vad säger EU AI Act om antagningssystem?**

### Utmanande (C-nivå)

1. **Vad är skillnaden mellan statistical parity och equalized odds?**
2. **Hur kan du testa om din modell har bias?**
3. **Vilka grupper är extra sårbara för AI-bias i Sverige?**

### Avancerat (A-nivå)

1. **Hur balanserar man fairness mot noggrannhet?**
2. **Vad är "proxy discrimination" och hur undviker man det?**
3. **Hur implementerar man "right to explanation" enligt GDPR?**

---

## 🇸🇪 Svenska facktermer

| Engelska | Svenska |
|----------|---------|
| Bias | Partiskhet / snedvridning |
| Fairness | Rättvisa |
| Protected attribute | Skyddad egenskap |
| Disparate impact | Ojämlik påverkan |
| Statistical parity | Statistisk jämlikhet |
| Equalized odds | Lika odds |
| Demographic parity | Demografisk jämlikhet |

---

## Nästa steg

- **Projekt 08**: Neural Net from Scratch – bygg ditt eget nätverk
- **Projekt 09**: End-to-End Pipeline – deployment och riskbedömning
- **Projekt 10**: Domain Capstone – tillämpa i valfri domän

---

## Resurser

- [AI Fairness 360](https://aif360.readthedocs.io/)
- [EU AI Act Official](https://artificialintelligenceact.eu/)
- [GDPR Article 22](https://gdpr-info.eu/art-22-gdpr/)
- [SCB - Statistikdatabasen](https://www.scb.se/)

---

*Projektet är en del av Gy25 AI Curriculum – Artificiell Intelligens för Svenska Gymnasium*
