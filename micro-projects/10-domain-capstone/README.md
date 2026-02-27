# 10 Domain Capstone – Självvalt Domänprojekt

**Nivå:** 2  
**Tid:** 180-240 minuter  
**Fokus:** Självvald domän, fullständig AI-lösning, Riskrapport, EU AI Act Compliance  
**Domäner:** Svenska trafik, Sjukvård, Finans, Utbildning

## Lärandemål

Efter detta projekt kan du:
- Tillämpa alla kunskaper från tidigare projekt i ett realistiskt scenario
- Bygga en komplett AI-lösning från idé till prototype
- Skapa en riskbedömning enligt EU AI Act
- Förklara och reflektera kring AI:s roll i samhället
- Presenter din lösning för andra

## Koppling till Skolverket Gy25

| Centralt innehåll | Täckning |
|------------------|----------|
| Användningsområden inom valt område | ✅ Huvudfokus |
| Nyanserade och välgrundade resonemang om möjligheter och risker | ✅ Riskrapport |
| Kritiskt förhållningssätt till AI:s begränsningar | ✅ Reflektion |
| Konsekvenser av lagar och bestämmelser | ✅ EU AI Act |

---

## Snabbstart

```bash
# Välj din domän och börja bygga!
# Använd allt du lärt dig från projekt 01-09

streamlit run micro-projects/10-domain-capstone/app.py
```

---

## Välj din domän

### 🇸🇪 Domän 1: Trafik och Transport (Trafikverket)

**Projektförslag:**
- Trafikprediktion för Stockholm/Göteborg
- Kollektivtrafik-optimering
- Fordonsklassificering för vägtullar
- Trafiksäkerhetsanalys

**AI-tekniker:**
- Tidsserieprediktion (LSTM, Prophet)
- Bildigenkänning (CNN)
- Reinforcement Learning (trafikljusoptimering)

**EU AI Act-koppling:**
- Högrisk: Självkörande fordon
- Dokumentation: Trafiksäkerhet

### 🏥 Domän 2: Sjukvård och Hälsa

**Projektförslag:**
- Sjukdomsprediktion baserat på symptom
- Medicinsk bildanalys (röntgen, MR)
- Patientriskbedömning
- Läkemedelsinteraktionsprediktion

**AI-tekniker:**
- Klassificering (Random Forest, XGBoost)
- Bildanalys (CNN)
- Förklarbar AI (SHAP, LIME)

**EU AI Act-koppling:**
- Högrisk: Medicinteknisk utrustning
- GDPR: Känslig hälsoinformation

### 💰 Domän 3: Finans och Bank

**Projektförslag:**
- Kreditbedömning
- Bedrägeridetektering
- Aktiemarknads-prediktion
- Försäkringsriskbedömning

**AI-tekniker:**
- Klassificering
- Anomalidetektering
- Tidsserieanalys

**EU AI Act-koppling:**
- Högrisk: Grundläggande tjänster
- GDPR: Automatiserade beslut (Art. 22)

### 📚 Domän 4: Utbildning (Skolverket)

**Projektförslag:**
- Studieprstationsprediktion
- Antagningssystem
- Lärandestilsanalys
- Betygsbedömning

**AI-tekniker:**
- Klassificering
- Clustering
- NLP för textanalys

**EU AI Act-koppling:**
- Högrisk: Utbildning
- GDPR: Elevdata

---

## Capstone-struktur

### Steg 1: Problembeskrivning (30 min)

```markdown
# Problemformulering

## Vald domän: [Trafik/Sjukvård/Finans/Utbildning]

## Problem:
[Vad vill du lösa?]

## Varför är detta viktigt?
[Påverkan på samhälle, företag, individer]

## Målgrupp:
[Vem påverkas av lösningen?]
```

### Steg 2: Datainsamling (30 min)

```python
# Exempel: Trafikprediktion
import pandas as pd

# Samla data från Trafikverket API
data = pd.read_csv('traffic_data.csv')

# Validera kvalitet
print(f"Rader: {len(data)}")
print(f"Kolumner: {list(data.columns)}")
print(f"Saknade värden: {data.isnull().sum().sum()}")
```

### Steg 3: Modellbyggande (60 min)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Förbered data
X = data.drop(columns=['target'])
y = data['target']

# Dela data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Träna modell
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Utvärdera
accuracy = model.score(X_test, y_test)
print(f"Noggrannhet: {accuracy:.2%}")
```

### Steg 4: Riskbedömning (30 min)

```markdown
# Riskbedömning enligt EU AI Act

## 1. Systemöversikt
- Domän: [Vald domän]
- Användningsområde: [Beskrivning]
- Målgrupp: [Vem påverkas]

## 2. Riskklassificering
- Är systemet högrisk? [Ja/Nej]
- Motivering: [Förklaring]

## 3. Krav och åtgärder
| Krav | Status | Åtgärd |
|------|--------|--------|
| Datahantering | | |
| Dokumentation | | |
| Transparens | | |
| Mänsklig översyn | | |

## 4. GDPR-efterlevnad
- Personuppgifter: [Ja/Nej]
- Automatiserade beslut: [Ja/Nej]
- Rätt till granskning: [Ja/Nej]
```

### Steg 5: Reflektion (30 min)

```markdown
# Reflektion

## Vad har du lärt dig?
[Sammanfattning av kunskaper]

## Vilka etiska frågor väcktes?
[Reflektion kring etik]

## Vad skulle du göra annorlunda?
[Förbättringsförslag]

## Hur påverkar detta samhället?
[Samhällspåverkan]
```

---

## Bedömningskriterier (E/C/A)

### E-nivå
- [ ] Valt en domän och beskrivit problemet
- [ ] Samlat och förbehandlat data
- [ ] Tränat en fungerande modell
- [ ] Gjort en grundläggande riskbedömning

### C-nivå
- [ ] Alla E-krav uppfyllda
- [ ] Jämfört flera modeller
- [ ] Tillämpat SHAP eller annan förklaringsmetod
- [ ] Skapat en komplett riskrapport

### A-nivå
- [ ] Alla C-krav uppfyllda
- [ ] Implementerat en interaktiv prototype
- [ ] Gjort nyanserade etiska resonemang
- [ ] Presenterat lösningen för andra

---

## Projektmall

### Portfolio Elements

1. **Problemformulering** (1 sida)
2. **Databeskrivning** (1 sida)
3. **Modell och resultat** (2 sidor)
4. **Riskbedömning** (1-2 sidor)
5. **Etisk reflektion** (1 sida)
6. **Demo/interaktiv prototype**

### Presentation (10 minuter)

1. Problemet och varför det är viktigt (2 min)
2. Din lösning och tekniker (4 min)
3. Riskbedömning och etik (2 min)
4. Frågor och diskussion (2 min)

---

## Resurser och Stöd

### Datakällor (Sverige)
- Trafikverket öppna data
- SCB statistikdatabas
- Socialstyrelsen
- Finansinspektionen

### Verktyg
- Streamlit för interaktiva demos
- SHAP för förklaringar
- scikit-learn för modellering

### Mallar
- Jämförelsematris (comparison-matrix-template.md)
- Etiskt ramverk (ethics-framework.md)
- Riskbedömningsmall (ovan)

---

## Exempel: Trafikprediktion (Färdigt exempel)

### Problem
Förutsäga trafikbelastning på E4:an utanför Stockholm för att optimera trafikljus.

### Data
- Historisk trafikdata från Trafikverket
- Väderdata (SMHI)
- Tid på dygnet, veckodag

### Modell
- RandomForest för klassificering
- SHAP för förklaring

### Riskbedömning
- Ej högrisk (ej direkt patientsäkerhet)
- GDPR: Anonymiserad data
- Transparens: SHAP-värden visas

---

*Projektet är en del av Gy25 AI Curriculum – Artificiell Intelligens för Svenska Gymnasium*
