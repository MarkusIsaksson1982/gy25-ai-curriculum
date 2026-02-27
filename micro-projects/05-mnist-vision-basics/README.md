# 05 MNIST Vision Basics – Objektigenkänning med CNN

**Nivå:** 1-2  
**Tid:** 90-120 minuter  
**Fokus:** Objektigenkänning, CNN-intro, Dataaugmentering  
**Dataset:** MNIST (handgeschriebene Ziffern)

## Lärandemål

Efter detta projekt kan du:
- Förklara hur ett convolutional neural network (CNN) fungerar
- Träna en bildklassificerare för handskrivna siffror
- Jämföra resultat med andra tekniker (k-NN, Decision Tree)
- Diskutera begränsningar och möjligheter med bildigenkänning

## Koppling till Skolverket Gy25

| Centralt innehåll | Täckning |
|------------------|----------|
| Översikt över användningen av AI (prediktion, robotik, vision, generativ AI) | ✅ Huvudfokus |
| Översikt av tekniker (objektigenkänning) | ✅ CNN |
| Vikten av data, datakvalitet för AI | ✅ Dataaugmentering |

##瑞典上下文

- **Trafikverket**: Objektigenkänning för fordonsdetektering
- **Healthcare**: Medicinsk bildanalys (röntgen, MRT)
- **BankID**: Handledesignaturigenkänning

## 🇪🇺 EU AI Act-koppling

- Högrisk-system för biomedicinsk bildanalys (bilaga III)
- Transparenskrav för ansiktsigenkänning
- Dokumentationskrav för träningsdata

---

## Snabbstart

```bash
# Installera extra beroenden
pip install tensorflow keras plotly

# Starta appen
streamlit run micro-projects/05-mnist-vision-basics/app.py
```

---

## Introduktion

### Vad är bildigenkänning?

Bildigenkänning (computer vision) är en del av AI som handlar om att få datorer att "se" och förstå bilder. Det används överallt:

- **Självkörande bilar**: Identifiera fotgängare, vägskyltar
- **Medicin**: Upptäcka tumörer i röntgenbilder
- **Ansiktsigenkänning**: BankID, säkerhetskontroller
- **Industri**: Kvalitetskontroll i tillverkning

### Hur fungerar CNN?

Ett **Convolutional Neural Network (CNN)** är speciellt utvecklat för att analysera bilder. Det fungerar så här:

1. **Input-lager**: Tar emot pixlarna från en bild (28x28 = 784 pixlar för MNIST)
2. **Convolutional lager**: Använder "filter" för att hitta mönster (kanter, former)
3. **Pooling-lager**: Komprimerar informationen utan att förlora viktiga features
4. **Fully connected lager**: Slutgiltig klassificering

### Varför MNIST?

MNIST är den "Hello World" för bildigenkänning:
- 70,000 bilder av handskrivna siffror (0-9)
- Gråskala, 28x28 pixlar
- Klassisk dataset för att lära sig CNN

---

## Steg 1: Ladda och utforska data

### Ladda MNIST

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

# Ladda data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"Träningsdata: {X_train.shape}")
print(f"Testdata: {X_test.shape}")
print(f"Antal klasser: {len(np.unique(y_train))}")
```

### Visualisera några exempel

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'Siffra: {y_train[i]}')
    ax.axis('off')
plt.suptitle('Exempel från MNIST-datasetet')
plt.show()
```

---

## Steg 2: Förbehandla data

### Normalisera pixlar

```python
# Normalisera till [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape för CNN (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Ny form: {X_train.shape}")
```

### Dataaugmentering

För att förbättra modellen kan vi skapa varianter av träningsbilder:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Detta genererar fler träningsexempel
# genom att rotera, förskjuta och zooma bilder
```

---

## Steg 3: Bygg CNN-modellen

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    # Första convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    # Andra convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Fully connected lager
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')  # 10 klasser (0-9)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### Modellens arkitektur

```
┌─────────────────────────────────────┐
│         Input (28x28x1)             │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Conv2D(32) + MaxPool + ReLU        │
│  32 filters, 3x3 kernel              │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Conv2D(64) + MaxPool + ReLU        │
│  64 filters, 3x3 kernel              │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Flatten + Dense(128) + Dropout     │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│         Output (10 klasser)          │
└─────────────────────────────────────┘
```

---

## Steg 4: Träna modellen

```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)
```

### Övervaka träningen

```python
import plotly.express as px

# Plot training history
fig = px.line(
    pd.DataFrame(history.history),
    y=['accuracy', 'val_accuracy'],
    labels={'index': 'Epoch', 'value': 'Accuracy'},
    title='Tränings- vs Validationsnoggrannhet'
)
fig.show()
```

---

## Steg 5: Utvärdera modellen

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Förutsägelser
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Klassificeringsrapport
print(classification_report(y_test, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### Resultat förväntade

| Metric | Värde |
|--------|-------|
| Test Accuracy | ~99% |
| Precision | ~99% |
| Recall | ~99% |

---

## Steg 6: Testa modellen

```python
# Välj en bild från testset
index = 42
img = X_test[index]
true_label = y_test[index]

# Prediktera
prediction = model.predict(img.reshape(1, 28, 28, 1))
predicted_class = np.argmax(prediction)

plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f'Sann: {true_label}, Predikterad: {predicted_class}')
plt.show()
```

### Interaktiv demo i Streamlit

I `app.py` kan du rita dina egna siffror och se hur modellen klassificerar dem!

---

## Jämförelse: CNN vs andra tekniker

| Teknik | Accuracy | Fördelar | Nackdelar |
|--------|----------|----------|-----------|
| k-NN | ~97% | Enkel, inget träning | Långsam vid prediction |
| Decision Tree | ~87% | Transparent | Dålig på bilder |
| **CNN** | **~99%** | Bäst för bilder | Kräver mer data |

---

## Etiska överväganden

### 🚨 Potentiella problem

1. **Bias i träningsdata**: Om datasetet inte är representativt
2. **Adversarial attacks**: Manipulerade bilder som lurar AI:n
3. **Övervakning**: Ansiktsigenkänning kräver GDPR-hänsyn

### 🇸🇪 Svenska exempel

- **Trafikverket**: AI för att detektera vägskyltar och fotgängare
- **Sjukvården**: AI för att analysera röntgenbilder (högrisk enligt EU AI Act)
- **Polisen**: Ansiktsigenkänning (kräver GDPR och integritetsskydd)

---

## Reflektionsfrågor

### Grundläggande (E-nivå)
1. Vad är en convolution och vad gör ett filter?
2. Varför behöver bilder normaliseras innan träning?
3. Hur många parametrar har modellen?

### Utmanande (C-nivå)
1. Varför fungerar CNN bättre än vanliga neural networks för bilder?
2. Vad är dataaugmentering och varför hjälper det?
3. Hur skulle du testa om modellen har bias?

### Avancerat (A-nivå)
1. Hur skulle du förklara modellens beslut för en icke-tekniker?
2. Vilka EU AI Act-krav gäller för medicinsk bildanalys?
3. Hur kan SHAP-värden användas för att förklara CNN-beslut?

---

## 🇸🇪 Svenska facktermer

| Engelska | Svenska |
|----------|---------|
| Convolutional Neural Network | Faltningsnätverk |
| Filter / Kernel | Filter / Kärna |
| Pooling | Poolning |
| Feature map | Featurekarta |
| Dropout | Dropout |
| Epoch | Epok |
| Batch size | Batchstorlek |

---

## Nästa steg

- **Projekt 06**: Träna en RL-robot (förstärkande inlärning)
- **Projekt 08**: Bygg ett neural network från scratch (förstå grunden)
- **Projekt 10**: Capstone med eget domänval

---

## Resurser

- [TensorFlow Keras Tutorial](https://www.tensorflow.org/guide/keras)
- [CNN Visualizer](https://www.cs.ryerson.ca/~aharley/vis/conv/)
- [EU AI Act Official](https://artificialintelligenceact.eu/)

---

*Projektet är en del av Gy25 AI Curriculum – Artificiell Intelligens för Svenska Gymnasium*
