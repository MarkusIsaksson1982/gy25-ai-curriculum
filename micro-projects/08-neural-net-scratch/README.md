# 08 Neural Net from Scratch – MLP + SHAP

**Nivå:** 2  
**Tid:** 120-180 minuter  
**Fokus:** Byggstenar neuronnät, Träning, Explainability med SHAP  
**Dataset:** Iris, Swedish Student Data

## Lärandemål

Efter detta projekt kan du:
- Bygga ett neural network från grunden med numpy
- Förklara forward propagation och backpropagation
- Använda SHAP för att förklara modellbeslut
- Diskutera transparens och förklarbarhet (EU AI Act)

## Koppling till Skolverket Gy25

| Centralt innehåll | Täckning |
|------------------|----------|
| Användning av djupa neuronnät | ✅ Huvudfokus |
| Svagheter i ML med avseende på transparens och förklarbarhet | ✅ SHAP |
| Implementering och träning av neuronnät | ✅ From scratch |

## 🇸🇪 Svenska sammanhang

- **Trafikverket**: Trafikprediktion med neuronnät
- **Sjukvård**: Diagnostik med förklarbara modeller
- **Bank**: Kreditbedömning med SHAP-förklaringar

## 🇪🇺 EU AI Act-koppling

- **Artikel 14**: Krav på transparens och förklarbarhet för högrisk-system
- **Bilaga IV**: Teknisk dokumentation måste inkludera förklaringar
- **SHAP som verktyg**: Uppfylla "right to explanation"

---

## Snabbstart

```bash
# Installera extra beroenden
pip install shap plotly

# Starta appen
streamlit run micro-projects/08-neural-net-scratch/app.py
```

---

## Introduktion

### Varför bygga från grunden?

Att bygga ett neural network från grunden hjälper dig att förstå:
- **Hur** neural networks faktiskt fungerar
- **Varför** de kan vara svåra att träna
- **Vad** som händer "under huven"

### Vad är SHAP?

**SHAP (SHapley Additive exPlanations)** är ett ramverk för att förklara 
modelldelenslutningar genom att beräkna varje features bidrag till prediktionen.

```
SHAP Value = "Hur mycket påverkade denna feature prediktionen?"
```

---

## Steg 1: Neural Network Grunder

### Enkelt Neuron

```
        input
           │
           ▼
    ┌────────────┐
    │  Weights    │  w * x
    │    + bias   │  + b
    └────────────┘
           │
           ▼
    ┌────────────┐
    │ Activation │  f(w*x + b)
    │ (ReLU,     │
    │  Sigmoid,  │
    │  Softmax)  │
    └────────────┘
           │
           ▼
       output
```

### Aktiveringsfunktioner

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

---

## Steg 2: Bygg MLP från Scratch

### Klassdefinition

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list med antal noder i varje lager
        ex: [4, 8, 8, 2] = 4 input, 2 hidden, 2 hidden, 2 output
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialisera vikter och bias
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            
            if i < len(self.weights) - 1:
                a = relu(z)  # Hidden layers
            else:
                a = softmax(z)  # Output layer
            
            self.activations.append(a)
        
        return self.activations[-1]
```

### Backpropagation

```python
    def backward(self, X, y, learning_rate):
        """Backpropagation med gradient descent"""
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer delta
        output = self.activations[-1]
        deltas[-1] = output - y
        
        # Hidden layers deltas (backwards)
        for i in range(len(self.weights) - 2, -1, -1):
            deltas[i] = np.dot(deltas[i+1], self.weights[i+1].T) * relu_derivative(self.activations[i+1])
        
        # Uppdatera vikter och bias
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.activations[i].T, deltas[i]) / m
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs, learning_rate):
        """Träna nätverket"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Beräkna loss (cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8))
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        return losses
```

---

## Steg 3: Träna på Iris-dataset

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Ladda Iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

# Standardisera features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dela upp
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.3, random_state=42)

# Skapa och träna modell
nn = NeuralNetwork([4, 16, 16, 3])
losses = nn.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Utvärdera
predictions = nn.forward(X_test)
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
print(f"Accuracy: {accuracy:.2%}")
```

---

## Steg 4: SHAP förklarbarhet

### Vad är SHAP?

SHAP (SHapley Additive exPlanations) ger dig en förklaring för varje prediktion:

```python
import shap

# Skapa SHAP explainer
explainer = shap.KernelExplainer(model.predict, X_train)

# Beräkna SHAP values för ett test-exempel
shap_values = explainer.shap_values(X_test[0:5])

# Visualisera
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### SHAP Waterfall Plot

```python
# För ett enskilt beslut
shap.initjs()
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0],
    X_test[0],
    feature_names=feature_names
)
```

### SHAP Feature Importance

```python
# Global feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")
```

---

## Steg 5: Jämför med andra modeller

### MLP med sklearn

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(16, 16),
    activation='relu',
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)
accuracy_sklearn = mlp.score(X_test, y_test)
print(f"Sklearn MLP Accuracy: {accuracy_sklearn:.2%}")
```

### Jämförelse

| Modell | Accuracy | Transparens | SHAP-stöd |
|--------|----------|-------------|-----------|
| Decision Tree | ~97% | Hög | Ja |
| **Custom NN** | ~95% | Låg | Ja |
| Sklearn MLP | ~95% | Låg | Ja |
| Random Forest | ~97% | Medel | Ja |

---

## Förklarbarhet i EU AI Act

### Artikel 14 – Övervakingsåtgärder

> *"AI-system som används i högrisk-tillämpningar ska vara utformade 
> och utvecklade på ett sätt som möjliggör ... tolkning av 
> utdata."*

### Hur SHAP hjälper

1. **Global förklaring**: Vilka features är viktigast totalt?
2. **Lokal förklaring**: Varför fick denna person detta beslut?
3. **Jämförbarhet**: Jämför med andra modeller

---

## Etiska överväganden

### Varför är transparens viktigt?

| Situation | Varför det spelar roll |
|-----------|----------------------|
| Sjukvård | Läkaren måste förstå AI:s diagnos |
| Rättsväsende | Den åtalade har rätt att veta bevisen |
| Krediter | Kunden har rätt att överklaga |
| Rekrytering | Kandiaten måste få feedback |

### Svagheter i neural networks

```
┌────────────────────────────────────────────────────────────┐
│              NEURAL NETWORK SVAGHETER                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Black box – svårt att förstå hur beslut fattas      │
│  2. Kräver mycket data – databehov                       │
│  3. Kan överanpassa – overfitting                        │
│  4. Instabila – små ändringar kan ge stora effekter     │
│  5. Kräver experter – svårt att finjustera               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Reflektionsfrågor

### Grundläggande (E-nivå)

1. **Vad är forward propagation?**
2. **Vad är aktiveringsfunktionens roll i ett neuron?**
3. **Varför är neural networks svårare att förklara än beslutsträd?**

### Utmanande (C-nivå)

1. **Hur fungerar backpropagation?**
2. **Vad är SHAP och varför är det viktigt för EU AI Act?**
3. **Hur kan du använda SHAP för att upptäcka bias?**

### Avancerat (A-nivå)

1. **Vad är gradient vanishing och hur påverkar det träningen?**
2. **Hur balanserar man mellan noggrannhet och förklarbarhet?**
3. **Designa ett system som uppfyller EU AI Act krav på transparens.**

---

## 🇸🇪 Svenska facktermer

| Engelska | Svenska |
|----------|---------|
| Neural Network | Neuronnät |
| Hidden layer | Dolt lager |
| Forward propagation | Framåtpropagering |
| Backpropagation | Bakåtpropagering |
| Weights | Vikter |
| Bias | Bias / offset |
| Activation function | Aktiveringsfunktion |
| Gradient descent | Gradientnedstigning |
| SHAP values | SHAP-värden |
| Feature importance | Featureviktighet |
| Explainability | Förklarbarhet |

---

## Nästa steg

- **Projekt 09**: End-to-End Pipeline – deployment och riskbedömning
- **Projekt 10**: Domain Capstone – tillämpa i valfri domän
- Utveckla egna modeller med SHAP-förklaringar

---

## Resurser

- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3p3)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [EU AI Act - Article 14](https://artificialintelligenceact.eu/article-14/)

---

*Projektet är en del av Gy25 AI Curriculum – Artificiell Intelligens för Svenska Gymnasium*
