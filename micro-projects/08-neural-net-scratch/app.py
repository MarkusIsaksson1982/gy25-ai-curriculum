"""
08 Neural Net from Scratch + SHAP - Streamlit App
Bygg ett neural network från grunden och lär dig förklara det med SHAP
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Neural Net from Scratch",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 08 Neural Net from Scratch + SHAP")
st.markdown("**Bygg ett neuronnät från grunden och lär dig förklara det** | Gy25 AI Curriculum")

with st.sidebar:
    st.header("📚 Lärandemål")
    st.markdown("""
    - Bygga NN från scratch med numpy
    - Förklara forward/backward propagation
    - Använda SHAP för förklaringar
    - Diskutera transparens (EU AI Act)
    """)
    
    st.header("⚙️ Välj Dataset")
    dataset_choice = st.selectbox(
        "Dataset",
        ["Iris", "Swedish Student Data", "Titanic"]
    )

tab1, tab2, tab3, tab4 = st.tabs(["🧠 teori", "🔧 Bygg NN", "📊 SHAP", "⚖️ Etik"])

with tab1:
    st.header("🧠 Neural Network – Teori")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Vad är ett Neural Network?
        
        Ett neural network (neuronnät) är inspirerat av hur hjärnan fungerar:
        - Massor av små enheter ("neuroner")
        - Samkopplade med "vikter"
        - Kan lära sig komplexa mönster
        
        ### Grundstruktur
        
        ```
        Input Layer → Hidden Layer(s) → Output Layer
        
        [x1]              [h1]
        [x2] ─────► [W] ──► [h2] ──► [W2] ──► [y]
        [x3]              [h3]
        
        x = input, h = hidden, y = output
        W = weights (vikter)
        ```
        
        ### Aktiveringsfunktioner
        
        | Funktion | Formel | Används för |
        |----------|--------|-------------|
        | Sigmoid | 1/(1+e^-x) | Output (0-1) |
        | ReLU | max(0,x) | Hidden layers |
        | Softmax | e^x/Σe^x | Multi-class output |
        """)
        
    with col2:
        st.markdown("""
        ### Forward Propagation
        
        ```
        1. Input: x
        2. Linear: z = x·W + b
        3. Activation: a = f(z)
        4. Repeat för varje lager
        5. Output: prediction
        ```
        
        ### Backpropagation
        
        ```
        1. Beräkna fel i output
        2. Gradient från output tillbaka till input
        3. Uppdatera vikter: W = W - α·gradient
        4. Upprepa tills loss är liten
        ```
        
        ### Varför SHAP?
        
        Neural networks är "black boxes" – svåra att förstå!
        
        SHAP ger oss:
        - ✅ Lokala förklaringar (varför detta beslut)
        - ✅ Globala förklaringar (viktiga features)
        - ✅ Jämförbarhet med andra modeller
        """)
    
    st.divider()
    
    st.subheader("🎯 Neuron – Visualisering")
    
    st.markdown("""
    ```
           input
             │
             ▼
    ┌────────────────┐
    │   x₁·w₁        │
    │ + x₂·w₂  = z   │  Linear transformation
    │ + x₃·w₃        │
    │   + bias        │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  f(z) = a      │  Activation (ReLU, Sigmoid...)
    └────────┬───────┘
             │
             ▼
          output
    ```
    """)

with tab2:
    st.header("🔧 Bygg Neural Network")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Neural Network Arkitektur")
        
        n_features = st.number_input("Antal features", 2, 20, 4)
        hidden_layers = st.text_input("Hidden layers (kommaseparerad)", "16, 16")
        n_classes = st.number_input("Antal klasser", 2, 10, 3)
        
        learning_rate = st.slider("Learning rate", 0.001, 0.5, 0.1)
        epochs = st.slider("Antal epoker", 100, 2000, 500)
        
        st.markdown("### Data")
        
        if dataset_choice == "Iris":
            st.info("Laddar Iris dataset...")
        elif dataset_choice == "Swedish Student Data":
            st.info("Genererar Swedish Student Data...")
        else:
            st.info("Laddar Titanic dataset...")
            
    with col2:
        st.markdown("### Kod: Neural Network från scratch")
        
        code = '''
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        np.random.seed(42)
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.softmax(z)
            self.activations.append(a)
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        output = self.activations[-1]
        deltas[-1] = output - y
        
        for i in range(len(self.weights) - 2, -1, -1):
            deltas[i] = np.dot(deltas[i+1], self.weights[i+1].T) * self.relu_derivative(self.activations[i+1])
        
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.activations[i].T, deltas[i]) / m
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = -np.mean(y * np.log(output + 1e-8))
            losses.append(loss)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        return losses
        '''
        
        st.code(code, language="python")
        
        train_button = st.button("🚀 Träna Neural Network", type="primary")
        
        if train_button:
            with st.spinner("Tränar neural network..."):
                try:
                    from sklearn.datasets import load_iris
                    from sklearn.preprocessing import OneHotEncoder, StandardScaler
                    from sklearn.model_selection import train_test_split
                    
                    iris = load_iris()
                    X = iris.data
                    y = iris.target.reshape(-1, 1)
                    
                    encoder = OneHotEncoder(sparse_output=False)
                    y_onehot = encoder.fit_transform(y)
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y_onehot, test_size=0.3, random_state=42
                    )
                    
                    layer_sizes = [n_features] + [int(x) for x in hidden_layers.split(',')] + [n_classes]
                    
                    class NeuralNetwork:
                        def __init__(self, layer_sizes):
                            self.weights = []
                            self.biases = []
                            np.random.seed(42)
                            
                            for i in range(len(layer_sizes) - 1):
                                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
                                b = np.zeros((1, layer_sizes[i+1]))
                                self.weights.append(w)
                                self.biases.append(b)
                        
                        def relu(self, x):
                            return np.maximum(0, x)
                        
                        def relu_derivative(self, x):
                            return (x > 0).astype(float)
                        
                        def softmax(self, x):
                            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
                        
                        def forward(self, X):
                            self.activations = [X]
                            for i in range(len(self.weights)):
                                z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
                                if i < len(self.weights) - 1:
                                    a = self.relu(z)
                                else:
                                    a = self.softmax(z)
                                self.activations.append(a)
                            return self.activations[-1]
                        
                        def backward(self, X, y, learning_rate):
                            m = X.shape[0]
                            deltas = [None] * len(self.weights)
                            
                            output = self.activations[-1]
                            deltas[-1] = output - y
                            
                            for i in range(len(self.weights) - 2, -1, -1):
                                deltas[i] = np.dot(deltas[i+1], self.weights[i+1].T) * self.relu_derivative(self.activations[i+1])
                            
                            for i in range(len(self.weights)):
                                self.weights[i] -= learning_rate * np.dot(self.activations[i].T, deltas[i]) / m
                                self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
                        
                        def train(self, X, y, epochs, learning_rate):
                            losses = []
                            for epoch in range(epochs):
                                output = self.forward(X)
                                loss = -np.mean(y * np.log(output + 1e-8))
                                losses.append(loss)
                                self.backward(X, y, learning_rate)
                            return losses
                        
                        def predict(self, X):
                            return self.forward(X)
                    
                    nn = NeuralNetwork(layer_sizes)
                    losses = nn.train(X_train, y_train, epochs, learning_rate)
                    
                    predictions = nn.predict(X_test)
                    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
                    
                    st.session_state['nn'] = nn
                    st.session_state['losses'] = losses
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['feature_names'] = iris.feature_names_
                    st.session_state['class_names'] = iris.target_names
                    
                    st.success(f"✅ Träning klar! Accuracy: {accuracy:.1%}")
                    
                except Exception as e:
                    st.error(f"Fel: {e}")

    if 'losses' in st.session_state:
        st.divider()
        
        losses = st.session_state['losses']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, mode='lines', name='Loss'))
        fig.update_layout(title='Träningsförlust över epoker', xaxis_title='Epok', yaxis_title='Loss')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("📊 SHAP – Förklara modellen")
    
    if 'nn' not in st.session_state:
        st.warning("⚠️ Träna en modell först i fliken 'Bygg NN'!")
    else:
        st.markdown("""
        ### Vad är SHAP?
        
        SHAP (SHapley Additive exPlanations) förklarar hur varje feature bidrar 
        till en specifik prediktion.
        
        - **Positiva SHAP values**: Pushar prediktionen uppåt
        - **Negativa SHAP values**: Pushar prediktionen neråt
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Välj ett test-exempel")
            
            X_test = st.session_state['X_test']
            feature_names = st.session_state['feature_names']
            
            example_idx = st.slider("Exempel index", 0, len(X_test)-1, 0)
            
        with col2:
            st.markdown("### Använd sklearn MLP för SHAP")
            
            use_sklearn = st.checkbox("Använd sklearn MLP (för SHAP)")
            
            if use_sklearn:
                with st.spinner("Tränar sklearn MLP..."):
                    from sklearn.neural_network import MLPClassifier
                    from sklearn.preprocessing import StandardScaler, LabelEncoder
                    from sklearn.datasets import load_iris
                    import warnings
                    warnings.filterwarnings('ignore')
                    
                    iris = load_iris()
                    
                    mlp = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000, random_state=42)
                    mlp.fit(iris.data, iris.target)
                    
                    st.session_state['mlp'] = mlp
                    st.session_state['iris'] = iris
        
        if 'mlp' in st.session_state:
            st.divider()
            
            st.subheader("📊 Feature Importance (SHAP-liknande analys)")
            
            mlp = st.session_state['mlp']
            iris = st.session_state['iris']
            
            X_sample = iris.data[example_idx:example_idx+1]
            
            feature_importance = np.abs(mlp.coefs_[0]).mean(axis=1)
            importance_df = pd.DataFrame({
                'Feature': iris.feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title='Feature Importance (baserat på vikter)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            st.subheader("🔍 Enskild prediktion")
            
            pred = mlp.predict(X_sample)[0]
            pred_proba = mlp.predict_proba(X_sample)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Predikterad klass:** {iris.target_names[pred]}")
                
            with col2:
                st.markdown("**Sannolikheter:**")
                for i, (name, prob) in enumerate(zip(iris.target_names, pred_proba)):
                    st.progress(prob)
                    st.caption(f"{name}: {prob:.1%}")
            
            st.info(f"""
            💡 **SHAP-analys:** Feature '{iris.feature_names[np.argmax(feature_importance)]}' 
            har högst genomsnittlig vikt i modellen, vilket gör den viktigast för prediktionerna.
            """)

with tab4:
    st.header("⚖️ Etik och Transparens")
    
    st.markdown("""
    ### 🧠 Neural Networks – Svagheter och Etik
    
    Neural networks är kraftfulla men har viktiga svagheter:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚨 Svagheter
        
        1. **Black Box** – Svårt att förstå hur beslut fattas
        2. **Kräver mycket data** – Databehov
        3. **Kan överanpassa** – Overfitting
        4. **Instabila** – Små ändringar kan ge stora effekter
        5. **Kräver experter** – Svårt att finjustera
        """)
        
    with col2:
        st.markdown("""
        ### 💡 Lösningar
        
        1. **SHAP** – Förklara enskilda beslut
        2. **LIME** – Lokala förklaringar
        3. **Feature importance** – Global förståelse
        4. **Regularization** – Minska overfitting
        5. **Dokumentation** – Förstå modellen
        """)
    
    st.divider()
    
    st.subheader("🇪🇺 EU AI Act – Krav på transparens")
    
    st.markdown("""
    | Artikel | Krav |
    |---------|------|
    | Art. 14.2 | AI-system ska vara utformade för tolkning |
    | Art. 14.4 | Tillräcklig traceability och loggning |
    | Bilaga IV | Dokumentation ska inkludera förklaringar |
    
    ### 🇸🇪 Svenska tillämpningar
    
    - **Sjukvård**: AI-diagnoser måste kunna förklaras
    - **Bank**: Kunder har rätt till förklaring av kreditbeslut
    - **Rekrytering**: Kandidater har rätt att veta hur bedömning gick till
    """)
    
    st.divider()
    
    st.subheader("📋 Checklist: Är din modell transparent?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Har du analyserat feature importance?")
        st.checkbox("Har du testat modellen på edge cases?")
        st.checkbox("Finns dokumentation av modellens beslut?")
        
    with col2:
        st.checkbox("Kan du förklara ett enskilt beslut?")
        st.checkbox("Har du gjort en riskanalys?")
        st.checkbox("Finns mänsklig översyn av viktiga beslut?")

st.divider()

st.markdown("""
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
| Transparency | Transparens |
""")

st.markdown("---")
st.caption("Gy25 AI Curriculum | Nivå 2 | Neural Net from Scratch + SHAP")
