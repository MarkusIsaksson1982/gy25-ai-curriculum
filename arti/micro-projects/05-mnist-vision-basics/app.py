"""
05 MNIST Vision Basics - Streamlit App
Objektigenkänning med CNN för handskrivna siffror
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import io

st.set_page_config(
    page_title="MNIST Vision Basics",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ 05 MNIST Vision Basics – Objektigenkänning")
st.markdown("**Lär dig hur CNN fungerar för bildigenkänning** | Gy25 AI Curriculum")

with st.sidebar:
    st.header("📚 Lärandemål")
    st.markdown("""
    - Förklara hur CNN fungerar
    - Träna en bildklassificerare
    - Jämföra med andra tekniker
    - Diskutera etik och EU AI Act
    """)
    
    st.header("⚙️ Inställningar")
    show_advanced = st.toggle("Visa avancerade alternativ", value=False)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔧 Träna", "🔍 Utvärdera", "✏️ Rita"])

with tab1:
    st.header("MNIST Dataset – Handskrivna siffror")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        ### Vad är MNIST?
        
        MNIST (Modified National Institute of Standards and Technology) är 
        ett klassiskt dataset för maskininlärning:
        
        - **70,000 bilder** av handskrivna siffror (0-9)
        - **28x28 pixlar** i gråskala
        - **10 klasser** (siffrorna 0-9)
        
        Det är "Hello World" för bildigenkänning!
        """)
        
        st.info("💡 **Svenskt sammanhang:** Samma teknik används av BankID för att känna igen din signatur!")

    with col2:
        if st.button("Ladda och visa exempel"):
            try:
                from tensorflow.keras.datasets import mnist
                (X_train, y_train), (X_test, y_test) = mnist.load_data()
                
                fig, axes = plt.subplots(2, 5, figsize=(14, 6))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(X_train[i], cmap='gray')
                    ax.set_title(f'Siffra: {y_train[i]}', fontsize=12)
                    ax.axis('off')
                plt.suptitle('Exempel från MNIST-datasetet', fontsize=14)
                st.pyplot(fig)
                
                st.success(f"✅ Laddat! Träningsdata: {len(X_train)} bilder, Testdata: {len(X_test)} bilder")
            except ImportError:
                st.error("Installera tensorflow först: pip install tensorflow")

with tab2:
    st.header("🔧 Träna din CNN-modell")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Modell-inställningar")
        
        epochs = st.slider("Antal epoker", 1, 20, 5)
        batch_size = st.selectbox("Batch-storlek", [32, 64, 128], index=2)
        
        use_augmentation = st.checkbox("Använd dataaugmentering", value=False)
        
        if use_augmentation:
            st.markdown("#### Augmentering:")
            rotation = st.slider("Rotation", 0, 30, 10)
            shift = st.slider("Förskjutning", 0.0, 0.3, 0.1)
        
        train_button = st.button("🚀 Starta träning", type="primary")
        
    with col2:
        if train_button:
            with st.spinner("Laddar MNIST..."):
                try:
                    from tensorflow.keras.datasets import mnist
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
                    
                    (X_train, y_train), (X_test, y_test) = mnist.load_data()
                    
                    X_train = X_train.astype('float32') / 255.0
                    X_test = X_test.astype('float32') / 255.0
                    X_train = X_train.reshape(-1, 28, 28, 1)
                    X_test = X_test.reshape(-1, 28, 28, 1)
                    
                    model = Sequential([
                        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                        MaxPooling2D((2, 2)),
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dropout(0.3),
                        Dense(10, activation='softmax')
                    ])
                    
                    model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    st.info(f"Tränar i {epochs} epoker med batch-storlek {batch_size}...")
                    
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        verbose=1
                    )
                    
                    st.session_state['model'] = model
                    st.session_state['history'] = history.history
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    
                    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                    st.success(f"✅ Träning klar! Testnoggrannhet: {test_acc:.2%}")
                    
                except ImportError:
                    st.error("Installera tensorflow: pip install tensorflow")
                except Exception as e:
                    st.error(f"Fel: {e}")
        else:
            st.info("👈 Konfigurera och klicka på 'Starta träning' för att träna modellen")
            
            st.markdown("### Modellarkitektur")
            st.code("""
model = Sequential([
    Conv2D(32, (3,3), activation='relu'),  # Första faltningslagret
    MaxPooling2D((2,2)),                    # Poolning
    Conv2D(64, (3,3), activation='relu'),  # Andra faltningslagret
    MaxPooling2D((2,2)),
    Flatten(),                              # Platta till vektor
    Dense(128, activation='relu'),         # Fullt kopplat
    Dropout(0.3),                          # Regularisering
    Dense(10, activation='softmax')        # Output: 10 klasser
])
            """, language="python")

    if 'history' in st.session_state:
        st.divider()
        st.subheader("📈 Träningshistorik")
        
        history = st.session_state['history']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['accuracy'], name='Träning', mode='lines'))
            fig.add_trace(go.Scatter(y=history['val_accuracy'], name='Validering', mode='lines'))
            fig.update_layout(title='Noggrannhet per epok', xaxis_title='Epok', yaxis_title='Noggrannhet')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['loss'], name='Träning', mode='lines'))
            fig.add_trace(go.Scatter(y=history['val_loss'], name='Validering', mode='lines'))
            fig.update_layout(title='Förlust per epok', xaxis_title='Epok', yaxis_title='Förlust')
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🔍 Utvärdera modellen")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Träna en modell först i fliken 'Träna'!")
    else:
        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Klassificeringsrapport")
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred_classes, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format('{:.2%}'))
            
        with col2:
            st.subheader("🎯 Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred_classes)
            
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predikterad", y="Sann", color="Antal"))
            fig.update_layout(title='Confusion Matrix')
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎲 Testa enskilda bilder")
            idx = st.slider("Välj bild-index", 0, len(X_test)-1, 42)
            
            img = X_test[idx]
            true_label = y_test[idx]
            pred = model.predict(img.reshape(1, 28, 28, 1))
            pred_label = np.argmax(pred)
            confidence = np.max(pred)
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img.reshape(28, 28), cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'Sann: {true_label} | Pred: {pred_label}\nKonfidens: {confidence:.1%}', color=color)
            ax.axis('off')
            st.pyplot(fig)
            
            if true_label != pred_label:
                st.error(f"❌ Fel klassificerad! Modellen trodde det var en {pred_label} men det var en {true_label}")
            else:
                st.success(f"✅ Rätt klassificerad som {pred_label}!")
                
        with col2:
            st.subheader("📊 Konfidensfördelning")
            confidences = np.max(y_pred, axis=1)
            
            fig = px.histogram(confidences, nbins=20, labels={'value': 'Konfidens'},
                              title='Fördelning av modellens konfidens')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Statistik:**
            - Medel-konfidens: {np.mean(confidences):.1%}
            - Min-konfidens: {np.min(confidences):.1%}
            - Max-konfidens: {np.max(confidences):.1%}
            """)

with tab4:
    st.header("✏️ Rita din egen siffra")
    st.markdown("Rita en siffra (0-9) i rutan nedan och se vad modellen predikterar!")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Träna en modell först i fliken 'Träna'!")
    else:
        model = st.session_state['model']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Rita här")
            
            canvas_placeholder = st.empty()
            
            if 'canvas_image' not in st.session_state:
                st.session_state['canvas_image'] = None
            
            img_file = st.file_uploader("Eller ladda upp en bild", type=['png', 'jpg', 'jpeg'])
            
            if img_file:
                image = Image.open(img_file).convert('L')
                image = image.resize((28, 28))
                img_array = np.array(image)
                img_array = 255 - img_array
                img_array = img_array.astype('float32') / 255.0
                img_array = img_array.reshape(28, 28, 1)
            else:
                st.info("👆 Ladda upp en bild av en handskriven siffra")
                st.markdown("Eller använd exemplen nedan:")
                
                example_idx = st.selectbox("Välj exempel", list(range(10)))
                img_array = X_test[np.where(y_test == example_idx)[0][0]]
            
            predict_button = st.button("🔮 Prediktera", type="primary")
            
            if predict_button:
                pred = model.predict(img_array.reshape(1, 28, 28, 1))
                pred_class = np.argmax(pred)
                confidence = np.max(pred)
                
            with col2:
                st.subheader("Resultat")
                
                if predict_button:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(img_array.reshape(28, 28), cmap='gray')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    st.markdown(f"## 🎯 {pred_class}")
                    st.progress(confidence)
                    st.caption(f"Konfidens: {confidence:.1%}")
                    
                    st.subheader("Sannolikheter per klass")
                    probs = pred[0]
                    
                    fig = px.bar(
                        x=list(range(10)),
                        y=probs,
                        labels={'x': 'Siffra', 'y': 'Sannolikhet'},
                        color=probs,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Rita eller ladda upp en bild och klicka på Prediktera")

st.divider()
st.markdown("""
## 🇸🇪 Svenska facktermer

| Engelska | Svenska |
|----------|---------|
| Convolutional Neural Network (CNN) | Faltningsnätverk |
| Filter / Kernel | Filter / Kärna |
| Pooling | Poolning |
| Feature map | Featurekarta |
| Epoch | Epok |
| Batch size | Batch-storlek |

## 📚 Läs mer

- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [EU AI Act - Högrisk system](https://artificialintelligenceact.eu/the-act/)
- [Trafikverket AI](https://www.trafikverket.se/)
""")

st.markdown("---")
st.caption("Gy25 AI Curriculum | Nivå 1-2 | MNIST Vision Basics")
