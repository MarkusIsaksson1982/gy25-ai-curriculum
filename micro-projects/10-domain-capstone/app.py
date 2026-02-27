"""
10 Domain Capstone - Streamlit App
Självvalt domänprojekt med fullständig AI-lösning
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Domain Capstone",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 10 Domain Capstone – Självvalt Domänprojekt")
st.markdown("**Tillämpa allt du lärt dig i ett realistiskt scenario** | Gy25 AI Curriculum")

with st.sidebar:
    st.header("🎯 Capstone Info")
    st.markdown("""
    **Sista projektet!**
    
    Välj en domän och bygg en komplett AI-lösning.
    
    **Domäner:**
    - 🚗 Trafik & Transport
    - 🏥 Sjukvård & Hälsa
    - 💰 Finans & Bank
    - 📚 Utbildning
    """)
    
    st.header("📋 Steg")
    st.markdown("""
    1. Välj domän
    2. Problemformulering
    3. Bygg lösning
    4. Riskbedömning
    5. Reflektion
    """)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Välj Domän", "📝 Problem", "🧠 Lösning", "⚖️ Risk", "💭 Reflektion"
])

with tab1:
    st.header("🏠 Välj din domän")
    
    col1, col2 = st.columns(2)
    
    with col1:
        domain = st.radio(
            "Välj ett område:",
            ["🚗 Trafik & Transport", "🏥 Sjukvård & Hälsa", "💰 Finans & Bank", "📚 Utbildning"]
        )
    
    with col2:
        if "Trafik" in domain:
            st.success("""
            ### 🚗 Trafik & Transport
            
            **Exempelprojekt:**
            - Trafikprediktion för Stockholm
            - Kollektivtrafik-optimering
            - Fordonsklassificering
            - Trafiksäkerhetsanalys
            
            **AI-tekniker:**
            - Tidsserieprediktion
            - Bildigenkänning (CNN)
            - Reinforcement Learning
            """)
            
        elif "Sjukvård" in domain:
            st.success("""
            ### 🏥 Sjukvård & Hälsa
            
            **Exempelprojekt:**
            - Sjukdomsprediktion
            - Medicinsk bildanalys
            - Patientriskbedömning
            - Läkemedelsinteraktion
            
            **AI-tekniker:**
            - Klassificering
            - CNN för bilder
            - SHAP för förklaringar
            """)
            
        elif "Finans" in domain:
            st.success("""
            ### 💰 Finans & Bank
            
            **Exempelprojekt:**
            - Kreditbedömning
            - Bedrägeridetektering
            - Aktiemarknads-prediktion
            - Försäkringsrisk
            
            **AI-tekniker:**
            - Klassificering
            - Anomalidetektering
            - Tidsserieanalys
            """)
            
        else:
            st.success("""
            ### 📚 Utbildning
            
            **Exempelprojekt:**
            - Studieprstationsprediktion
            - Antagningssystem
            - Lärandestilsanalys
            - Betygsbedömning
            
            **AI-tekniker:**
            - Klassificering
            - Clustering
            - NLP
            """)

with tab2:
    st.header("📝 Problemformulering")
    
    st.markdown("""
    ### Beskriv ditt problem
    
    Fyll i nedanstående för att definiera ditt projekt:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Projekt-titel")
        project_title = st.text_input("Titel", "Mitt AI-projekt")
        
        st.subheader("Problem")
        problem = st.text_area("Vad är problemet du vill lösa?", 
                              height=100,
                              placeholder="Exempel: Trafiken i Stockholm är oförutsägbar...")
        
    with col2:
        st.subheader("Målgrupp")
        target_group = st.text_input("Vem påverkas?", 
                                    placeholder="Pendlare, patienter, kunder...")
        
        st.subheader("Värde")
        value = st.text_area("Vilket värde skapar lösningen?", 
                            height=100,
                            placeholder="Snabbare restider, bättre diagnoser...")

    st.divider()
    
    st.subheader("📋 Projektplan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        step1 = st.checkbox("Steg 1: Välj domän")
    with col2:
       box("Ste step2 = st.checkbox("Steg 2: Samla data")
    with col3:
        step3 = st.checkbox("Steg 3: Bygg modell")
    with col4:
        step4 = st.checkbox("Steg 4: Riskbedömning")

with tab3:
    st.header("🧠 Bygg din lösning")
    
    st.markdown("""
    ### Bygg din AI-lösning
    
    Använd tekniker från tidigare projekt:
    - **Dataanalys** (Projekt 01)
    - **Beslutsträd** (Projekt 02)
    - **Bildigenkänning** (Projekt 05)
    - **Förstärkande inlärning** (Projekt 06)
    - **Bias & Fairness** (Projekt 07)
    - **Neural Networks + SHAP** (Projekt 08)
    - **End-to-End Pipeline** (Projekt 09)
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Välj AI-teknik")
        
        ai_technique = st.selectbox(
            "Teknik",
            ["Klassificering (Random Forest)", 
             "Neural Network (MLP)",
             "Bildigenkänning (CNN)",
             "Reinforcement Learning",
             "Clustering"]
        )
        
        generate_solution = st.button("Generera lösning", type="primary")
    
    with col2:
        if generate_solution:
            st.success("✅ Här är din lösning:")
            
            if "Klassificering" in ai_technique:
                st.code("""
# Klassificering med Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP förklaring
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
                """, language="python")
                
            elif "Neural" in ai_technique:
                st.code("""
# MLP med TensorFlow/Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=10)
                """, language="python")
                
            elif "Bild" in ai_technique:
                st.code("""
# CNN för bildanalys
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential([
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(n_classes, activation='softmax')
])
                """, language="python")
    
    st.divider()
    
    st.subheader("📊 Demo: Exempel-lösning")
    
    if st.button("Kör exempel"):
        np.random.seed(42)
        
        n = 500
        data = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.randn(n),
            'target': np.random.choice([0, 1], n)
        })
        
        st.session_state['capstone_data'] = data
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        
        st.success(f"Exempel modell tränad! Accuracy: {accuracy:.1%}")
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                    title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("⚖️ Riskbedömning (EU AI Act)")
    
    st.markdown("""
    ### Riskbedömning för ditt AI-system
    
    Fyll i riskbedömningen nedan:
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 System-översikt")
        
        system_type = st.text_input("Typ av AI-system", 
                                   placeholder="Exempel: Prediktivt system")
        
        impact = st.select_slider(
            "Påverkansnivå",
            options=["Låg", "Medel", "Hög"]
        )
        
        uses_personal = st.checkbox("Använder personuppgifter")
        automated_decisions = st.checkbox("Fattar automatiserade beslut")
    
    with col2:
        st.subheader("🚨 Risk-klassificering")
        
        high_risk_domains = ["Sjukvård", "Finans", "Utbildning"]
        
        selected_domain = domain.split()[1] if len(domain.split()) > 1 else ""
        
        is_high_risk = impact == "Hög" or any(d in domain for d in high_risk_domains)
        
        if is_high_risk:
            st.error("🚨 HÖGRISK-SYSTEM")
        else:
            st.success("✅ Inte högrisk")
    
    st.divider()
    
    st.subheader("📝 EU AI Act Checklista")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.checkbox("Riskbedömning gjord")
    with col2:
        st.checkbox("Datahanteringsplan")
    with col3:
        st.checkbox("Teknisk dokumentation")
    with col4:
        st.checkbox("Transparenskrav uppfyllda")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.checkbox("Mänsklig översyn")
    with col2:
        st.checkbox("Loggning och spårbarhet")
    with col3:
        st.checkbox("Incidentrapportering")
    with col4:
        st.checkbox("CE-märkning (om högrisk)")

with tab5:
    st.header("💭 Reflektion och Avslutning")
    
    st.markdown("""
    ### 🎉 Grattis! Du har klarat hela Gy25 AI Curriculum!
    
    Nu är det dags att reflektera över din resa:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Kunskaper")
        
        learned = st.text_area(
            "Vad har du lärt dig under kursen?",
            height=150,
            placeholder="Jag har lärt mig..."
        )
        
        surprised = st.text_area(
            "Vad förvånade dig mest?",
            height=100,
            placeholder="Det som överraskade mig mest var..."
        )
    
    with col2:
        st.subheader("🌍 Samhällspåverkan")
        
        impact = st.text_area(
            "Hur kan AI påverka det svenska samhället?",
            height=150,
            placeholder="AI kan bidra till..."
        )
        
        concerns = st.text_area(
            "Vilka oroar du dig för?",
            height=100,
            placeholder="Jag är oroad över..."
        )
    
    st.divider()
    
    st.subheader("📊 Jämförelsematris")
    
    st.markdown("""
    ### Uppdatera din jämförelsematris
    
    Lägg till dina egna erfarenheter:
    """)
    
    st.table(pd.DataFrame({
        'Teknik': ['Decision Tree', 'CNN', 'RL', 'MLP', 'Pipeline'],
        'Använt i': ['02', '05', '06', '08', '09'],
        'Noggrannhet': ['~97%', '~99%', '~100%', '~95%', 'Varierar'],
        'Transparens': ['Hög', 'Låg', 'Låg', 'Låg', 'Medel'],
        'Rekommenderad': ['Ja', 'Ja', 'Begränsat', 'Ja', 'Ja']
    }))
    
    st.divider()
    
    st.success("""
    # 🎉 GRATTIS!
    
    Du har nu slutfört hela Gy25 AI Curriculum!
    
    **Nivå 1 (01-04)** - Grundläggande
    **Nivå 2a (05-08)** - Avancerat
    **Nivå 2b (09-10)** - Capstone
    
    Du är nu redo att:
    - Bygga egna AI-system
    - Förstå och diskutera AI-etik
    - Tillämpa EU AI Act och GDPR
    - Reflektera kritiskt kring AI i samhället
    """)
    
    st.balloons()

st.markdown("---")
st.caption("Gy25 AI Curriculum | Nivå 2 | Domain Capstone - FULLSTÄNDIG KURS!")
