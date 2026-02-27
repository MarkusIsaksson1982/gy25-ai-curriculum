"""
09 End-to-End Pipeline - Streamlit App
Komplett ML-pipeline med kvalitetssäkring, riskbedömning och deployment
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="End-to-End Pipeline",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 09 End-to-End Pipeline – Komplett ML-process")
st.markdown("**Bygg, utvärdera, och deploya AI-system** | Gy25 AI Curriculum")

with st.sidebar:
    st.header("📚 Lärandemål")
    st.markdown("""
    - Bygga komplett ML-pipeline
    - Kvalitetssäkring och testning
    - Riskbedömning enligt EU AI Act
    - Deployment och övervakning
    """)
    
    st.header("⚙️ Pipeline-steg")
    st.markdown("""
    1. Datahantering
    2. Förbehandling
    3. Modellträning
    4. Utvärdering
    5. Riskbedömning
    6. Deployment
    """)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data", "🔧 Förbehandling", "🧠 Träning", 
    "📈 Utvärdering", "⚖️ Risk", "🚀 Deployment"
])

with tab1:
    st.header("📊 Steg 1: Datahantering")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        ### Datahantering
        
        Första steget i varje ML-projekt är att:
        
        1. **Ladda data** från källa
        2. **Validera** datakvalitet
        3. **Dela** i träning/test
        4. **Logga** för spårbarhet
        """)
        
        dataset_choice = st.selectbox(
            "Välj dataset",
            ["Swedish Student Data", "Iris", "Titanic"]
        )
        
        if st.button("Ladda och validera data"):
            np.random.seed(42)
            
            if dataset_choice == "Swedish Student Data":
                n = 1000
                data = pd.DataFrame({
                    'student_id': range(1, n+1),
                    'gender': np.random.choice(['M', 'F'], n, p=[0.48, 0.52]),
                    'birth_country': np.random.choice(['Sweden', 'Other'], n, p=[0.75, 0.25]),
                    'parent_education': np.random.choice(['High', 'Medium', 'Low'], n),
                    'study_hours': np.random.normal(15, 5, n).clip(0, 40),
                    'math_grade': np.random.normal(75, 15, n).clip(0, 100),
                    'english_grade': np.random.normal(72, 15, n).clip(0, 100),
                    'admitted': np.zeros(n)
                })
                data['admitted'] = ((data['math_grade'] + data['english_grade']) / 2 > 60).astype(int)
                
            elif dataset_choice == "Iris":
                from sklearn.datasets import load_iris
                iris = load_iris()
                data = pd.DataFrame(iris.data, columns=iris.feature_names)
                data['target'] = iris.target
                
            else:  # Titanic
                from sklearn.datasets import fetch_openml
                titanic = fetch_openml('titanic', version=1, as_frame=True)
                data = titanic.frame
    
    with col2:
        if 'data' in locals() or 'data' in st.session_state:
            try:
                if dataset_choice == "Swedish Student Data":
                    st.session_state['data'] = data
                    
                st.write(f"**Dataset:** {len(data)} rader, {len(data.columns)} kolumner")
                st.dataframe(data.head(), use_container_width=True)
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Rader", len(data))
                with col_b:
                    st.metric("Kolumner", len(data.columns))
                with col_c:
                    missing = data.isnull().sum().sum()
                    st.metric("Saknade värden", missing)
                    
            except Exception as e:
                st.info("Klicka på 'Ladda och validera data' för att börja")
        else:
            st.info("Klicka på 'Ladda och validera data' för att börja")
    
    st.divider()
    
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        st.subheader("📊 Datafördelning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if dataset_choice == "Swedish Student Data":
                fig = px.histogram(data, x='math_grade', title='Betyg i matte')
                st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            if dataset_choice == "Swedish Student Data":
                fig = px.histogram(data, x='english_grade', title='Betyg i engelska')
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🔧 Steg 2: Förbehandling")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ Ladda data först i fliken 'Data'!")
    else:
        data = st.session_state['data']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### Preprocessing Pipeline
            
            Standardisering och transformering:
            
            ```python
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
            ])
            ```
            """)
            
            apply_preprocessing = st.button("Applicera förbehandling")
            
        with col2:
            if apply_preprocessing:
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                
                data_clean = data.copy()
                
                numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    scaler = StandardScaler()
                    data_clean[numeric_cols] = scaler.fit_transform(data_clean[numeric_cols])
                    
                    st.success("✅ Förbehandling klar!")
                    
                    st.write("**Efter standardisering:**")
                    st.dataframe(data_clean.head(), use_container_width=True)
                    
                    st.session_state['data_clean'] = data_clean

with tab3:
    st.header("🧠 Steg 3: Modellträning")
    
    if 'data_clean' not in st.session_state:
        st.warning("⚠️ Förbehandla data först!")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Modellval")
            
            model_choice = st.selectbox(
                "Välj modell",
                ["RandomForest", "GradientBoosting", "LogisticRegression", "DecisionTree"]
            )
            
            test_size = st.slider("Teststorlek", 0.1, 0.5, 0.3)
            
            train_button = st.button("Träna modell", type="primary")
            
        with col2:
            if train_button:
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.tree import DecisionTreeClassifier
                
                data = st.session_state['data_clean']
                
                if 'admitted' in data.columns:
                    target = 'admitted'
                elif 'target' in data.columns:
                    target = 'target'
                else:
                    target = data.columns[-1]
                
                X = data.drop(columns=[target])
                y = data[target]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                models = {
                    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                    'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42)
                }
                
                model = models[model_choice]
                model.fit(X_train, y_train)
                
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                st.session_state['model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                st.success(f"✅ Träning klar!")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Träningsnoggrannhet", f"{train_score:.1%}")
                with col_b:
                    st.metric("Testnoggrannhet", f"{test_score:.1%}")

with tab4:
    st.header("📈 Steg 4: Utvärdering")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Träna en modell först!")
    else:
        from sklearn.metrics import classification_report, confusion_matrix
        
        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Klassificeringsrapport")
            
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format('{:.2%}'))
            
        with col2:
            st.subheader("🎯 Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predikterad", y="Sann"))
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        st.subheader("📉 Cross-validation")
        
        from sklearn.model_selection import cross_val_score
        
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=cv_scores,
            marker_color='steelblue'
        ))
        fig.update_layout(title='Cross-validation scores', yaxis_title='Accuracy')
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Medel CV-score", f"{cv_scores.mean():.1%}", 
                 delta=f"±{cv_scores.std():.1%}")

with tab5:
    st.header("⚖️ Steg 5: Riskbedömning (EU AI Act)")
    
    st.markdown("""
    ### EU AI Act Riskbedömning
    
    Enligt EU AI Act klassificeras vissa AI-system som **högrisk** och 
    kräver strikta krav.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌍 Välj domän")
        
        domain = st.selectbox(
            "Användningsdomän",
            ["Utbildning", "Arbetsmarknad", "Sjukvård", "Transport", "Finans"]
        )
        
        st.subheader("📋 Systemegenskaper")
        
        uses_personal_data = st.checkbox("Använder personuppgifter", value=True)
        automated_decisions = st.checkbox("Fattar automatiserade beslut", value=True)
        profiling = st.checkbox("Profiling", value=False)
        
    with col2:
        st.subheader("📊 Riskbedömning")
        
        high_risk_domains = ["Utbildning", "Arbetsmarknad", "Sjukvård"]
        is_high_risk = domain in high_risk_domains
        
        if is_high_risk:
            st.error("🚨 HÖGRISK-SYSTEM")
            st.markdown("""
            **Kräver enligt EU AI Act:**
            - Riskbedömning före idrifttagande
            - Datahanteringsplan
            - Teknisk dokumentation
            - CE-märkning
            - Kvalitetshanteringssystem
            - Mänsklig översyn
            - Loggning och övervakning
            - Incidentrapportering
            """)
        else:
            st.success("✅ Inte högrisk")
            st.markdown("Standardkrav för transparens och dokumentation gäller.")
    
    st.divider()
    
    st.subheader("🇸🇪 GDPR Efterlevnad")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if uses_personal_data:
            st.checkbox("Artikel 5 - Princip för laglighet", value=True, disabled=True)
        else:
            st.checkbox("Artikel 5 - Princip för laglighet", disabled=True)
            
    with col2:
        if automated_decisions:
            st.checkbox("Artikel 22 - Automatiserat beslutsfattande", value=True, disabled=True)
        else:
            st.checkbox("Artikel 22 - Automatiserat beslutsfattande", disabled=True)
            
    with col3:
        if profiling:
            st.checkbox("Artikel 13 - Rätt till information", value=True, disabled=True)
        else:
            st.checkbox("Artikel 13 - Rätt till information", disabled=True)

with tab6:
    st.header("🚀 Steg 6: Deployment")
    
    st.markdown("""
    ### Deploya din modell
    
    När modellen är tränad och utvärderad kan den deployas för produktion.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        #### Spara modellen
        
        ```python
        import joblib
        joblib.dump(model, 'model.joblib')
        ```
        
        #### Streamlit App
        
        ```python
        import streamlit as st
        import joblib
        
        model = joblib.load('model.joblib')
        prediction = model.predict(user_input)
        st.write(f"Prediction: {prediction}")
        ```
        """)
        
        deploy_button = st.button("📦 Spara modell")
        
    with col2:
        if deploy_button and 'model' in st.session_state:
            import joblib
            from datetime import datetime
            
            model = st.session_state['model']
            
            joblib.dump(model, 'deployed_model.joblib')
            
            metadata = {
                'model_type': type(model).__name__,
                'created_at': datetime.now().isoformat(),
                'test_accuracy': st.session_state['model'].score(
                    st.session_state['X_test'], 
                    st.session_state['y_test']
                )
            }
            
            import json
            with open('model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            st.success("✅ Modell sparad!")
            st.code("deployed_model.joblib", language="text")
            st.json(metadata)
        else:
            st.info("Träna en modell först för att deploya")
    
    st.divider()
    
    st.subheader("🖥️ Interaktiv Prediction")
    
    if 'model' in st.session_state:
        model = st.session_state['model']
        
        st.write("**Testa modellen med egna data:**")
        
        if 'data_clean' in st.session_state:
            X_sample = st.session_state['data_clean'].iloc[:1].copy()
            
            st.dataframe(X_sample, use_container_width=True)
            
            if st.button("Förutsäg"):
                pred = model.predict(X_sample)
                st.success(f"Prediktion: {pred[0]}")

st.divider()

st.markdown("""
## 🇸🇪 Svenska facktermer

| Engelska | Svenska |
|----------|---------|
| Pipeline | Flöde / Rörledning |
| Deployment | Idrifttagande |
| Monitoring | Övervakning |
| Risk assessment | Riskbedömning |
| Documentation | Dokumentation |
| Quality assurance | Kvalitetssäkring |
""")

st.markdown("---")
st.caption("Gy25 AI Curriculum | Nivå 2 | End-to-End Pipeline")
