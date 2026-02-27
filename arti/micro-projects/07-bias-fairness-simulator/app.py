"""
07 Bias & Fairness Simulator - Streamlit App
AI-bias detektering och EU AI Act
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bias & Fairness Simulator",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 07 Bias & Fairness Simulator – AI, Etik och EU AI Act")
st.markdown("**Lär dig identifiera och hantera AI-bias** | Gy25 AI Curriculum")

with st.sidebar:
    st.header("📚 Lärandemål")
    st.markdown("""
    - Identifiera bias i dataset
    - Använda fairness metrics
    - Förklara EU AI Act
    - Diskutera GDPR
    """)
    
    st.header("⚙️ Dataset")
    dataset_choice = st.selectbox(
        "Välj dataset",
        ["Swedish Student Data", "Credit Data", "Hiring Data"]
    )

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & Bias", "🔧 Träna modell", "📏 Fairness Metrics", "🇪🇺 EU AI Act"])

with tab1:
    st.header("📊 Utforska Data och Bias")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        ### Swedish Student Data
        
        Vi använder en simulerad dataset med svenska elever för att 
        förutsäga antagning till gymnasiet.
        
        **Features:**
        - gender (M/F)
        - birth_country (Sweden/Other)
        - parent_education (High/Medium/Low)
        - study_hours
        - math_grade, english_grade
        
        **Target:**
        - admitted (0/1)
        """)
        
        if st.button("Generera dataset"):
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
            })
            
            data.loc[data['parent_education'] == 'Low', 'math_grade'] -= 5
            data.loc[data['parent_education'] == 'Low', 'english_grade'] -= 5
            
            threshold = 60
            data['admitted'] = ((data['math_grade'] + data['english_grade']) / 2 > threshold).astype(int)
            
            data['gender_enc'] = (data['gender'] == 'M').astype(int)
            data['country_enc'] = (data['birth_country'] == 'Sweden').astype(int)
            data['edu_enc'] = data['parent_education'].map({'Low': 0, 'Medium': 1, 'High': 2})
            
            st.session_state['data'] = data
            st.success(f"✅ Genererat {n} elever")
    
    with col2:
        if 'data' in st.session_state:
            data = st.session_state['data']
            
            st.dataframe(data.head(10), use_container_width=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                fig = px.histogram(data, x='math_grade', color='birth_country', 
                                 title='Betyg i matte per födelseland')
                st.plotly_chart(fig, use_container_width=True)
                
            with col_b:
                fig = px.histogram(data, x='english_grade', color='birth_country',
                                 title='Betyg i engelska per födelseland')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Klicka på 'Generera dataset' för att börja")

    st.divider()
    
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        st.subheader("🔍 Analysera potential bias")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Antagningsgrad per födelseland:**")
            by_country = data.groupby('birth_country')['admitted'].mean()
            st.bar_chart(by_country)
            
        with col2:
            st.markdown("**Antagningsgrad per kön:**")
            by_gender = data.groupby('gender')['admitted'].mean()
            st.bar_chart(by_gender)
            
        with col3:
            st.markdown("**Antagningsgrad per föräldrautbildning:**")
            by_edu = data.groupby('parent_education')['admitted'].mean()
            st.bar_chart(by_edu)

with tab2:
    st.header("🔧 Träna din modell")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ Generera data först i fliken 'Data & Bias'!")
    else:
        data = st.session_state['data']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Modell-inställningar")
            
            include_country = st.checkbox("Inkludera födelseland som feature", value=False)
            include_gender = st.checkbox("Inkludera kön som feature", value=False)
            
            max_depth = st.slider("Max djup (Decision Tree)", 2, 10, 5)
            
            train_button = st.button("🚀 Träna modell", type="primary")
            
        with col2:
            if train_button:
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.model_selection import train_test_split
                
                if include_country:
                    features = ['study_hours', 'math_grade', 'english_grade', 'gender_enc', 'country_enc', 'edu_enc']
                else:
                    features = ['study_hours', 'math_grade', 'english_grade', 'gender_enc', 'edu_enc']
                
                X = data[features]
                y = data['admitted']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                
                data['prediction'] = model.predict(data[features])
                
                st.session_state['model'] = model
                st.session_state['features'] = features
                st.session_state['include_country'] = include_country
                
                st.success(f"✅ Träning klar! Accuracy: {accuracy:.1%}")
                
                st.subheader("📊 Feature Importance")
                importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                            title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👈 Konfigurera och klicka på 'Träna modell'")

with tab3:
    st.header("📏 Fairness Metrics")
    
    if 'data' not in st.session_state or 'model' not in st.session_state:
        st.warning("⚠️ Träna en modell först!")
    else:
        data = st.session_state['data']
        
        st.markdown("""
        ### Vad är Fairness Metrics?
        
        Fairness metrics mäter hur rättvist en modell behandlar olika grupper.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Välj skyddad attribut")
            
            protected_attr = st.selectbox(
                "Skyddad attribut",
                ["birth_country", "gender", "parent_education"]
            )
            
        with col2:
            st.markdown("#### Beräkna metrics")
            
            if st.button("Beräkna Fairness Metrics"):
                metrics = calculate_fairness_metrics(data, protected_attr)
                st.session_state['metrics'] = metrics
        
        if 'metrics' in st.session_state:
            metrics = st.session_state['metrics']
            
            st.divider()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Statistical Parity", f"{metrics['spd']:.3f}", 
                         delta_color="inverse" if abs(metrics['spd']) > 0.1 else "normal")
                
            with col2:
                st.metric("Disparate Impact", f"{metrics['di']:.3f}",
                         delta_color="inverse" if metrics['di'] < 0.8 else "normal")
                
            with col3:
                st.metric("Equal Opportunity", f"{metrics['eod']:.3f}",
                         delta_color="inverse" if abs(metrics['eod']) > 0.1 else "normal")
                
            with col4:
                st.metric("Accuracy Gap", f"{metrics['acc_gap']:.3f}",
                         delta_color="inverse" if abs(metrics['acc_gap']) > 0.1 else "normal")
            
            st.divider()
            
            st.subheader("📊 Visualisera Bias")
            
            if protected_attr == 'birth_country':
                groups = ['Sweden', 'Other']
                group_col = 'birth_country'
            elif protected_attr == 'gender':
                groups = ['M', 'F']
                group_col = 'gender'
            else:
                groups = ['High', 'Medium', 'Low']
                group_col = 'parent_education'
            
            fig = go.Figure()
            
            for group in groups:
                group_data = data[data[group_col] == group]
                preds = group_data['prediction'].value_counts(normalize=True)
                fig.add_trace(go.Bar(
                    name=group,
                    x=['Ej antagen', 'Antagen'],
                    y=[1 - preds.get(1, 0), preds.get(1, 0)]
                ))
            
            fig.update_layout(barmode='group', title=f'Antagningsfördelning per {protected_attr}')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Tolkning av Metrics
            
            | Metric | Beskrivning | Ideal |
            |--------|-------------|-------|
            | Statistical Parity Difference | Skillnad i positive rate | 0.0 |
            | Disparate Impact | Ratio mellan grupper | 1.0 |
            | Equal Opportunity Difference | Skillnad i TPR | 0.0 |
            | Accuracy Gap | Skillnad i noggrannhet | 0.0 |
            
            **Varning:** Om metricarna visar bias (>0.1 från ideal), behöver modellen justeras!
            """)

with tab4:
    st.header("🇪🇺 EU AI Act och GDPR")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### EU AI Act – Högrisk-system
        
        Enligt EU AI Act klassificeras vissa AI-system som **högrisk** och 
        måste uppfylla strikta krav:
        
        | Område | Exempel |
        |--------|---------|
        | Utbildning | Antagningssystem, betygsbedömning |
        | Arbetsmarknad | Rekrytering, befordran |
        | Grundläggande tjänster | Krediter, försäkringar |
        | Rättsväsende | Riskbedömning |
        
        ### Krav för högrisk-system
        
        1. ✅ Riskbedömning före deployment
        2. ✅ Datahantering (representativ data)
        3. ✅ Teknisk dokumentation
        4. ✅ Loggning och spårbarhet
        5. ✅ Transparens till användare
        6. ✅ Mänsklig översyn
        7. ✅ Noggrannhet och robusthet
        """)
        
    with col2:
        st.markdown("""
        ### GDPR – Artikel 22
        
        > *"Den registrerade har rätt att inte bli föremål för ett beslut 
        > som enbart grundas på automatiserad behandling ... och som 
        > har rättsliga följder eller på liknande sätt betydande påverkan."*
        
        ### Praktiska krav
        
        | Situation | Krav |
        |-----------|------|
        | Antagning till gymnasiet | Mänsklig granskning |
        | Kreditbedömning | Rätt till manuell översyn |
        | AI-granskning | Elever måste få veta hur det fungerar |
        
        ### Din rättighet
        
        - Rätt till information om hur AI:n fungerar
        - Rätt att begära manuell granskning
        - Rätt att protestera mot automatiserade beslut
        """)
    
    st.divider()
    
    st.subheader("🇸🇪 Svenska tillämpningar")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Skolverket
        - Antagningssystem till gymnasiet
        - Betygsbedömning
        - Likvärdighetsprüvning
        """)
        
    with col2:
        st.markdown("""
        ### SCB
        - Registerdata för AI-träning
        - Integritetsskydd
        - Anonymisering
        """)
        
    with col3:
        st.markdown("""
        ### Trafikverket
        - Trafikprediktion
        - Självkörande fordon
        - Infrastruktur
        """)
    
    st.divider()
    
    st.subheader("📝 Checklist: Är din AI rättvis?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Har du analyserat träningsdata för underrepresentation?")
        st.checkbox("Har du testat modellen på olika grupper?")
        st.checkbox("Har du beräknat fairness metrics?")
        
    with col2:
        st.checkbox("Finns det mänsklig översyn av besluten?")
        st.checkbox("Är modellen dokumenterad?")
        st.checkbox("Har du gjort en riskanalys?")

def calculate_fairness_metrics(data, protected_attr):
    """Beräkna fairness metrics"""
    
    if protected_attr == 'birth_country':
        privileged = 'Sweden'
        unprivileged = 'Other'
    elif protected_attr == 'gender':
        privileged = 'M'
        unprivileged = 'F'
    else:
        privileged = 'High'
        unprivileged = 'Low'
    
    priv_data = data[data[protected_attr] == privileged]
    unpriv_data = data[data[protected_attr] == unprivileged]
    
    priv_positive_rate = priv_data['prediction'].mean()
    unpriv_positive_rate = unpriv_data['prediction'].mean()
    
    priv_positive_actual = priv_data[data['admitted'] == 1]['prediction'].mean()
    unpriv_positive_actual = unpriv_data[data['admitted'] == 1]['prediction'].mean()
    
    priv_accuracy = (priv_data['prediction'] == priv_data['admitted']).mean()
    unpriv_accuracy = (unpriv_data['prediction'] == unpriv_data['admitted']).mean()
    
    return {
        'spd': priv_positive_rate - unpriv_positive_rate,
        'di': unpriv_positive_rate / (priv_positive_rate + 1e-8),
        'eod': priv_positive_actual - unpriv_positive_actual,
        'acc_gap': priv_accuracy - unpriv_accuracy
    }

st.divider()

st.markdown("""
## 🇸🇪 Svenska facktermer

| Engelska | Svenska |
|----------|---------|
| Bias | Partiskhet / snedvridning |
| Fairness | Rättvisa |
| Protected attribute | Skyddad egenskap |
| Disparate impact | Ojämlik påverkan |
| Statistical parity | Statistisk jämlikhet |
| Equal opportunity | Lika möjligheter |
""")

st.markdown("---")
st.caption("Gy25 AI Curriculum | Nivå 2 | Bias & Fairness Simulator")
