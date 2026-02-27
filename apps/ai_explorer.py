import streamlit as st
import os

st.set_page_config(page_title="Gy25 AI Explorer", page_icon="🤖", layout="wide")

st.title("🤖 Gy25 AI Curriculum Explorer")
st.markdown("**Svensk gymnasium – Artificiell intelligens (Gy25)** – Nivå 1 & 2")

tab1, tab2, tab3, tab4 = st.tabs(["📚 Nivå 1", "🚀 Nivå 2", "⚖️ Etik Arena", "📁 Portfolio"])

with tab1:
    st.header("Nivå 1 – Grundläggande (**4 projekt KLARA** ✅)")
    projects = [
        ("01 Data Detective", "Datakvalitet, bias, rensning", "micro-projects/01-data-detective/app.py"),
        ("02 Decision Tree Detective", "Beslutsträd, transparens", "micro-projects/02-decision-tree-detective/app.py"),
        ("03 Search & Game Agent", "Sökning + Spelagent (Tic-Tac-Toe)", "micro-projects/03-search-game-agent/app.py"),
        ("04 Sentiment NLP Lab", "NLP, generativ AI, etik", "micro-projects/04-sentiment-nlp-lab/app.py"),
    ]
    for i, (name, desc, path) in enumerate(projects, 1):
        with st.expander(f"**{i}. {name}**"):
            st.write(desc)
            if path and os.path.exists(path):
                st.markdown(f"[🚀 Starta interaktiv demo]({path})")
            else:
                st.info("Kommer snart")

with tab2:
    st.header("Nivå 2 – Avancerat (**6 projekt KLARA** ✅)")
    
    projects_n2 = [
        ("05 MNIST Vision Basics", "Objektigenkänning med CNN", "micro-projects/05-mnist-vision-basics/app.py"),
        ("06 Simple RL Robot", "Förstärkande inlärning, Q-learning", "micro-projects/06-simple-rl-robot/app.py"),
        ("07 Bias & Fairness Simulator", "EU AI Act, GDPR, fairness", "micro-projects/07-bias-fairness-simulator/app.py"),
        ("08 Neural Net from Scratch", "MLP + SHAP, förklarbarhet", "micro-projects/08-neural-net-scratch/app.py"),
        ("09 End-to-End Pipeline", "ML-pipeline, deployment, risk", "micro-projects/09-end-to-end-pipeline/app.py"),
        ("10 Domain Capstone", "Självvalt domänprojekt", "micro-projects/10-domain-capstone/app.py"),
    ]
    
    for i, (name, desc, path) in enumerate(projects_n2, 5):
        with st.expander(f"**{i}. {name}**"):
            st.write(desc)
            if path and os.path.exists(path):
                st.markdown(f"[🚀 Starta interaktiv demo]({path})")
            else:
                st.warning("Projektmapp hittades inte")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**✅ Nivå 2 KLAR:** 05-10")
        st.markdown("""
        - MNIST Vision (CNN)
        - RL Robot (Q-learning)
        - Bias & Fairness
        - Neural Net + SHAP
        - End-to-End Pipeline
        - Domain Capstone
        """)
    
    with col2:
        st.info("**🎉 FULLSTÄNDIG KURS!**")
        st.markdown("""
        10/10 projekt klara!
        Hela Gy25 AI Curriculum
        är nu redo för klassrummet.
        """)

with tab3:
    st.header("⚖️ Etik Arena")
    st.markdown("[Läs Ethics Framework](./ethics-framework.md)")
    st.markdown("[Bias Simulator](./micro-projects/07-bias-fairness-simulator/app.py)")
    
    st.subheader("🇪🇺 EU AI Act")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Högrisk-system
        - Utbildning (antagning)
        - Arbetsmarknad (rekrytering)
        - Sjukvård (diagnostik)
        """)
        
    with col2:
        st.markdown("""
        ### Krav
        - Riskbedömning
        - Datahantering
        - Transparens
        - Mänsklig översyn
        """)
        
    with col3:
        st.markdown("""
        ### GDPR Art. 22
        - Rätt till manuell översyn
        - Rätt till förklaring
        - Förbud mot helt automatiska beslut
        """)

with tab4:
    st.header("📁 Min Portfolio")
    st.checkbox("01 Data Detective – klar")
    st.checkbox("02 Decision Tree Detective – klar")
    st.checkbox("03 Search & Game Agent – klar")
    st.checkbox("04 Sentiment NLP Lab – klar")
    st.checkbox("05 MNIST Vision Basics – klar")
    st.checkbox("06 Simple RL Robot – klar")
    st.checkbox("07 Bias & Fairness Simulator – klar")
    st.checkbox("08 Neural Net from Scratch – klar")
    st.checkbox("09 End-to-End Pipeline – klar")
    st.checkbox("10 Domain Capstone – klar")
    st.checkbox("Comparison Matrix uppdaterad")
    st.checkbox("Fullständig portfolio – klar")

st.sidebar.title("Snabbstart")
if st.sidebar.button("Förbered dataset"):
    st.sidebar.success("Kör: python scripts/prepare_datasets.py")
st.sidebar.markdown("**Nivå 1 KLARA:** 01 + 02 + 03 + 04")
st.sidebar.markdown("**Nivå 2 KLARA:** 05 + 06 + 07 + 08 + 09 + 10")
st.sidebar.markdown("**FULL KURS:** 10/10 projekt")
