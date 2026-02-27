"""
06 Simple RL Robot - Streamlit App
Förstärkande inlärning med Q-Learning och CartPole
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import random
import time

st.set_page_config(
    page_title="Simple RL Robot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 06 Simple RL Robot – Förstärkande Inlärning")
st.markdown("**Lär dig hur AI lär sig genom trial-and-error** | Gy25 AI Curriculum")

with st.sidebar:
    st.header("📚 Lärandemål")
    st.markdown("""
    - Förklara RL-grunder
    - Implementera Q-learning
    - Träna en CartPole-agent
    - Jämföra med andra ML-typer
    """)
    
    st.header("⚙️ Välj miljö")
    env_choice = st.selectbox("Miljö", ["CartPole-v1", "Gridworld (enkel)"])

tab1, tab2, tab3, tab4 = st.tabs(["📖 Teori", "🔧 Träna", "🎮 Testa", "⚖️ Etik"])

with tab1:
    st.header("📖 Förstärkande Inlärning – Teori")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Vad är Reinforcement Learning?
        
        **Förstärkande inlärning (RL)** är en typ av maskininlärning där en 
        **agent** lär sig genom att interagera med en **miljö** och få 
        **belöningar** (rewards).
        
        ### RL-komponenter
        
        | Komponent | Beskrivning |
        |-----------|-------------|
        | **Agent** | AI:n som tar beslut |
        | **Miljö** | Världen agenten interagerar med |
        | **Action** | Vad agenten kan göra |
        | **State** | Nuvarande situation |
        | **Reward** | Feedback ( positiv/negativ) |
        
        ### Skillnaden från andra ML-typer
        
        | Typ | Träningsdata | Lär från |
        |-----|--------------|----------|
        | Övervakad | Märkta exempel | labels |
        | Oövervakad | Omärkta data | mönster |
        | **Förstärkande** | **Ingen data!** | **trial-and-error** |
        """)
        
    with col2:
        st.markdown("""
        ### RL Process
        
        ```
        ┌────────────────────────────────────┐
        │         RL Loop                    │
        ├────────────────────────────────────┤
        │                                    │
        │    ┌───────┐    Action            │
        │    │ Agent │ ───────────┐         │
        │    └───────┘             ▼         │
        │         ▲                ┌───────┐ │
        │         │   Reward       │Miljö  │ │
        │         └─────────────── │       │ │
        │                          └───────┘ │
        └────────────────────────────────────┘
        ```
        
        ### Q-Learning
        
        Q-Learning är en populär RL-algoritm:
        
        1. **Initiera** Q-table med nollor
        2. **Välj** action (epsilon-greedy)
        3. **Få** reward och nytt state
        4. **Uppdatera** Q-värde
        5. **Upprepa**
        
        ### Q-update formel
        
        $$Q(s,a) = Q(s,a) + \\alpha \\cdot [r + \\gamma \\cdot max(Q(s',a')) - Q(s,a)]$$
        
        - $\\alpha$ = inlärningshastighet
        - $\\gamma$ = diskonteringsfaktor
        - $r$ = reward
        """)
    
    st.divider()
    
    st.subheader("🎯 CartPole-problemet")
    st.markdown("""
    **Mål:** Balansera en pinne på en vagn genom att flytta vagnen vänster eller höger.
    
    - **State:** 4 värden (vagn-position, vagn-hastighet, pinne-vinkel, pinne-hastighet)
    - **Actions:** 2 (push left, push right)
    - **Reward:** +1 för varje steg pinnen är balanserad
    - **Done:** När vinkeln > 15° eller position > 2.4
    """)
    
    st.info("💡 **Svenskt sammanhang:** Samma teknik används för självkörande bilar och industrirobotar!")

with tab2:
    st.header("🔧 Träna din RL-agent")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Träningsinställningar")
        
        n_episodes = st.slider("Antal episoder", 100, 1000, 500)
        epsilon_start = st.slider("Start epsilon (utforskning)", 0.1, 1.0, 1.0)
        epsilon_decay = st.slider("Epsilon decay", 0.9, 0.999, 0.995)
        epsilon_min = st.slider("Min epsilon", 0.01, 0.1, 0.01)
        alpha = st.slider("Alpha (inlärning)", 0.01, 0.5, 0.1)
        gamma = st.slider("Gamma (diskontering)", 0.9, 0.999, 0.99)
        
        train_button = st.button("🚀 Starta träning", type="primary")
        
    with col2:
        if train_button:
            try:
                import gymnasium as gym
                
                env = gym.make('CartPole-v1')
                
                n_bins = 10
                n_actions = env.action_space.n
                q_table = np.zeros((n_bins, n_bins, n_bins, n_bins, n_actions))
                
                def discretize(state):
                    state_min = np.array([-2.4, -3.0, -0.21, -3.0])
                    state_max = np.array([2.4, 3.0, 0.21, 3.0])
                    ratios = (state - state_min) / (state_max - state_min + 1e-8)
                    return (ratios * (n_bins - 1)).astype(int).clip(0, n_bins - 1)
                
                epsilon = epsilon_start
                rewards = []
                recent_rewards = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for episode in range(n_episodes):
                    state = env.reset()[0]
                    state = discretize(state)
                    
                    done = False
                    total_reward = 0
                    
                    while not done:
                        if random.random() < epsilon:
                            action = env.action_space.sample()
                        else:
                            action = np.argmax(q_table[state])
                        
                        next_state, reward, done, _, _ = env.step(action)
                        next_state = discretize(next_state)
                        
                        best_next = np.max(q_table[next_state])
                        q_table[state][action] += alpha * (reward + gamma * best_next - q_table[state][action])
                        
                        state = next_state
                        total_reward += reward
                    
                    epsilon = max(epsilon_min, epsilon * epsilon_decay)
                    rewards.append(total_reward)
                    recent_rewards.append(np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards))
                    
                    if episode % (n_episodes // 10) == 0:
                        progress_bar.progress(episode / n_episodes)
                        status_text.text(f"Episode {episode}/{n_episodes}: Reward = {total_reward:.1f}, Epsilon = {epsilon:.3f}")
                
                env.close()
                
                progress_bar.progress(1.0)
                status_text.text("✅ Träning klar!")
                
                st.session_state['q_table'] = q_table
                st.session_state['rewards'] = rewards
                st.session_state['recent_rewards'] = recent_rewards
                
                st.success(f"✅ Träning klar! Genomsnittlig belöning (sista 50): {np.mean(rewards[-50:]):.1f}")
                
            except ImportError:
                st.error("Installera gymnasium: pip install gymnasium")
            except Exception as e:
                st.error(f"Fel: {e}")
        else:
            st.info("👈 Konfigurera och klicka på 'Starta träning'")
            
            st.markdown("### Q-Table struktur")
            st.code("""
# För CartPole med 10 bins per dimension:
q_table.shape = (10, 10, 10, 10, 2)
# = 10,000 states × 2 actions = 20,000 Q-värden
            """, language="python")

    if 'rewards' in st.session_state:
        st.divider()
        st.subheader("📈 Träningshistorik")
        
        rewards = st.session_state['rewards']
        recent = st.session_state['recent_rewards']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=rewards, name='Belöning', mode='lines', opacity=0.3))
            fig.add_trace(go.Scatter(y=recent, name='Rörande medel (50)', mode='lines', line=dict(width=2)))
            fig.update_layout(title='Belöning per episode', xaxis_title='Episode', yaxis_title='Total Reward')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            df = pd.DataFrame({
                'Metric': ['Max reward', 'Genomsnitt', 'Senaste 50', 'Total episoder'],
                'Value': [max(rewards), np.mean(rewards), np.mean(rewards[-50:]), len(rewards)]
            })
            st.dataframe(df, hide_index=True)

with tab3:
    st.header("🎮 Testa din agent")
    
    if 'q_table' not in st.session_state:
        st.warning("⚠️ Träna en agent först i fliken 'Träna'!")
    else:
        st.markdown("### Köra agenten (simulation)")
        
        run_button = st.button("▶️ Kör 5 episoder", type="primary")
        
        if run_button:
            try:
                import gymnasium as gym
                
                env = gym.make('CartPole-v1', render_mode='human')
                q_table = st.session_state['q_table']
                n_bins = 10
                
                def discretize(state):
                    state_min = np.array([-2.4, -3.0, -0.21, -3.0])
                    state_max = np.array([2.4, 3.0, 0.21, 3.0])
                    ratios = (state - state_min) / (state_max - state_min + 1e-8)
                    return (ratios * (n_bins - 1)).astype(int).clip(0, n_bins - 1)
                
                results = []
                
                for ep in range(5):
                    state = env.reset()[0]
                    state = discretize(state)
                    
                    done = False
                    total_reward = 0
                    steps = 0
                    
                    while not done:
                        action = np.argmax(q_table[state])
                        next_state, reward, done, _, _ = env.step(action)
                        next_state = discretize(next_state)
                        
                        state = next_state
                        total_reward += reward
                        steps += 1
                        
                        if done:
                            break
                    
                    results.append({'Episode': ep + 1, 'Steps': steps, 'Reward': total_reward})
                    st.success(f"Episode {ep + 1}: {steps} steg, {total_reward} reward")
                
                env.close()
                
                st.session_state['test_results'] = results
                
            except ImportError:
                st.error("Installera gymnasium: pip install gymnasium")
        else:
            st.info("Klicka på 'Kör 5 episoder' för att se agenten i aktion")
            
        if 'test_results' in st.session_state:
            results = st.session_state['test_results']
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            fig = px.bar(df, x='Episode', y='Steps', title='Antal steg per episode')
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("⚖️ Etik och Samhälle")
    
    st.markdown("""
    ### 🤖 Robotik och AI i samhället
    
    Förstärkande inlärning används för att träna robotar och autonoma system. 
    Men det finns viktiga etiska frågor att diskutera:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Risker")
        
        risks = {
            "Säkerhet": "Vad händer om roboten gör ett fel som skadar någon?",
            "Ansvar": "Vem är ansvarig när AI:n orsakar en olycka?",
            "Jobb": "Vad händer med arbetare som ersätts av robotar?",
            "Militär": "Autonoma vapen är kontroversiella",
            "Bias": "Kan RL-agenter lära sig oönskade beteenden?"
        }
        
        for risk, desc in risks.items():
            with st.expander(f"⚠️ {risk}"):
                st.markdown(desc)
                
    with col2:
        st.subheader("💡 Möjligheter")
        
        opportunities = {
            "Sjukvård": "Robotkirurgi, rehabilitering",
            "Transport": "Självkörande bilar, trafikoptimering",
            "Industri": "Effektivare tillverkning",
            "Forskning": "Nya upptäckter genom AI-experiment",
            "Miljö": "Optimerad energianvändning"
        }
        
        for opp, desc in opportunities.items():
            with st.expander(f"✅ {opp}"):
                st.markdown(desc)

    st.divider()
    
    st.subheader("🇸🇪 Svenska exempel")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🚗 Trafikverket
        - Självkörande fordon
        - Trafikflödesoptimering
        - Väderprognoser för vägar
        """)
        
    with col2:
        st.markdown("""
        ### 🏭 Industri
        - ABB robotar
        - Automatisering i tillverkning
        - Lagerhantering
        """)
        
    with col3:
        st.markdown("""
        ### 🏥 Sjukvård
        - Robotkirurgi (Karolinska)
        - AI för medicinsk bildanalys
        - Rehabrobotar
        """)

    st.divider()
    
    st.subheader("🇪🇺 EU AI Act och robotik")
    
    st.markdown("""
    Enligt EU AI Act klassificeras vissa robotiksystem som **högrisk**:
    
    | System | Risk-nivå | Krav |
    |--------|-----------|------|
    | Robotkirurgi | Hög | CE-märkning, dokumentation |
    | Självkörande fordon | Hög | Säkerhetsbedömning |
    | Industrirobotar | Medium | Transparens |
    | Leksaker | Förbjudet | --- |
    """)

st.divider()

st.markdown("""
## 🇸🇪 Svenska facktermer

| Engelska | Svenska |
|----------|---------|
| Reinforcement Learning | Förstärkande inlärning |
| Agent | Agent |
| Environment | Miljö |
| Reward | Belöning |
| Q-learning | Q-inlärning |
| Exploration | Utforskning |
| Exploitation | Utnyttjande |
| Policy | Strategi |
| Episode | Episode |
| Episode | Epok |
""")

st.markdown("---")
st.caption("Gy25 AI Curriculum | Nivå 1-2 | Simple RL Robot")
