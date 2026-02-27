# 06 Simple RL Robot – Förstärkande Inlärning

**Nivå:** 1-2  
**Tid:** 90-120 minuter  
**Fokus:** Reinforcement Learning, Q-learning, Robotik  
**Miljö:** CartPole, Gridworld

## Lärandemål

Efter detta projekt kan du:
- Förklara skillnaden mellan övervakat, oövervakat och förstärkande lärande
- Implementera en Q-learning agent
- Träna en robot att balansera en pinne (CartPole)
- Diskutera möjligheter och risker med robotik

## Koppling till Skolverket Gy25

| Centralt innehåll | Täckning |
|------------------|----------|
| Översikt över användningen av AI (robotik) | ✅ Huvudfokus |
| Principer för maskininlärning och robotik | ✅ Q-learning |
| Vikten av data (belöningsfunktion) | ✅ Centralt |

## 🇸🇪 Svenska sammanhang

- **Trafikverket**: Självkörande fordon, trafikoptimering
- **Industri**: Robotar i tillverkning
- **Sjukvård**: Robotkirurgi, rehabilitering

## 🇪🇺 EU AI Act-koppling

- Högrisk-system för robotkirurgi (bilaga III)
- Säkerhetskrav för autonoma fordon
- Dokumentationskrav för beslutsfattande system

---

## Snabbstart

```bash
# Installera extra beroenden
pip install gymnasium numpy

# Starta appen
streamlit run micro-projects/06-simple-rl-robot/app.py
```

---

## Introduktion

### Vad är förstärkande inlärning?

**Reinforcement Learning (RL)** är en typ av maskininlärning där en "agent" lär sig genom att interagera med en miljö:

```
┌─────────────────────────────────────────────────────────┐
│                    RL Process                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│    ┌───────┐    Action     ┌────────────┐              │
│    │ Agent │ ─────────────▶│   Miljö     │              │
│    └───────┘               └────────────┘              │
│         ▲                       │                      │
│         │     Reward            │                      │
│         │     (Bonus/Straff)    │                      │
│         └───────────────────────┘                      │
│                                                         │
│    Agent: "Vad ska jag göra för att få mer belöning?" │
└─────────────────────────────────────────────────────────┘
```

### Skillnaden från andra ML-typer

| Typ | Vad den lär sig | Exempel |
|-----|-----------------|---------|
| **Övervakad** | Från märkta exempel | Spam-filter |
| **Oövervakad** | Från omärkta mönster | Kundsegment |
| **Förstärkande** | Från trial-and-error | Spel, robotik |

---

## Steg 1: Installera och utforska Gymnasium

### Vad är Gymnasium?

Gymnasium (tidigare OpenAI Gym) är en miljö för att träna RL-agenter. Den innehåller många klassiska problem:

- **CartPole**: Balansera en pinne på en vagn
- **MountainCar**: Köra en bil uppför ett berg
- **Atari-spel**: Spela gammaldags videospel

### Installera

```bash
pip install gymnasium numpy
```

### Testa CartPole-miljön

```python
import gymnasium as gym

env = gym.make('CartPole-v1')

# Reset miljön
state = env.reset()
print(f"Initial state: {state}")

# Ta ett slumpmässigt steg
action = 0  # 0 = vänster, 1 = höger
next_state, reward, done, info = env.step(action)

print(f"Next state: {next_state}")
print(f"Reward: {reward}")
print(f"Done: {done}")
```

---

## Steg 2: Q-Learning Grunderna

### Q-table

En Q-table lagrar "kvaliteten" av varje action i varje state:

```
         Action
State    | Vänster  | Höger
---------|-----------|--------
[0,0,0,0]|   0.0    |  0.0
[0,1,0,0]|   0.5    |  0.3
[1,0,0,1]|   0.2    |  0.8
```

### Q-Learning Algoritm

```python
import numpy as np
import random

# Hyperparametrar
alpha = 0.1      # Inlärningshastighet
gamma = 0.99     # Diskonteringsfaktor
epsilon = 1.0    # Exploration rate

# Initialize Q-table
q_table = np.zeros((state_bins, state_bins, action_bins, n_actions))

def choose_action(state):
    """Epsilon-greedy policy"""
    if random.random() < epsilon:
        return random.choice([0, 1])  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

def update_q(state, action, reward, next_state):
    """Q-learning update"""
    best_next = np.max(q_table[next_state])
    q_table[state, action] += alpha * (reward + gamma * best_next - q_table[state, action])
```

---

## Steg 3: Träna agenten (CartPole)

### Komplett träningskod

```python
import gymnasium as gym
import numpy as np
import random

env = gym.make('CartPole-v1')

# Discretize continuous state to bins
def discretize(state, bins=10):
    state_min = env.observation_space.low
    state_max = env.observation_space.high
    ratios = (state - state_min) / (state_max - state_min)
    return (ratios * bins).astype(int).clip(0, bins - 1)

# Q-learning parametrar
n_bins = 10
n_actions = env.action_space.n
q_table = np.zeros((n_bins, n_bins, n_bins, n_bins, n_actions))

alpha = 0.1      # Inlärningshastighet
gamma = 0.99     # Diskonteringsfaktor
epsilon = 1.0     # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 500

for episode in range(episodes):
    state = env.reset()[0]
    state = discretize(state)
    
    done = False
    total_reward = 0
    
    while not done:
        # Epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize(next_state)
        
        # Q-learning update
        best_next = np.max(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * best_next - q_table[state][action])
        
        state = next_state
        total_reward += reward
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if episode % 50 == 0:
        print(f"Episode {episode}: Total reward = {total_reward}, Epsilon = {epsilon:.3f}")

env.close()
```

---

## Steg 4: Visualisera träningen

### Plot reward over time

```python
import matplotlib.pyplot as plt
import pandas as pd

# Spara rewards under träningen
rewards = []

# Efter träningen:
plt.figure(figsize=(10, 5))
plt.plot(pd.Series(rewards).rolling(50).mean())
plt.title('Belöning över tid (rörande medelvärde)')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
```

---

## Steg 5: Testa den tränade agenten

```python
# Efter träningen - kör agenten
env = gym.make('CartPole-v1', render_mode='human')

for episode in range(5):
    state = env.reset()[0]
    state = discretize(state)
    done = False
    total_reward = 0
    
    while not done:
        action = np.argmax(q_table[state])  # Alltid välj bästa
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize(next_state)
        
        state = next_state
        total_reward += reward
    
    print(f"Episode {episode + 1}: Total reward = {total_reward}")

env.close()
```

---

## Gridworld: Enklare RL-problem

### Vad är Gridworld?

Ett enkelt 2D-rutnät där agenten ska lära sig navigera:

```
┌───┬───┬───┬───┐
│ S │   │   │ G │  S = Start, G = Goal
├───┼───┼───┼───┤
│   │ 🔒│   │ 🔥│  🔒 = Hinder, 🔥 = Fara
├───┼───┼───┼───┤
│   │   │   │   │
└───┴───┴───┴───┘
```

### Q-table för Gridworld

```python
# 4x4 grid med 4 actions (upp, ner, vänster, höger)
q_table = np.zeros((4, 4, 4))

# Belöningar
goal_reward = 100
step_reward = -1
pit_reward = -100
```

---

## Jämförelse: Olika RL-algoritmer

| Algoritm | Fördelar | Nackdelar | Användning |
|----------|----------|-----------|------------|
| **Q-Learning** | Enkel, offline | Sakta, state'space explosion | Små problem |
| **Deep Q-Learning** | Kan hantera stora states | Instabil | Atari-spel |
| **Policy Gradient** | Kontinuerliga actions | Hög varians | Robotik |
| **PPO/SAC** | Stabil, effektiv | Komplex | S state-of-the-art |

---

## Etiska överväganden

### 🚨 Potentiella problem

1. **Säkerhet**: Vad händer om roboten gör fel?
2. **Ansvar**: Vem är ansvarig när AI:n skadar någon?
3. **Jobb**: Vad händer med arbetare som ersätts av robotar?
4. **Militär**: Autonoma vapen

### 🇸🇪 Svenska exempel

- **Volvo**: Självkörande bilar i Göteborg
- **IKEA**: Robotar i lagerhantering
- **Karolinska**: Robotkirurgi

---

## Reflektionsfrågor

### Grundläggande (E-nivå)

1. Vad är skillnaden mellan "exploration" och "exploitation"?
2. Vad är en belöningsfunktion?
3. Varför behöver agenten "trial and error"?

### Utmanande (C-nivå)

1. Hur påverkar epsilon-värdet träningen?
2. Vad händer om belöningen är feldesignad?
3. Hur kan RL användas för trafikoptimering?

### Avancerat (A-nivå)

1. Vilka EU AI Act-krav gäller för autonoma fordon?
2. Hur skiljer sig Deep Q-Learning frånvanlig Q-Learning?
3. Vad är "reward hacking" och hur undviker man det?

---

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

---

## Nästa steg

- **Projekt 05**: MNIST Vision – bildigenkänning
- **Projekt 07**: Bias & Fairness – etik och EU AI Act
- **Projekt 08**: Neural Net from Scratch – bygg ditt eget nätverk

---

## Resurser

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- [EU AI Act - Robotics](https://artificialintelligenceact.eu/)

---

*Projektet är en del av Gy25 AI Curriculum – Artificiell Intelligens för Svenska Gymnasium*
