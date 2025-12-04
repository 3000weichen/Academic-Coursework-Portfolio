# Assignment 1 – Exploration & Dynamic Programming (Group Project)

This project covers **two core RL fundamentals**:

1️⃣ Exploration–Exploitation trade-off in **multi-armed bandits**  
2️⃣ **Dynamic Programming** (Policy Iteration & Value Iteration) in MDPs

---

## 🧪 Part 1 — Exploration (Bandits)
:contentReference[oaicite:0]{index=0}

We compared three action-selection strategies:

| Method | Strength | Weakness |
|--------|----------|---------|
| ε-greedy | Simple, converges well with moderate ε | Too small or too large ε harms performance |
| Optimistic Initialization | Forces early exploration | Over-optimism slows learning |
| UCB | Targets uncertain actions efficiently | Slower start with high exploration constant |

**Key finding**  
Moderate exploration (e.g., 𝜖 = 0.05–0.08) achieves the best balance of learning speed and final reward.

---

## 🧮 Part 2 — Dynamic Programming (Windy GridWorld)
:contentReference[oaicite:1]{index=1}

Implemented three agents:

| Agent | Result | Efficiency |
|-------|--------|------------|
| Policy Iteration | Optimal policy | Slowest |
| ε-greedy Policy Iteration | Near-optimal | Fast but slightly lower expected reward |
| Value Iteration | Optimal policy | Fastest convergence |

**Key finding**  
Value Iteration converges much faster while retaining optimality.

---

## 🎯 Overall Insights

- Exploration needs **balance**, not pure greediness nor pure randomness  
- DP algorithms **guarantee** optimal solutions when the model is perfect  
- Value Iteration proves more **efficient** than Policy Iteration

---

## 👥 Team
Hao Chen · Simone de Vos Burchart  
Group 88

Score: **8.3 / 10**

Course: *Introduction to Reinforcement Learning – Leiden University*
