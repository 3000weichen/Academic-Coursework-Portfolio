# Assignment 3 – Model-based Reinforcement Learning (Group Project)

We extend RL learning to **model-based planning** using:

- **Dyna-Q** – combines Q-learning with simulated planning updates
- **Prioritized Sweeping (PS)** – focuses updates on transitions with the largest TD error

These agents are evaluated in **Windy GridWorld** under deterministic (wind=1.0) and stochastic (wind=0.9) settings. :contentReference[oaicite:0]{index=0}

---

## Key Work

- Implemented learned transition + reward models from real experience
- Varied planning updates {1, 3, 5} to analyze sample efficiency
- Compared learning curves and runtime vs. model-free Q-learning baseline
- Examined adaptability under changing wind conditions

---

## Main Insights

- **More planning steps → much faster learning**
- **Prioritized Sweeping** typically outperforms Dyna in both:
  - **Final performance**
  - **Runtime efficiency (~3× faster)**
- In stochastic environments, **over-planning** may hurt final
