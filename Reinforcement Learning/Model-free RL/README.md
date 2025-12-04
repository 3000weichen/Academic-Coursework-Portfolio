# Assignment 2 – Model-free Reinforcement Learning (Group Project)

Goal  
Compare several **model-free RL algorithms** in a cliff-walking Shortcut environment and study how hyperparameters affect learning. :contentReference[oaicite:0]{index=0}  

Environment & Code  
- `ShortCutEnvironment.py` – 12×12 gridworld with cliffs and optional wind
- `ShortCutAgents.py` – implementations of Q-learning, SARSA, Expected SARSA, n-step SARSA
- `ShortCutExperiment.py` – runs experiments, logs rewards and renders learned policies

What we did  
- Implemented **Q-learning** and **SARSA** with ε-greedy exploration  
- Added **Expected SARSA** and **n-step SARSA** (n = 1, 2, 5, 10, 25)  
- Ran large-scale experiments (100 runs × 1000 episodes) for multiple learning rates α  
- Compared average cumulative rewards and learned paths, including a windy variant of the environment

Main findings  
- Higher α → faster learning but more variance; α ≈ 0.5 works best for SARSA and Expected SARSA  
- Expected SARSA achieves the best performance among single-step agents  
- Small n (1–2) in n-step SARSA outperforms larger n and beats Q-learning in our setting  
- SARSA tends to learn **safer paths** around cliffs, especially in windy environments

Team  
Hao Chen · Simone de Vos Burchart (Group 80)  
Score: **9 / 10**
