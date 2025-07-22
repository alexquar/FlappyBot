# FlappyBot 🤖

A PyTorch-based Deep Q-Network (DQN) agent trained to master **Flappy Bird** and other games offered on the **Gymnasium** platform using modern reinforcement learning techniques.

This project demonstrates how AI agents can learn **optimal policies** through self-play using **experience replay**, **epsilon-greedy exploration**, and advanced Q-learning strategies like **Double DQN** and **Dueling DQN**.

---

## 🎮 Overview

FlappyBot learns to play the classic side-scrolling game *Flappy Bird* by interacting with a Gym-style environment. Over thousands of episodes, the bot improves its decision-making by learning from rewards and penalties.

---

## 🚀 Features

✅ Deep Q-Learning  
✅ Experience Replay  
✅ Epsilon-Greedy Exploration  
✅ Target Network Syncing  
✅ Double DQN (optional)  
✅ Dueling DQN (optional)  
✅ PyTorch-based architecture  
✅ Fully configurable via YAML  
✅ Reward and epsilon tracking graphs  
✅ Model saving & loading

---

## 📦 Setup

### Prerequisites

- Python 3.7+
- `pip` package manager
- PyTorch

 
### Installation

```bash
git clone https://github.com/yourusername/flappybot.git
cd flappybot
pip install -r requirements.txt

### Running the bots/training

- Add config to hyperparams.yml for a new game 
- For easy training add the new game to the launch.json file and run from the run and debug tab of vscode 
- Model training info is added to the runs directory
- To run a model just run python agent.py {game_name}