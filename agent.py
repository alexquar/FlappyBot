#flappy bird docs: https://github.com/markub3327/flappy-bird-gymnasium

import flappy_bird_gymnasium
import gymnasium

#instance of flappy bird
env = gymnasium.make("CartPole-v1", render_mode="human")
# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

#reset env
obs, _ = env.reset()
while True:
    # Next action:
    # 0 for nothing 1 for flap 
    action = env.action_space.sample()

    # execute action
    #obs is the observation, reward is the score, terminated is if the game is over
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()