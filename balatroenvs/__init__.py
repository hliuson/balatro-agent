import gymnasium as gym
import logging
gym.register(
    id='Balatro-blindonly-v0',
    entry_point='balatroenvs.blindonly:BlindEnv',
    max_episode_steps=100,
    reward_threshold=600,
)
