import gym
import walking_spider
import pybullet as p
import pybullet_data
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


env = gym.make('WalkingSpider-v0')

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save("experience_learned/custom_experiences/ppo2_fall_1.1")