import gymnasium as gym
import gymnasium_robotics

# Registers the robotics environments
gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchReach-v4"
# Other options:
# "FetchPush-v4"
# "FetchSlide-v4"
# "FetchPickAndPlace-v4"

env = gym.make(ENV_ID, render_mode="human")

obs, info = env.reset(seed=0)

print("Observation keys:", obs.keys())
print("Observation shape:", obs["observation"].shape)
print("Achieved goal shape:", obs["achieved_goal"].shape)
print("Desired goal shape:", obs["desired_goal"].shape)
print("Action space:", env.action_space)

for step in range(500):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if step % 25 == 0:
        print(
            f"step={step:03d}",
            f"reward={reward}",
            f"success={info.get('is_success')}",
        )

    if terminated or truncated:
        obs, info = env.reset()

env.close()