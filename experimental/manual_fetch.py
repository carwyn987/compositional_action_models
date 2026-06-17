# manual_fetch_terminal_fixed.py

import sys
import select
import termios
import tty
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchReach-v4"
# ENV_ID = "FetchPickAndPlace-v4"

env = gym.make(ENV_ID, render_mode="human")
obs, info = env.reset(seed=0)

print("""
Click/focus this terminal, not the MuJoCo render window.

Controls:
  w/s : forward/back
  a/d : left/right
  r/f : up/down
  o/p : open/close gripper
  n   : reset
  q   : quit
""")

old_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())


def get_key():
    dr, _, _ = select.select([sys.stdin], [], [], 0.0)
    if dr:
        return sys.stdin.read(1)
    return None


try:
    step = 0

    while True:
        key = get_key()
        action = np.zeros(4, dtype=np.float32)

        if key == "q":
            break
        elif key == "n":
            obs, info = env.reset()
            continue
        elif key == "w":
            action[1] = 1.0
        elif key == "s":
            action[1] = -1.0
        elif key == "a":
            action[0] = -1.0
        elif key == "d":
            action[0] = 1.0
        elif key == "r":
            action[2] = 1.0
        elif key == "f":
            action[2] = -1.0
        elif key == "o":
            action[3] = 1.0
        elif key == "p":
            action[3] = -1.0

        if key is not None:
            print(f"read key: {repr(key)} -> action {action}")

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

        step += 1
        time.sleep(1 / 25)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    env.close()