# manual_fetch_terminal_fixed.py

import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

import numpy as np

from fetch_blockworld.envs import make_skill_env


SKILL = "pickup"
STATE_FILE = Path("live_facts.txt")
STATE_UPDATE_PERIOD = 0.10  # Update dashboard at 10 Hz


def get_key() -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    return sys.stdin.read(1) if readable else None


def format_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=4, suppress_small=True)

    if isinstance(value, (list, tuple)):
        return np.array2string(
            np.asarray(value),
            precision=4,
            suppress_small=True,
        )

    if isinstance(value, (np.floating, float)):
        return f"{float(value):.5f}"

    return str(value)


def write_state_dashboard(
    *,
    step: int,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> None:
    """Atomically rewrite the live-state dashboard file."""
    facts = info.get("facts", {})
    numeric_state = info.get("numeric_state", {})

    lines = [
        "FETCH LIVE STATE",
        "=" * 52,
        f"step:       {step}",
        f"reward:     {reward:.5f}",
        f"terminated: {terminated}",
        f"truncated:  {truncated}",
        f"success:    {bool(info.get('is_success', False))}",
        f"action:     {np.array2string(action, precision=3)}",
        "",
        "FACTS",
        "-" * 52,
    ]

    for name, value in sorted(facts.items()):
        marker = "TRUE " if bool(value) else "false"
        lines.append(f"{marker:5}  {name}")

    lines.extend(
        [
            "",
            "NUMERIC STATE",
            "-" * 52,
        ]
    )

    for name, value in sorted(numeric_state.items()):
        lines.append(f"{name:24} {format_value(value)}")

    lines.append("")

    # Atomic replacement prevents `watch` from reading a half-written file.
    temporary_path = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    temporary_path.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary_path, STATE_FILE)


env = make_skill_env(
    SKILL,
    render_mode="human",
    seed=0,
)

obs, info = env.reset(seed=0)

print(
    """
Keep this terminal focused for controls.

Controls:
  w/s : forward/back
  a/d : left/right
  r/f : up/down
  o/p : open/close gripper
  n   : reset
  q   : quit

Live facts are written to:
  live_facts.txt

Open another terminal and run:
  watch -n 0.1 -d cat live_facts.txt
"""
)

old_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())

step = 0
last_dashboard_update = 0.0

try:
    zero_action = np.zeros(4, dtype=np.float32)

    write_state_dashboard(
        step=step,
        action=zero_action,
        reward=0.0,
        terminated=False,
        truncated=False,
        info=info,
    )

    while True:
        key = get_key()
        action = np.zeros(4, dtype=np.float32)

        if key == "q":
            break
        elif key == "n":
            obs, info = env.reset()
            step = 0

            write_state_dashboard(
                step=step,
                action=action,
                reward=0.0,
                terminated=False,
                truncated=False,
                info=info,
            )
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

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        now = time.monotonic()
        if now - last_dashboard_update >= STATE_UPDATE_PERIOD:
            write_state_dashboard(
                step=step,
                action=action,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            last_dashboard_update = now

        if terminated or truncated:
            obs, info = env.reset()
            step = 0

        time.sleep(1 / 25)

finally:
    termios.tcsetattr(
        sys.stdin,
        termios.TCSADRAIN,
        old_settings,
    )
    env.close()
