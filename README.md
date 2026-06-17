# Fetch Blockworld Skills

Minimal symbolic-fact and skill-training scaffolding for Gymnasium-Robotics Fetch v4.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Linux rendering issues, try:

```bash
sudo apt-get install -y libgl1-mesa-dev libglfw3 libglfw3-dev patchelf
MUJOCO_GL=glfw python inspect_facts.py --skill pickup --render
```

## Inspect facts

```bash
python inspect_facts.py --skill pickup
MUJOCO_GL=glfw python inspect_facts.py --skill pickup --render
```

## Train a skill

```bash
python train_skill.py --skill pickup --algo sac --timesteps 100000
python train_skill.py --skill pushleft --algo sac --timesteps 100000
```

## Roll out a trained skill

```bash
MUJOCO_GL=glfw python rollout_skill.py --skill pickup --algo sac --model models/pickup_sac.zip
```
