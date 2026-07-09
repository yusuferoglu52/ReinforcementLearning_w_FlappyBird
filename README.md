# Flappy AI — Teaching Flappy Bird with Reinforcement Learning

This project trains an AI agent to play Flappy Bird in the
[flappy_bird_gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) environment
using **Stable-Baselines3**. It is also a debugging and model-improvement story:
a step-by-step account of how an agent that scored 0 became one that clears
hundreds of pipes.

## Files

| File | Purpose |
|---|---|
| `train.py` | DQN training (5M steps + best-model checkpointing) |
| `train_ppo.py` | PPO training (5M steps, 8 parallel environments) |
| `enjoy.py` | Runs the trained model with rendering so you can watch it play |
| `flappy_bird_ppo_model.zip` | **The winning model** — loaded by enjoy.py |
| `flappy_bird_dqn_model.zip` | Final model from DQN training |
| `models/dqn_best/`, `models/ppo_best/` | Best-moment checkpoints saved by EvalCallback |
| `tensorboard_logs/` | Training curves |

## Setup and Usage

```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Watch the trained model play
python enjoy.py

# Retrain from scratch (headless, fast mode)
python train.py      # DQN, ~1 hour
python train_ppo.py  # PPO, ~30 minutes

# Monitor training curves
tensorboard --logdir ./tensorboard_logs/
```

## The Story: "The game keeps restarting even though the bird didn't die!"

When the first model (100k-step DQN) was run, the game seemed to restart
constantly — and the bird never visibly hit anything. Systematic debugging revealed:

1. In the environment's source code, `terminated` fires **only on a real collision**
   (pipe or ground); there is no hidden timeout or phantom reset.
2. Running the model headless and logging every death exposed the truth:
   the bird **really was dying — crashing into the first pipe after ~56 steps
   (about 1.5 seconds) in every single episode**.
3. It *looked* like it never died because `enjoy.py` **reset instantly on death**:
   at 45 fps the collision frame stayed on screen for ~22 ms — too fast for the eye.

> **Lesson:** When an RL agent "behaves strangely," look at the data before the render.
> Running headless and logging death events diagnosed the problem in one pass.

## Model Improvement, Step by Step

### Step 1 — The right observation space (`use_lidar=False`)

By default, `FlappyBird-v0` returns a **180-dimensional lidar** observation. Learning
from that with a small MLP is very hard for DQN. With `use_lidar=False` the environment
returns 12 simple features (pipe positions, the bird's position and velocity) — and
learning becomes dramatically easier.

**Critical rule:** environment parameters must be **identical** between training and
playback. A model trained on a different observation format either crashes or plays
nonsense.

### Step 2 — Balancing exploration and exploitation

The first attempt used `exploration_fraction=0.5`: half of the 100k-step run was spent
on random actions, leaving the model almost no time to learn an actual policy.
Lowering it to 5–10% ends exploration early and spends the remaining steps on learning.

### Step 3 — Enough training time

| Training run | Result (10 deterministic episodes) |
|---|---|
| 100k steps, DQN | score 0–1, dies at the first pipe |
| 500k steps, DQN | score exactly 1, dies at the second pipe |
| 2M steps, DQN | median 2, best 33 — inconsistent |
| 5M steps, DQN | median 41, mean 72 |
| **5M steps, PPO** | **median 200 (the cap), worst episode 116** |

> **Lesson:** In RL, "it isn't learning" usually means "it wasn't trained long enough."
> Flappy Bird looks simple, yet consistent DQN play takes millions of steps.

### Step 4 — Keeping the best model (EvalCallback)

The model at the end of training is not necessarily the best one; performance
fluctuates. `EvalCallback` plays 10 deterministic episodes every 25k steps and saves
the best-so-far model to `models/.../best_model.zip`. The evaluation environment uses
`score_limit=50` — otherwise, as the model improves, evaluation episodes would run
forever.

### Step 5 — Choosing the algorithm: DQN vs PPO

Both algorithms competed under identical conditions (5M steps, same observations,
same `[128, 128]` network architecture):

- **DQN** (value-based, learns from a replay buffer): has good episodes but is
  inconsistent — sometimes 200, sometimes 2.
- **PPO** (policy-gradient, learns from 8 parallel environments): trained faster
  (~30 min vs ~60 min) and played far more consistently, reaching the 200-pipe cap
  in 8 out of 10 episodes.

That is why `enjoy.py` loads the PPO model.

## Details worth noting in enjoy.py

- `deterministic=True`: the model always picks its most confident action (no randomness).
- On death, the score is printed and the game **pauses for 1.5 seconds** — so you can
  actually see the collision.
- `time.sleep(1/45)` slows the game down to a speed the human eye can follow.

## Tech Stack

- Python + [Gymnasium](https://gymnasium.farama.org/) (the standard RL environment API)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (DQN and PPO implementations)
- [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) (the game environment)
- TensorBoard (training-metric visualization)
