import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

N_ENVS = 8  # PPO deneyimi paralel toplar — 8 ortam eğitimi ciddi hızlandırır

def main():
    print("Flappy Bird PPO Eğitimi Başlatılıyor...")

    # 8 paralel ortam; gözlem formatı DQN'dekiyle aynı (use_lidar=False)
    env = make_vec_env(
        "FlappyBird-v0",
        n_envs=N_ENVS,
        env_kwargs=dict(render_mode=None, use_lidar=False),
    )

    # Değerlendirme ortamı: score_limit=50 ile bölümler sınırlı
    eval_env = Monitor(
        gym.make("FlappyBird-v0", render_mode=None, use_lidar=False, score_limit=50)
    )

    # eval_freq vektörlü ortamda her env.step() çağrısında sayılıyor;
    # 25000 // N_ENVS ile toplamda ~25k adımda bir değerlendirme yapılıyor.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/ppo_best/",
        eval_freq=25000 // N_ENVS,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,        # Her güncellemede ortam başına 1024 adım deneyim
        batch_size=256,
        gamma=0.99,
        ent_coef=0.01,       # Küçük entropi bonusu — erken ezberlemeyi önler
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log="./tensorboard_logs/",
    )

    print("PPO eğitimi başladı! (5M adım, 8 paralel ortam)")
    model.learn(total_timesteps=5000000, callback=eval_callback)

    model.save("flappy_bird_ppo_model")
    print("PPO eğitimi tamamlandı. En iyi model: models/ppo_best/best_model.zip")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
