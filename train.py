import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

def main():
    print("Flappy Bird Ortamı Arka Planda (Hızlı Mod) Başlatılıyor...")

    # HIZLANDIRMA SIRRI: render_mode=None yaparak ekran çizimini kapatıyoruz.
    # use_lidar=False: 180 boyutlu lidar yerine 12 basit özellik — DQN bununla çok daha kolay öğreniyor.
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)

    # Değerlendirme ortamı: score_limit=50 ile bölümler sınırlanıyor,
    # yoksa model iyileştikçe değerlendirme bölümleri sonsuza kadar sürebilir.
    eval_env = Monitor(
        gym.make("FlappyBird-v0", render_mode=None, use_lidar=False, score_limit=50)
    )

    # Her 25k adımda 10 deterministik bölüm oynatıp o ana kadarki EN İYİ modeli kaydediyor.
    # Böylece eğitim sonunda kötüleşse bile en iyi anındaki model elimizde kalıyor.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/dqn_best/",
        eval_freq=25000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    print("Yapay Zeka Modeli (DQN) TensorBoard Entegrasyonu ile Kuruluyor...")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,           # Uzun eğitimde kararlılık için düşürdük
        buffer_size=100000,           # Hafızayı genişlettik
        learning_starts=10000,        # İlk 10k adım sadece rastgele dünyayı keşfetsin
        batch_size=128,               # Daha büyük batch = daha kararlı Q güncellemeleri
        target_update_interval=1000,  # Hedef ağı daha seyrek güncellensin
        exploration_fraction=0.05,    # 5M adımda %5 = 250k adım keşif, kalanı öğrenme
        exploration_final_eps=0.01,   # En son aşamada bile %1 rastgelelik kalsın
        policy_kwargs=dict(net_arch=[128, 128]), # Sinir ağını biraz daha büyüterek kapasitesini artırdık
        tensorboard_log="./tensorboard_logs/"
    )

    # 2M adımda skor medyanı 2'de kaldı; 5M adım + en iyi model kaydıyla tutarlılığı hedefliyoruz.
    print("Eğitim başladı! Grafikleri izlemek için terminalde yeni sekme açıp TensorBoard'u başlatabilirsin.")
    model.learn(total_timesteps=5000000, callback=eval_callback)

    # Son modeli de kaydedelim (en iyi model zaten models/dqn_best/best_model.zip içinde)
    model.save("flappy_bird_dqn_model")
    print("Mükemmel! Eğitim tamamlandı. En iyi model: models/dqn_best/best_model.zip")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
