import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
import time

def main():
    # 1. Ortamı başlatıyoruz — use_lidar eğitimdekiyle BİREBİR aynı olmalı!
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    
    # 2. Modelini yükle — karşılaştırmada en tutarlı çıkan model: PPO (5M adım)
    model = PPO.load("flappy_bird_ppo_model")
    print("Model başarıyla yüklendi! Kuşun gerçek performansını izliyoruz...")
    
    obs, info = env.reset()
    
    while True:
        # Yapay zeka en kararlı hamlesini seçiyor
        action, _states = model.predict(obs, deterministic=True)
        
        # Oyunda bir adım atıyoruz
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Gecikmeyi biraz azaltıyoruz ki fizik motoru stabil kalsın
        time.sleep(1 / 45) 
        
        # Kuş boruya veya yere çarparsa baştan başlıyoruz.
        # Anında reset atınca çarpışma karesi görünmüyordu; 1.5 sn bekleyip öyle başlıyoruz.
        if terminated:
            print(f"Kuş elendi (skor: {info.get('score', '?')}), yeni tura geçiliyor.")
            time.sleep(1.5)
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()