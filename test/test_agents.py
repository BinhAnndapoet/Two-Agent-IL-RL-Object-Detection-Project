# test_agents.py

import torch
import numpy as np

# Import các thành phần cần thiết
from config import env_config
from data.dataset import load_pascal_voc
from environment.env import DetectionEnv
from training.il_trainer import initialize_replay_buffer
from agents.center_agent import CenterDQNAgent
from agents.size_agent import SizeDQNAgent

def test_dqn_agents_functionality():
    """
    Hàm này kiểm tra các chức năng cốt lõi của các lớp Agent.
    """
    print("=============================================")
    print("--- STEP 6: TESTING agents/ ---")
    print("=============================================")

    # --- Phần 1: Chuẩn bị các thành phần phụ thuộc (Dependencies) ---
    print("\n[INFO] 1. Preparing dependencies (Env, IL models, Buffers)...")
    try:
        # Tải dữ liệu và cấu hình
        root_dir = "./VOC2012"
        datasets, class_names = load_pascal_voc(root_dir)
        env_config['dataset'] = datasets
        env_config['current_class'] = class_names
        
        # Khởi tạo môi trường cho IL
        env_il = DetectionEnv(env_config)
        
        # Chạy giai đoạn IL để lấy model và buffer đã được điền dữ liệu
        # Sử dụng số lượng nhỏ để test nhanh
        center_buffer, size_buffer, center_il_model, _ = initialize_replay_buffer(
            env_il, num_trajectories=2
        )
        
        # Tạo môi trường mới cho pha DQN
        env_config['phase'] = 'dqn'
        env_dqn = DetectionEnv(env_config)
        
        # Đảm bảo buffer của center không rỗng để có thể test
        assert len(center_buffer) > 0, "Center buffer from IL phase is empty, cannot proceed."
        
        print("[SUCCESS] Dependencies prepared.")
    except Exception as e:
        print(f"[FAILED] Could not prepare dependencies. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Phần 2: Kiểm tra CenterDQNAgent ---
    print("\n-----------------------------------------")
    print("--- 2. Testing CenterDQNAgent ---")
    print("-----------------------------------------")
    
    try:
        # 2a. Khởi tạo Agent
        print("[INFO] Initializing CenterDQNAgent...")
        center_agent = CenterDQNAgent(env=env_dqn, replay_buffer=center_buffer)
        print("[SUCCESS] Agent initialized.")

        # 2b. Tải trọng số IL
        print("[INFO] Loading pre-trained IL weights...")
        if center_il_model:
            center_agent.policy_net.load_state_dict(center_il_model.state_dict())
            center_agent.target_net.load_state_dict(center_il_model.state_dict())
            print("[SUCCESS] IL weights loaded.")
        else:
            print("[WARNING] No IL model to load.")

        # 2c. Chọn hành động
        print("[INFO] Testing select_action()...")
        state, _ = env_dqn.reset() # Lấy một state ban đầu
        action = center_agent.select_action(state)
        print(f"    - Agent selected action: {action}")
        # Hành động trả về phải là một tuple (pos_action, class_action)
        assert isinstance(action, tuple) and len(action) == 2, "Action format is incorrect."
        print("[SUCCESS] select_action() works.")

        # 2d. Cập nhật mạng
        print("[INFO] Testing update()...")
        # Giảm batch_size của buffer để test với ít dữ liệu
        center_agent.replay_buffer.batch_size = 2 
        center_agent.update() # Thực hiện một bước cập nhật
        print("[SUCCESS] update() ran without errors.")
        
    except Exception as e:
        print(f"[FAILED] CenterDQNAgent test failed. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- CenterDQNAgent TEST PASSED ---")

    # --- Phần 3: Kiểm tra SizeDQNAgent (trường hợp buffer rỗng) ---
    # Trong các bài test ngắn, size_buffer thường rỗng. Chúng ta cần kiểm tra xem
    # agent có xử lý được trường hợp này một cách an toàn không.
    print("\n-----------------------------------------")
    print("--- 3. Testing SizeDQNAgent ---")
    print("-----------------------------------------")

    try:
        # 3a. Khởi tạo Agent
        print("[INFO] Initializing SizeDQNAgent...")
        # Sử dụng size_buffer (có thể rỗng)
        size_agent = SizeDQNAgent(env=env_dqn, replay_buffer=size_buffer)
        print("[SUCCESS] Agent initialized.")

        # 3b. Cập nhật mạng (khi buffer rỗng)
        print("[INFO] Testing update() with an empty buffer...")
        size_agent.update() # Hàm update nên có kiểm tra và return nếu buffer không đủ lớn
        print("[SUCCESS] update() handled empty buffer gracefully.")

    except Exception as e:
        print(f"[FAILED] SizeDQNAgent test failed. Error: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("\n--- SizeDQNAgent TEST PASSED ---")
    
    print("\n=============================================")
    print("--- agents/: ALL TESTS PASSED ---")
    print("=============================================")

# Chạy hàm test khi file này được thực thi
if __name__ == "__main__":
    test_dqn_agents_functionality()