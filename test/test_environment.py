# test_environment.py

import numpy as np
import cv2
import os

# Import các thành phần cần thiết
from config import env_config
from data.dataset import load_pascal_voc
from environment.env import DetectionEnv

def test_environment_functionality():
    """
    Hàm này kiểm tra các chức năng cốt lõi của DetectionEnv.
    """
    print("=============================================")
    print("--- STEP 4: TESTING environment/ ---")
    print("=============================================")
    
    # --- Phần 1: Chuẩn bị Input cho Môi trường ---
    print("\n[INFO] 1. Preparing inputs for the environment (config and dataset)...")
    try:
        root_dir = "./VOC2012"
        datasets, class_names = load_pascal_voc(root_dir)
        
        # Cập nhật env_config với dataset đã tải
        env_config['dataset'] = datasets
        env_config['current_class'] = class_names
        env_config['n_classes'] = len(class_names)
        
        print("[SUCCESS] Inputs prepared.")
    except Exception as e:
        print(f"[FAILED] Could not prepare inputs. Error: {e}")
        return

    # --- Phần 2: Khởi tạo Môi trường ---
    print("\n[INFO] 2. Initializing the DetectionEnv...")
    try:
        env = DetectionEnv(env_config)
        print("[SUCCESS] Environment initialized.")
    except Exception as e:
        print(f"[FAILED] Environment initialization failed. Error: {e}")
        return

    # --- Phần 3: Kiểm tra hàm reset() ---
    print("\n[INFO] 3. Testing env.reset()...")
    try:
        # Gọi reset lần đầu
        state, info = env.reset()
        
        # Kiểm tra output của reset
        print(f"    - Reset successful.")
        print(f"    - Initial state shape: {state.shape} (Expected shape with {env.observation_space.shape[0]} features)")
        print(f"    - Initial info: IoU = {info.get('iou', 0):.3f}, Phase = {info.get('phase', 'N/A')}")
        
        assert isinstance(state, np.ndarray), "State should be a numpy array."
        assert state.shape[0] == env.observation_space.shape[0], "State shape mismatch with observation space."
        assert info['phase'] == 'center', "Initial phase should be 'center'."
        print("[SUCCESS] env.reset() works as expected.")
        
        # Trực quan hóa trạng thái ban đầu
        initial_frame = env.render()
        output_dir = "test_outputs"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "env_reset_state.jpg"), cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR))
        print(f"    - Saved initial state visualization to '{output_dir}/env_reset_state.jpg'")

    except Exception as e:
        print(f"[FAILED] env.reset() failed. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Phần 4: Kiểm tra hàm step() ---
    print("\n[INFO] 4. Testing env.step() for a few steps...")
    try:
        num_steps_to_test = 5
        print(f"    - Simulating {num_steps_to_test} steps in 'center' phase...")
        
        for i in range(num_steps_to_test):
            # Lấy một hành động ngẫu nhiên hợp lệ
            # action_space cho 'center' là một Tuple(Discrete, Discrete)
            action_sample = env.action_space.sample()
            pos_action, class_action = action_sample
            
            # Thực thi bước
            new_state, reward, done, truncated, info = env.step((pos_action, class_action))
            
            # In ra kết quả của bước
            print(f"    -> Step {i+1}: Action=({pos_action}, {class_action}), Reward={reward:.4f}, Phase={info['phase']}, IoU={info['iou']:.3f}, Done={done}")
            
            # Kiểm tra các giá trị trả về
            assert isinstance(new_state, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(truncated, bool)

        print("[SUCCESS] env.step() works without errors.")
    except Exception as e:
        print(f"[FAILED] env.step() failed. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=============================================")
    print("--- environment/: TEST PASSED ---")
    print("=============================================")

# Chạy hàm test khi file này được thực thi
if __name__ == "__main__":
    test_environment_functionality()