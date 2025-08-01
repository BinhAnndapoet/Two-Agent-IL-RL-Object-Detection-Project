# test_training.py

import torch
import os

# Import các thành phần cần thiết
from config import env_config, NUM_IL_EPOCHS
from data.dataset import load_pascal_voc
from environment.env import DetectionEnv
from training.il_trainer import initialize_replay_buffer # Hàm chính cần test

def test_imitation_learning_phase():
    """
    Hàm này kiểm tra toàn bộ luồng của giai đoạn Imitation Learning.
    """
    print("=============================================")
    print("--- STEP 5: TESTING training/ (IL Phase) ---")
    print("=============================================")

    # --- Phần 1: Chuẩn bị Môi trường và Cấu hình ---
    print("\n[INFO] 1. Preparing environment for IL testing...")
    try:
        root_dir = "./VOC2012"
        datasets, class_names = load_pascal_voc(root_dir)
        
        # Cập nhật env_config
        env_config['dataset'] = datasets
        env_config['current_class'] = class_names
        env_config['n_classes'] = len(class_names)
        env_config['phase'] = 'il' # Đặt pha là 'il' để env dùng đúng tập dữ liệu
        
        # Khởi tạo môi trường
        env = DetectionEnv(env_config)
        
        print("[SUCCESS] Environment prepared.")
    except Exception as e:
        print(f"[FAILED] Could not prepare environment. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Phần 2: Chạy hàm điều phối IL ---
    print("\n[INFO] 2. Running the main IL function 'initialize_replay_buffer'...")
    
    # Giảm số lượng để test nhanh hơn
    num_test_trajectories = 5
    # Tạm thời ghi đè số epochs để test nhanh
    original_epochs = NUM_IL_EPOCHS
    env_config['epochs'] = 3 # Chỉ huấn luyện 3 epochs để kiểm tra
    
    print(f"    - Generating {num_test_trajectories} expert trajectories...")
    print(f"    - Training IL models for {env_config['epochs']} epochs...")
    
    try:
        # INPUT: Môi trường đã khởi tạo và số lượng quỹ đạo mong muốn
        # Đây là hàm chính chúng ta cần kiểm tra
        center_buffer, size_buffer, center_il_model, size_il_model = initialize_replay_buffer(
            env, num_trajectories=num_test_trajectories
        )
        
        # OUTPUT: Kiểm tra các giá trị trả về
        print("\n[SUCCESS] 'initialize_replay_buffer' completed without errors.")
        
        # Kiểm tra Center Agent's buffer và model
        print("\n--- Verifying Center Agent outputs ---")
        print(f"    - Center Replay Buffer size: {len(center_buffer)}")
        assert len(center_buffer) > 0, "Center buffer should not be empty."
        assert center_il_model is not None, "Center IL model should be returned."
        assert isinstance(center_il_model, torch.nn.Module), "Center model should be a torch Module."
        print(f"    - Center IL Model: Created successfully.")

        # Kiểm tra Size Agent's buffer và model
        print("\n--- Verifying Size Agent outputs ---")
        print(f"    - Size Replay Buffer size: {len(size_buffer)}")
        # Lưu ý: Size buffer có thể rỗng nếu không có quỹ đạo nào chuyển sang pha 'size' thành công
        if len(size_buffer) == 0:
            print("    - WARNING: Size buffer is empty. This can happen if no trajectories successfully transitioned to 'size' phase in this short test.")
            assert size_il_model is None, "Size model should be None if no trajectories were generated."
        else:
            assert size_il_model is not None, "Size IL model should be returned."
            assert isinstance(size_il_model, torch.nn.Module), "Size model should be a torch Module."
            print(f"    - Size IL Model: Created successfully.")

    except Exception as e:
        print(f"[FAILED] An error occurred during the IL phase. Error: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        # Khôi phục lại giá trị epochs ban đầu
        env_config['epochs'] = original_epochs

    print("\n=============================================")
    print("--- training/: TEST PASSED ---")
    print("=============================================")

# Chạy hàm test khi file này được thực thi
if __name__ == "__main__":
    test_imitation_learning_phase()