# test_models.py

import torch
import numpy as np

# Import các lớp mạng cần kiểm tra
from models.networks import ResNet18FeatureExtractor, ILModel, DQN

# Import các biến cấu hình để xác định kích thước
from config import N_CLASSES, ACTION_HISTORY_SIZE, NUMBER_ACTIONS, FEATURE_DIM

def test_all_networks():
    """
    Hàm này kiểm tra tất cả các kiến trúc mạng trong models/networks.py
    """
    print("=============================================")
    print("--- STEP 3: TESTING models/networks.py ---")
    print("=============================================")
    
    # Thiết lập thiết bị (GPU nếu có)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    # --- Test 1: ResNet18FeatureExtractor ---
    print("\n-----------------------------------------")
    print("--- 1. Testing ResNet18FeatureExtractor ---")
    print("-----------------------------------------")
    try:
        # INPUT: Một batch giả gồm 2 ảnh, mỗi ảnh có 3 kênh màu, kích thước 448x448
        batch_size = 2
        dummy_image_batch = torch.randn(batch_size, 3, 448, 448).to(device)
        
        feature_extractor = ResNet18FeatureExtractor().to(device)
        feature_extractor.eval() # Chuyển sang chế độ đánh giá

        with torch.no_grad(): # Không cần tính gradient khi test
            features = feature_extractor(dummy_image_batch)
        
        # OUTPUT: Kiểm tra shape của vector đặc trưng
        print(f"    - Input shape: {dummy_image_batch.shape}")
        print(f"    - Output feature vector shape: {features.shape}")
        print(f"    - Expected output shape: torch.Size([{batch_size}, {FEATURE_DIM}])")
        assert features.shape == (batch_size, FEATURE_DIM)
        print("[SUCCESS] ResNet18FeatureExtractor passed.")
    except Exception as e:
        print(f"[FAILED] ResNet18FeatureExtractor failed. Error: {e}")
        return

    # --- Test 2: ILModel và DQN (kiến trúc tương tự) ---
    print("\n-----------------------------------------")
    print("--- 2. Testing ILModel and DQN Networks ---")
    print("-----------------------------------------")
    
    # Chuẩn bị INPUT: một state giả
    # Kích thước state = feature_dim + center_coords + history_dim
    # Kích thước history cho Center Agent = 7 * (6 actions + 20 classes)
    center_history_dim = ACTION_HISTORY_SIZE * (NUMBER_ACTIONS + N_CLASSES)
    center_input_dim = FEATURE_DIM + 2 + center_history_dim

    # Kích thước history cho Size Agent = 7 * 6 actions
    size_history_dim = ACTION_HISTORY_SIZE * NUMBER_ACTIONS
    size_input_dim = FEATURE_DIM + 2 + size_history_dim
    
    batch_size = 4
    dummy_center_state = torch.randn(batch_size, center_input_dim).to(device)
    dummy_size_state = torch.randn(batch_size, size_input_dim).to(device)

    # --- Test 2a: ILModel for Center Agent ---
    print("\n[INFO] Testing ILModel (Center)...")
    try:
        il_center_net = ILModel(input_dim=center_input_dim, phase="center", n_classes=N_CLASSES).to(device)
        il_center_net.eval()
        with torch.no_grad():
            pos, cls, conf, done = il_center_net(dummy_center_state)
        
        print(f"    - IL Center Output Shapes:")
        print(f"        - Position head: {pos.shape} (Expected: [{batch_size}, 4])")
        print(f"        - Class head:    {cls.shape} (Expected: [{batch_size}, {N_CLASSES}])")
        print(f"        - Conf head:     {conf.shape} (Expected: [{batch_size}, 1])")
        print(f"        - Done head:     {done.shape} (Expected: [{batch_size}, 1])")
        
        assert pos.shape == (batch_size, 4)
        assert cls.shape == (batch_size, N_CLASSES)
        assert conf.shape == (batch_size, 1)
        assert done.shape == (batch_size, 1)
        print("[SUCCESS] ILModel (Center) passed.")
    except Exception as e:
        print(f"[FAILED] ILModel (Center) failed. Error: {e}")

    # --- Test 2b: DQN for Size Agent ---
    print("\n[INFO] Testing DQN (Size)...")
    try:
        # n_outputs cho Size Agent chỉ là các hành động (không có class)
        dqn_size_net = DQN(input_dim=size_input_dim, n_outputs=NUMBER_ACTIONS, phase="size").to(device)
        dqn_size_net.eval()
        with torch.no_grad():
            size_q, conf_q = dqn_size_net(dummy_size_state)
            
        print(f"    - DQN Size Output Shapes:")
        print(f"        - Size Q-values head: {size_q.shape} (Expected: [{batch_size}, 4])")
        print(f"        - Conf Q-values head: {conf_q.shape} (Expected: [{batch_size}, 1])")
        
        assert size_q.shape == (batch_size, 4) # 4 hành động thay đổi kích thước
        assert conf_q.shape == (batch_size, 1)
        print("[SUCCESS] DQN (Size) passed.")
    except Exception as e:
        print(f"[FAILED] DQN (Size) failed. Error: {e}")

    print("\n=============================================")
    print("--- models/networks.py: ALL TESTS PASSED ---")
    print("=============================================")

# Chạy hàm test khi file này được thực thi
if __name__ == "__main__":
    test_all_networks()