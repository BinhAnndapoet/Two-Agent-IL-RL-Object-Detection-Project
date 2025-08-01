# test_utils.py

import torch
import numpy as np

# Import các thành phần cần kiểm tra
from utils.metrics import calculate_iou, calculate_map
from utils.replay_buffer import ReplayBuffer, Transition

def test_metrics_module():
    """
    Hàm này kiểm tra các chức năng trong utils/metrics.py
    """
    print("\n-----------------------------------------")
    print("--- Testing utils/metrics.py ---")
    print("-----------------------------------------")

    # --- Test 1: calculate_iou ---
    print("\n[INFO] 1. Testing 'calculate_iou' function...")
    
    # Trường hợp 1: Có giao nhau tốt
    box_gt = [100, 100, 200, 200]        # Ground Truth Box (diện tích = 100*100 = 10000)
    box_pred_good = [110, 110, 210, 210] # Predicted Box (diện tích = 10000)
    # Intersection: [110, 110, 200, 200] -> diện tích = 90*90 = 8100
    # Union: 10000 + 10000 - 8100 = 11900
    # Expected IoU = 8100 / 11900 = 0.6806...
    iou_good = calculate_iou(box_gt, box_pred_good)
    print(f"    - IoU (good overlap): {iou_good:.4f} (Expected ~0.6807)")
    assert np.isclose(iou_good, 0.68067, atol=1e-4), "Good IoU calculation failed!"

    # Trường hợp 2: Không giao nhau
    box_pred_bad = [300, 300, 400, 400]
    iou_bad = calculate_iou(box_gt, box_pred_bad)
    print(f"    - IoU (no overlap): {iou_bad:.4f} (Expected 0.0)")
    assert iou_bad == 0.0, "Bad IoU calculation failed!"

    # Trường hợp 3: Trùng khít hoàn toàn
    iou_perfect = calculate_iou(box_gt, box_gt)
    print(f"    - IoU (perfect overlap): {iou_perfect:.4f} (Expected 1.0)")
    assert iou_perfect == 1.0, "Perfect IoU calculation failed!"
    print("[SUCCESS] 'calculate_iou' passed.")

    # --- Test 2: calculate_map (kiểm tra đơn giản) ---
    print("\n[INFO] 2. Testing 'calculate_map' function (simple case)...")
    gt_boxes = [[100, 100, 200, 200]]
    gt_labels = ['cat']
    pred_boxes = [[110, 110, 210, 210], [50, 50, 80, 80]] # Một box đúng, một box sai
    pred_labels = ['cat', 'cat']
    pred_scores = [0.9, 0.8] # Box đúng có score cao hơn
    
    # Với IoU threshold = 0.5, box đầu tiên là True Positive, box thứ hai là False Positive.
    # Precision = [1/1, 1/2] -> [1.0, 0.5]
    # Recall = [1/1, 1/1] -> [1.0, 1.0]
    # AP sẽ là 1.0
    mAP = calculate_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    print(f"    - Simple mAP: {mAP:.4f} (Expected 1.0)")
    assert np.isclose(mAP, 1.0), "mAP calculation failed!"
    print("[SUCCESS] 'calculate_map' passed basic check.")
    print("--- utils/metrics.py: TEST PASSED ---")


def test_replay_buffer_module():
    """
    Hàm này kiểm tra các chức năng trong utils/replay_buffer.py
    """
    print("\n-----------------------------------------")
    print("--- Testing utils/replay_buffer.py ---")
    print("-----------------------------------------")

    # --- Test 1: Khởi tạo và thêm dữ liệu ---
    print("\n[INFO] 1. Testing buffer initialization and appending...")
    # INPUT: Khởi tạo buffer với capacity=100, batch_size=10
    buffer = ReplayBuffer(capacity=100, batch_size=10)
    
    # Tạo 20 transition giả để thêm vào buffer
    # State giả có kích thước 512, giống như feature vector từ ResNet
    dummy_state_dim = 512
    for i in range(20):
        state = np.random.rand(dummy_state_dim)
        next_state = np.random.rand(dummy_state_dim)
        # Transition chứa các numpy array và số, không phải tensor
        transition = Transition(
            state=state, 
            action=i % 4,  # Hành động di chuyển 0, 1, 2, 3
            reward=np.random.rand(), 
            done=(i == 19), # Transition cuối cùng là done
            next_state=next_state
        )
        buffer.append(transition)
    
    print(f"    - Current buffer size: {len(buffer)} (Expected 20)")
    assert len(buffer) == 20, "Buffer append failed!"
    print("[SUCCESS] Buffer append works correctly.")

    # --- Test 2: Lấy mẫu một batch ---
    print("\n[INFO] 2. Testing batch sampling...")
    # INPUT: buffer đã chứa dữ liệu
    states, actions, rewards, dones, next_states = buffer.sample_batch()
    
    batch_size = 10
    print(f"    - Sampled batch shapes:")
    print(f"        - States: {states.shape} (Expected torch.Size([{batch_size}, 1, {dummy_state_dim}]))")
    print(f"        - Actions: {actions.shape} (Expected torch.Size([{batch_size}, 1]))")
    print(f"        - Rewards: {rewards.shape} (Expected torch.Size([{batch_size}, 1]))")
    print(f"        - Dones: {dones.shape} (Expected torch.Size([{batch_size}, 1]))")
    print(f"        - Next States: {next_states.shape} (Expected torch.Size([{batch_size}, 1, {dummy_state_dim}]))")
    
    # Kiểm tra shape
    assert states.shape == (batch_size, 1, dummy_state_dim)
    assert actions.shape == (batch_size, 1)
    assert rewards.shape == (batch_size, 1)
    
    # Kiểm tra kiểu dữ liệu
    assert isinstance(states, torch.Tensor)
    assert actions.dtype == torch.int64
    assert rewards.dtype == torch.float32
    print("[SUCCESS] Batch sampling works correctly.")
    print("--- utils/replay_buffer.py: TEST PASSED ---")

# Chạy tất cả các hàm test
if __name__ == "__main__":
    print("=============================================")
    print("--- STEP 2: TESTING utils/ ---")
    print("=============================================")
    test_metrics_module()
    test_replay_buffer_module()
    print("\n=============================================")
    print("--- All utils tests completed. ---")
    print("=============================================")