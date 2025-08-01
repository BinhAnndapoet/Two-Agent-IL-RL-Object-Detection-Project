# test_dataset.py (Updated to generate 5 sample images)

import torch
import cv2
import numpy as np
import os
from data.dataset import load_pascal_voc # Import hàm chính từ module dataset

def visualize_and_save_sample(img_tensor, bboxes, labels, output_path):
    """
    Hàm trợ giúp để trực quan hóa một mẫu và lưu ra file ảnh.
    """
    # 1. Đảo ngược quá trình chuẩn hóa (Normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Đưa tensor về CPU, chuyển thành numpy, và đổi trục từ [C, H, W] thành [H, W, C]
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Áp dụng công thức đảo ngược: img = (img * std) + mean
    img_np = (img_np * std + mean) * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8) # Thêm clip để đảm bảo giá trị trong khoảng [0, 255]

    # OpenCV sử dụng hệ màu BGR, trong khi ảnh gốc là RGB, nên cần chuyển đổi
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2. Vẽ các bounding box và nhãn lên ảnh
    if not bboxes:
        print(f"    - WARNING: No bounding boxes found for this sample.")
    else:
        for bbox, label in zip(bboxes, labels):
            # Tọa độ bounding box phải là số nguyên để vẽ
            xmin, ymin, xmax, ymax = map(int, bbox)
            # Vẽ hình chữ nhật
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Màu xanh, độ dày 2
            # Viết tên nhãn
            cv2.putText(img_bgr, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # 3. Lưu ảnh kết quả ra file
    cv2.imwrite(output_path, img_bgr)


def test_data_loading_and_processing():
    """
    Hàm này kiểm tra toàn bộ chức năng của data/dataset.py và xuất ra 5 ảnh mẫu.
    """
    print("=============================================")
    print("--- STEP 1: TESTING data/dataset.py ---")
    print("=============================================")

    # --- Phần 1: Tải và chia dữ liệu ---
    print("\n[INFO] 1. Attempting to load and split the dataset...")
    
    root_dir = "./VOC2012"
    if not os.path.exists(root_dir):
        print(f"[ERROR] Dataset directory not found at: {root_dir}")
        return

    try:
        datasets, class_names = load_pascal_voc(root_dir, il_ratio=0.2)
        print("\n[SUCCESS] Dataset loaded and split successfully.")
        print(f"    - Train IL set size: {len(datasets['train_il'][0])} samples.")
        print(f"    - Train DQN set size: {len(datasets['train_dqn'][0])} samples.")
        print(f"    - Test set size: {len(datasets['test'][0])} samples.")
    except Exception as e:
        print(f"[FAILED] Could not load the dataset. Error: {e}")
        return

    # --- Phần 2: Kiểm tra và trực quan hóa 5 mẫu ---
    print("\n[INFO] 2. Visualizing the first 5 samples from the test set...")

    # Tạo thư mục output nếu chưa tồn tại
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"    - Output images will be saved to '{output_dir}/'")

    try:
        test_loader = datasets['test'][1]
        data_iterator = iter(test_loader)
        num_samples_to_check = 5

        for i in range(num_samples_to_check):
            print(f"\n--- Processing Sample {i+1}/{num_samples_to_check} ---")
            
            # Lấy batch tiếp theo
            img_ids, imgs_tensor, bboxes_list, labels_list = next(data_iterator)
            
            # Trích xuất thông tin của mẫu đầu tiên trong batch
            img_id = img_ids[0]
            img_tensor = imgs_tensor[0]
            bboxes = bboxes_list[0]
            labels = labels_list[0]

            print(f"    - Image ID: {img_id}")
            print(f"    - Image Tensor Shape: {img_tensor.shape}")
            print(f"    - Found {len(bboxes)} bounding boxes.")
            
            # Tạo đường dẫn file output
            output_filename = f"sample_{i+1}_{img_id}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Gọi hàm trực quan hóa
            visualize_and_save_sample(img_tensor, bboxes, labels, output_path)
            print(f"    - Saved visualization to: '{output_path}'")

        print("\n=======================================================")
        print(f"--- data/dataset.py: TEST PASSED ---")
        print(f"Please check the {num_samples_to_check} images in the '{output_dir}' folder.")
        print("=======================================================")

    except StopIteration:
        print("[WARNING] The test set has fewer than 5 samples. The test completed with the available samples.")
    except Exception as e:
        print(f"[FAILED] An error occurred during sample processing. Error: {e}")
        import traceback
        traceback.print_exc()

# Chạy hàm test khi file này được thực thi
if __name__ == "__main__":
    test_data_loading_and_processing()