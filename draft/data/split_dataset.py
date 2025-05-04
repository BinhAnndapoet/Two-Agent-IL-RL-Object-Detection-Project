import os
import shutil 
import random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thiếu lập thư mục
DATASET_DIR = os.path.join(BASE_DIR, "VOCdevkit", "VOC2012")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_split")

JPEG_DIR = os.path.join(DATASET_DIR, "JPEGImages")
ANNO_DIR = os.path.join(DATASET_DIR, "Annotations")


# Tỉ lệ chia
IL_ratio = 0.2
RL_ratio = 0.8




# Copy ảnh qua
def copy_files(name_list, split_name):
    img_out = os.path.join(OUTPUT_DIR, split_name, "JPEGImages")
    anno_out = os.path.join(OUTPUT_DIR, split_name, "Annotations")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(anno_out, exist_ok=True)
    
    # duyệt qua từng ảnh
    for name in tqdm(name_list, desc=f"Copying {split_name}"):
        img_src = os.path.join(JPEG_DIR, name + '.jpg')
        anno_src = os.path.join(ANNO_DIR, name + '.xml')
        
        # xác định nơi copy
        img_dst = os.path.join(img_out, name + '.jpg')
        anno_dst = os.path.join(anno_out, name + '.xml')
        
        # Kiểm tra ảnh và anno có tổn tại hay không
        if os.path.exists(img_src) and os.path.exists(anno_src):
            shutil.copy2(img_src, img_dst)
            shutil.copy2(anno_src, anno_dst)
        else:
            print(f"Missing file: {name}")
            
            
if __name__ == '__main__':
    # Lấy danh sách các file
    image_names = []

    for f in os.listdir(JPEG_DIR):
        if f.endswith('.jpg'):
            image_names.append(f[:-4]) # chỉ lấy tên file

    random.shuffle(image_names)

    total = len(image_names)
    il_end = int(total * IL_ratio)

    # Chia danh sách ảnh thành 2 phần
    il_list = image_names[:il_end]
    rl_list = image_names[il_end:]

    print(f"Total images: {total} | IL: {il_end} | RL: {total - il_end}")
    
    
    copy_files(il_list, "IL")
    copy_files(rl_list, "RL")

    print("✅ Completed copy to folders IL and RL.")
