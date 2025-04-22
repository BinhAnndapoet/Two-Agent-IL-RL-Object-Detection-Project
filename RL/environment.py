import gym
from gym import spaces
import gym.spaces
import numpy as np
import cv2 as cv
class CusomEnv(gym.Env):
    def __init__(self, grid_size=5, img_shape=(224, 224, 3)):
        """  
            - Khởi tạo môi trường cho bài toán object detection với 2 agent.
                + agent 1: điều chỉnh tâm bounding box
                + agent 2: điều chỉnh kích thước bounding box
            - paramters:
                + grid_size(int): kích thước lưới
                + img_shape(tuple): kích thước ảnh đầu vào
        """
        
        super().__init__()
        self.grid_size = grid_size
        self.image_shape = img_shape
        
        # không gian quan sát: ảnh RGB (0, 255)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=img_shape, dtype=np.unit8)

        # action space cho agent 1 (center): nhãn các lớp (rời rạc) - tọa độ (liên tục)
        self.action_space_agent1 = gym.spaces.Discrete(20) # nhãn lớp trong VOC - 20 
        self.action_space_center = gym.spaces.Box(low=-10, high=10, shape=(2, ), dtype=np.float32) #  Δx, Δy
        
        # action space cho agent 2 (size box) - điều chỉnh kích thước bounding box
        self.action_space_agent2 = gym.spaces.Box(low=-20, high=20, shape=(2, ), dtype=np.float32) # Δw, Δh
        
        # khỏi tạo lại môi trường
        self.reset()
    
    
    def reset(self):
        """ 
            - 
        """
        pass
    
    
    def step(self, action1, action2):
        """ 
            - Cập nhật vị trí và kích thước của bounding box dựa trên hành động của Agent 1 và Agent 2
            - paramters: 
                + action1: hành động  agent1 -> điều chỉnh vị trí tâm
                + action2: hành động agent2 -> điều chỉnh kích thước bounding box
                
            - returns:
                + image(np.array): ảnh sau khi được thực hiện
                + reward(float): Phần thưởng tính toán dựa trên IoU giữa bounding box dự đoán và ground truth.
                + done(bool): Trạng thái kết thúc của tác vụ (true nếu IoU vượt quá 0.5).
                + info(dict): 
        """
        pass
    
    
    def adjust_center(self, action):
        """  
            -  Điều chỉnh vị trí của tâm bounding box dựa trên hành động (action).
            - parameters:
                + action (int): Chỉ số hành động cho Agent 1
                
            - returns:
                + new center (list): Vị trí tọa độ mới của tâm bounding box (Δx, Δy)
        """
        pass
    
    
    def adjust_size(self, action):
        """  
            - Điều chỉnh kích thước của bounding box dựa trên hành động (action).
            - parameters:
                + action (int): Chỉ số hành động cho Agent 2
            
            - returns:
                + new_size (list): Kích thước mới của bounding box (# Δw, Δh) - không vượt quá giới hạn lưới.
        """
        pass
    
    
    def calculate_bounding_box(self, center, size):
        """  
            - Tính toán các tọa độ của bounding box dựa trên vị trí tâm và kích thước đã cho.
            - parameters:
                + center (list): Vị trí tâm của bounding box (x, y).
                + size (list): Kích thước bounding box (width, height).
                
            - Returns:
                + bounding_box (numpy array): Các tọa độ của bounding box (x_min, x_max, y_min, y_max).
        """
        pass
    
    
    def calculate_iou(self, pred_box, gt_box):
        """  
            - Parameters:
                + pred_box (numpy array): Bounding box dự đoán (x_min, x_max, y_min, y_max).
                + gt_box (numpy array): Ground truth box (x_min, x_max, y_min, y_max).
        
            - Returns:
                + iou (float): Giá trị IoU giữa bounding box dự đoán và ground truth.
        """
        pass
    
    
    def render(self,  mode='human'):
        """  
            Hàm render cho môi trường, vẽ bounding box lên ảnh.
        """
        pass
    
    
    def close(self):
        cv.destroyAllWindows() # dọn  dẹp tài nguyên
    
    