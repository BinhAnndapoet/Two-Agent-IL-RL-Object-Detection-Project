# Phân tích file env.py


## Báo cáo Hyperparameters

### Bảng Tóm Tắt

| Tên               | Mặc định | Kiểu   | Mô tả                                                                                                   |
|-------------------|----------|--------|---------------------------------------------------------------------------------------------------------|
| `alpha`           | 0.2      | float  | Tỉ lệ bước di chuyển/thu phóng của bbox (fractional step size).                                         |
| `nu`              | 3.0      | float  | Hệ số nhân reward khi trigger (reward = 2·nu·IoU nếu thành công, –nu nếu thất bại).                      |
| `threshold`       | 0.6      | float  | Ngưỡng IoU để nhận reward dương khi trigger.                                                            |
| `max_steps`       | 200      | int    | Số bước tối đa (không trigger) trước khi episode bị truncated.                                          |
| `trigger_steps`   | 40       | int    | Số bước tối thiểu trước khi trigger được coi là hợp lệ (trigger sớm bị phạt –1).                         |
| `history_size`    | 10       | int    | Độ dài sliding-window lưu one-hot của các base action trong state.                                      |
| `obj_config`      | 0        | int    | Chế độ đối tượng: `0` = single-object (kết thúc episode khi trigger), `1` = multi-object (cho phép nhiều). |
| `base_actions`    | 4        | int    | Số action cơ bản (move/resize) trước action trigger.                                                    |
| `center_actions`  | 24       | int    | Tổng số action trong `CenterEnv` (move, trigger, class/confidence flags, done flags…).                  |
| `size_actions`    | 12       | int    | Tổng số action trong `SizeEnv` (resize, trigger, aspect/confidence flags…).                             |

---

## Mô Tả Chi Tiết

1. **`alpha`** (`float`, mặc định 0.2)  
   - **Vai trò**: Xác định số pixel dịch chuyển hoặc phóng/thu nhỏ theo `int(alpha × dimension)`.  
   - **Ảnh hưởng**:  
     - `alpha` lớn → bước di chuyển/phóng to nhanh, dễ overshoot.  
     - `alpha` nhỏ → điệu chỉnh mịn hơn, cần nhiều bước hơn.

2. **`nu`** (`float`, mặc định 3.0)  
   - **Vai trò**: Hệ số nhân cho reward khi thực hiện trigger:  
     - Thành công: `+2 × nu × IoU`  
     - Thất bại: `−nu`  
   - **Ảnh hưởng**:  
     - `nu` cao → khuyến khích trigger mạnh, rủi ro cao.  
     - `nu` thấp → trigger thận trọng hơn.

3. **`threshold`** (`float`, mặc định 0.6)  
   - **Vai trò**: Ngưỡng IoU tối thiểu để trigger nhận reward dương.  
   - **Ảnh hưởng**:  
     - Ngưỡng cao → yêu cầu bbox khớp khắt khe.  
     - Ngưỡng thấp → dễ trigger thành công.

4. **`max_steps`** (`int`, mặc định 200)  
   - **Vai trò**: Giới hạn số bước move/resize trong một episode; vượt quá → bị truncated.  
   - **Ảnh hưởng**:  
     - Giá trị thấp → episode ngắn, ép agent tối ưu nhanh.  
     - Giá trị cao → nhiều thời gian khám phá.

5. **`trigger_steps`** (`int`, mặc định 40)  
   - **Vai trò**: Số bước tối thiểu trước khi trigger được coi hợp lệ; trigger sớm bị phạt −1.  
   - **Ảnh hưởng**:  
     - Giá trị lớn → bắt agent điều chỉnh đủ lâu trước khi trigger.  
     - Giá trị nhỏ → cho phép trigger sớm hơn.

6. **`history_size`** (`int`, mặc định 10)  
   - **Vai trò**: Số lượng one-hot vectors của base action gần nhất để đưa vào state.  
   - **Ảnh hưởng**:  
     - Lớn → state giàu ngữ cảnh lịch sử (đầu vào có chiều cao hơn).  
     - Nhỏ → state gọn, ít thông tin quá khứ.

7. **`obj_config`** (`int`, mặc định 0)  
   - **Vai trò**: Chọn chế độ single-object vs multi-object:  
     - `0`: single-object (kết thúc episode sau trigger thành công).  
     - `1`: multi-object (cho phép nhiều trigger liên tiếp).  
   - **Ảnh hưởng**:  
     - Chế độ single-object đơn giản hơn.  
     - Multi-object hỗ trợ phát hiện nhiều đối tượng.

8. **`base_actions`** (`int`, mặc định 4)  
   - **Vai trò**: Số action cơ bản (ví dụ: di chuyển 4 hướng hoặc thay đổi kích thước).  
   - **Ảnh hưởng**: Xác định độ chi tiết của biến đổi core.

9. **`center_actions`** (`int`, mặc định 24)  
   - **Vai trò**: Kích thước action-space cho agent Center.  
   - **Thành phần**: 4 moves + 1 trigger + 10 class predictions + 3 confidence bins + 2 done flags.

10. **`size_actions`** (`int`, mặc định 12)  
    - **Vai trò**: Kích thước action-space cho agent Size.  
    - **Thành phần**: 4 resizes + 1 trigger + 3 confidence bins + 4 aspect choices.

-----------------------------------------------------------------------------------

- Mỗi **subclass** chỉ cần override:
  -  `_init_spaces`
  -  `_get_state`
  -  `calculate_reward`
  
-  Các method chung như `reset`, `step`, `render`, `close`, `transform_action`, `_update_history`, `get_state`,  `current_gt_bboxes` đã được cài sẵn ở **BaseDetectionEnv**

-----------------------------------------------------------------------------------

### calculate_iou(bbox1: list[float], bbox2: list[float]) → float
- **Chức năng**: Tính Intersection-over-Union giữa hai bounding-box.
- **Tham số**: bbox1, bbox2: mỗi cái là [x1, y1, x2, y2].

-----------------------------------------------------------------------------------

# 1) class BaseDetectionEnv(Env, ABC)

### __init__(self, env_config: dict)
- **Chức năng**: Khởi tạo chung cho cả hai agent (center & size).

- **Tham số env_config** bắt buộc:

  - `image (np.ndarray)`: ảnh RGB đầu vào.

  - `target_bbox (list[float])`: ground-truth bbox [x1,y1,x2,y2].

  - `agent_type (str)`: 'center' hoặc 'size'.

- Tham số **env_config** tuỳ chọn (override mặc định):

  - `alpha, nu, threshold, max_steps, trigger_steps, history_size, obj_config`

  - `center_actions, size_actions, base_actions`.


### _init_spaces(self)
- **Chức năng**: Thiết lập self.action_space và self.observation_space.

- **Tham số**: Không. (Abstract – bắt buộc override ở subclass.)

### _get_state(self) → np.ndarray
- **Chức năng**: Xây vector quan sát (bbox norm + history).

- **Tham số**: Không. (Abstract – bắt buộc override.)

### calculate_reward(self, new_bbox, prev_bbox, target_bboxes) → float
- **Chức năng**: Tính reward cho bước non-trigger.

- **Tham số**:

  - `new_bbox`: bbo sau khi apply action.

  - `prev_bbox`: bbox trước khi apply.

  - `target_bboxes`: list ground-truth bboxes.

  => **Lưu ý**: Abstract – override ở CenterEnv/SizeEnv với logic khác nhau.


### get_state(self) → np.ndarray
- **Chức năng**: Bọc `_get_state()` thành batch-dim (1,D).

- **Tham số**: Không.

### reset(self, *, seed: int=None, options: dict=None) → (obs, info)
- **Chức năng**: Thiết lập lại env cho episode mới.

- **Tham số (từ Gymnasium API)**:

  - `seed`: (optional) random seed.

  - `options`: (optional) dict tuỳ chỉnh.

- **Trả về**:

  - `obs`: initial observation (raw state).

  - `info`: dict rỗng mặc định


### _update_history(self, action_idx: int) → None
- **Chức năng**: Đẩy one-hot của action_idx vào sliding window self.action_history.

- **Tham số**:

  - `action_idx`: index của base action (0…base_actions-1).


### transform_action(self, action: int) → list[int]
- **Chức năng**: Di chuyển hoặc thay đổi kích thước bbox theo action.

- **Tham số**:

  - `action`: index của action (0…base_actions-1).

- **Trả về**
  - `Bbox mới` [x1,y1,x2,y2].


### step(self, action: int) → (obs, reward, done, truncated, info)
- **Chức năng**: Thực thi một action, tính reward, cập nhật state.

- **Tham số**:

  - `action`: index action (0…n_base-1: move/resize; n_base: trigger).

- **Trả về**:

  - `obs`: next observation (raw).

  - `reward`: float.

  - `done`: True nếu trigger thành công (single-obj).

  - `truncated`: True nếu vượt max_steps.

  - `info`: dict chứa 'iou', 'step', 'triggered'.


### render(self, mode: str='human') → np.ndarray
- **Chức năng**: Vẽ bbox lên ảnh, show window nếu mode='human'.

- **Tham số**:

  - `mode`: 'human' hoặc 'rgb_array'.

- **Trả về**: 
  - `Ảnh có bounding-box`.

### close(self) → None
- **Chức năng**: Đóng mọi cửa sổ render.

### current_gt_bboxes (property)
- **Chức năng**: Trả về list ground-truth bboxes (1 hoặc nhiều).


-----------------------------------------------------------------------------------
# 2) class CenterEnv(BaseDetectionEnv)

### _init_spaces(self)
  - `Action space`: Discrete(self.center_actions)

  - `Observation space`: Box(0,1, shape=(4 + history_size*base_actions,))

### _get_state(self) → np.ndarray
- **Chức năng**:

    - `[x1/W, y1/H, x2/W, y2/H] + flattened action_history`

### calculate_reward(self, new_bbox, prev_bbox, target_bboxes) → float
- **Chức năng**: +1 nếu khoảng cách center→GT giảm, ngược lại −1.

- **Tham số**:

  - `new_bbox, prev_bbox`: list[int]

  - `target_bboxes`: list[list[int]] (chỉ dùng phần tử đầu)


-----------------------------------------------------------------------------------
# 3) class SizeEnv(BaseDetectionEnv)

### _init_spaces(self)
  - `Action space`: Discrete(self.size_actions)

  - `Observation space`: Box(0,1, shape=(4 + history_size*base_actions,))

### _get_state(self) → np.ndarray
- **Chức năng**: Tương tự CenterEnv.

### calculate_reward(self, new_bbox, prev_bbox, target_bboxes, phase) → float

- **Chức năng**: +1 nếu IoU(new_bbox) > IoU(prev_bbox), ngược lại −1.

- **Tham số**:

  - `new_bbox, prev_bbox`: list[int]

  - `target_bboxes`: list[list[int]]
