# Phân tích file agent.py


### Hàm & 	                                                Chức năng: **DQNAgent**
- DQN agent cơ bản, đại diện cho một agent học chính sách Q.

- `__init__(self, ...)`:	                                Khởi tạo agent, setup network, optimizer, replay buffer, các chỉ số training.

- `select_action(self, state)`:                             Chọn hành động theo epsilon-greedy: ngẫu nhiên (explore) hoặc theo Q-value (exploit).

- `expert_agent_action_selection(self)	Placeholder`:       bắt buộc các subclass phải override (ví dụ CenterDQNAgent/SizeDQNAgent).

- `update(self)`:	                                        Cập nhật policy network từ sample minibatch bằng Bellman equation.

- `update_epsilon(self)`:	                                Giảm dần epsilon theo chiến lược epsilon-decay.

- `train(self)`:	                                        Tiến hành vòng lặp huấn luyện chính: chọn action, lưu transition, update mạng.

- `save(self, path="models/dqn")`:	                        Lưu trạng thái mạng và thông tin huấn luyện.

- `load(self, path="models/dqn")`:	                        Load trạng thái mạng và thông tin huấn luyện đã lưu.




### Hàm & 	                                                Chức năng: **CenterDQNAgent**
- CenterDQNAgent dùng cho dịch chuyển tâm bbox.

- `__init__(self, ...)`:	                                Gọi DQNAgent init nhưng set số action = 4 (lên, xuống, trái, phải).

- `expert_agent_action_selection(self)`:                    Chọn action dẫn đến cải thiện khoảng cách tâm bbox.



### Hàm & 	                                                Chức năng: **SizeDQNAgent**
- SizeDQNAgent dùng cho điều chỉnh kích thước bbox.

- `__init__(self, ...)`:                                    Gọi DQNAgent init nhưng set số action = 4 (to, nhỏ, béo, cao).

- `expert_agent_action_selection(self)`                 	Chọn action dẫn đến cải thiện IoU bbox.



# Các hàm gọi từ env

- `env.get_state()`:            	                                    Lấy trạng thái hiện tại (state) của environment.

- `env.action_space.sample()`:	                                        Lấy ngẫu nhiên một action từ action space.

- `env.bbox`:	                                                        Lấy bounding box hiện tại. (Biến, không phải hàm)

- `env.current_gt_bboxes`:      	                                    Lấy ground-truth bounding boxes của ảnh. (Biến)

- `env.transform_action(action)`:	                                    Tính toán bbox mới sau khi thực hiện action cụ thể.

- `env.calculate_reward(new_state, old_state, target_bboxes, phase)`: 	Tính toán reward giữa 2 bbox dựa trên phase: "center" hoặc "size".


- `env.reset()`:                                                    	Reset môi trường về trạng thái ban đầu.

- `env.step(action)`:                                               	Thực hiện action và nhận về (obs, reward, terminated, truncated, info).

- `env.step_count`:                                                 	Số bước đã đi trong 1 episode (Biến).

- `env.epochs`:                                                     	Số epochs đã hoàn thành (Biến).


### Mục còn thiếu sót trong agent.py

- Chưa khai báo các tham số:
  - BATCH_SIZE
  - GAMMA
  - MAX_STEPS
  - Transition
  - USE_EPISODE_CRITERIA
  - SUCCES_CRITERIA_EPS
  - SUCCES_CRITERIA_EPOCHS