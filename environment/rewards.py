# --- START OF FILE environment/rewards.py ---
import numpy as np
from utils.metrics import calculate_best_iou

def calculate_reward(env, current_state, previous_state, action):
    """Calculate reward based on proposal specifications for the current phase."""
    target_bboxes = env.current_gt_bboxes
    target_labels = env.current_gt_labels

    if not target_bboxes:
        return -1.0
    
    if env.phase == "center":
        pos_action, class_action = action if isinstance(action, tuple) else (action, None)

        cx_new, cy_new = (current_state[0] + current_state[2]) / 2, (current_state[1] + current_state[3]) / 2
        cx_prev, cy_prev = (previous_state[0] + previous_state[2]) / 2, (previous_state[1] + previous_state[3]) / 2
        
        dis_new = min(np.hypot(cx_new - (b[0] + b[2]) / 2, cy_new - (b[1] + b[3]) / 2) for b in target_bboxes)
        dis_prev = min(np.hypot(cx_prev - (b[0] + b[2]) / 2, cy_prev - (b[1] + b[3]) / 2) for b in target_bboxes)
        R_center = 1.0 if dis_new < dis_prev else -1.0

        R_class = 0.0
        if class_action is not None and 0 <= class_action < len(env.class_names):
            pred_class = env.class_names[class_action]
            gt_class = target_labels[env.current_gt_index]
            R_class = 1.0 if pred_class == gt_class else -1.0
        
        R_conf = 1.0 if pos_action == 5 and calculate_best_iou([current_state], target_bboxes) >= env.threshold else -1.0 if pos_action == 5 else 0.0

        R_redundant = 0.0
        for dcx, dcy in env.detected_centers:
            if np.hypot(cx_new - dcx, cy_new - dcy) < env.alpha * min(env.width, env.height):
                R_redundant = -1.0
                break
        
        R_done = calculate_trigger_reward(env, current_state, target_bboxes) if pos_action == 4 else 0.0

        return 0.4 * R_center + 0.3 * R_class + 0.1 * R_conf + 0.1 * R_redundant + 0.1 * R_done

    else:  # "size"
        iou_current = calculate_best_iou([current_state], target_bboxes)
        iou_previous = calculate_best_iou([previous_state], target_bboxes)
        R_IoU = 1.0 if iou_current > iou_previous else -1.0

        current_aspect = (current_state[2] - current_state[0]) / max(1, current_state[3] - current_state[1])
        target_aspect = (target_bboxes[0][2] - target_bboxes[0][0]) / max(1, target_bboxes[0][3] - target_bboxes[0][1])
        R_aspect = 1.0 if abs(current_aspect - target_aspect) < 0.1 else -1.0
        
        R_conf = 1.0 if action == 5 and iou_current >= env.threshold else -1.0 if action == 5 else 0.0

        return 0.5 * R_IoU + 0.2 * R_IoU + 0.1 * R_aspect + 0.2 * R_conf # R_size is R_IoU

def calculate_trigger_reward(env, current_state, target_bboxes):
    """Calculate reward specifically for a trigger action."""
    iou = calculate_best_iou([current_state], target_bboxes)
    return min(env.nu * 2 * iou, 1.0) if iou >= env.threshold else -1.0
# --- END OF FILE environment/rewards.py ---