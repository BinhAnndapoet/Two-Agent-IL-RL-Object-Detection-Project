# --- START OF FILE training/policies.py ---
import random
from utils.metrics import calculate_iou

def select_expert_action_center(env, current_bbox, target_bbox, target_label):
    """Determines the expert action for the Center Agent."""
    current_center = [(current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2]
    target_center = [(target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2]
    dx = target_center[0] - current_center[0]
    dy = target_center[1] - current_center[1]
    
    alpha_w = env.alpha * (current_bbox[2] - current_bbox[0])
    alpha_h = env.alpha * (current_bbox[3] - current_bbox[1])

    iou = calculate_iou(current_bbox, target_bbox)
    
    # Decide if the "done" action should be taken
    is_centered = abs(dx) < alpha_w / 2 and abs(dy) < alpha_h / 2
    done = 1.0 if is_centered and iou >= env.threshold else 0.0
    
    # Choose position action
    if is_centered:
        pos_action = 4 # Trigger action
    else:
        # Determine the best move direction
        actions = []
        if dx > alpha_w: actions.append(0)  # Move right
        if dx < -alpha_w: actions.append(1) # Move left
        if dy < -alpha_h: actions.append(2) # Move up
        if dy > alpha_h: actions.append(3)  # Move down
        pos_action = random.choice(actions) if actions else 4 # If no move is good, trigger
    
    # Choose class action
    class_action = env.get_class_names().index(target_label) if target_label in env.get_class_names() else 0

    return pos_action, class_action, 1.0, done

def select_expert_action_size(env, current_bbox, target_bbox):
    """Determines the expert action for the Size Agent."""
    current_w = current_bbox[2] - current_bbox[0]
    current_h = current_bbox[3] - current_bbox[1]
    target_w = target_bbox[2] - target_bbox[0]
    target_h = target_bbox[3] - target_bbox[1]
    
    alpha_w = env.alpha * current_w
    alpha_h = env.alpha * current_h

    dw = target_w - current_w
    dh = target_h - current_h

    # Decide if the "done" action should be taken
    is_sized_correctly = abs(dw) < alpha_w and abs(dh) < alpha_h
    
    if is_sized_correctly:
        size_action = 4 # Trigger action
    else:
        # Determine best size adjustment
        actions = []
        if dw > 0: actions.append(3)  # Taller (Make fatter in proposal, mapping might differ)
        if dw < 0: actions.append(3)  # Should be make thinner
        if dh > 0: actions.append(2)  # Fatter (Make taller in proposal)
        if dh < 0: actions.append(2)  # Should be make shorter
        # Let's use the proposal's logic: 0: bigger, 1: smaller, 2: taller, 3: fatter
        actions = []
        if dw > alpha_w: actions.append(3) # fatter
        if dw < -alpha_w: actions.append(3) # thinner (map to fatter for simplicity or add action)
        if dh > alpha_h: actions.append(2) # taller
        if dh < -alpha_h: actions.append(2) # shorter (map to taller)
        if dw > 0 and dh > 0: actions.append(0) # bigger
        if dw < 0 and dh < 0: actions.append(1) # smaller

        size_action = random.choice(actions) if actions else 4 # If no good move, trigger

    return size_action
# --- END OF FILE training/policies.py ---