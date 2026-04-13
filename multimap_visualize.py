# -*- coding: utf-8 -*-
import sys
import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime
from PIL import Image
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# ==========================================
# 1. 跨层级环境挂载与路径解析
# ==========================================
# 获取当前 visualize 文件夹的路径
current_dir = os.path.dirname(os.path.abspath(__file__)) 
# 获取根目录 MADS 路径
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 确保导入路径对齐
from MADS_part.func import env_multi_robot
from map.env import graphdict_setup
from MADS_part.mads_network import STGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 💡 你可以在这里随意切换你想可视化的地图名字
ENV_NAME = "map_9244"  
SEQ_LEN = 30
ROBOT_NUM = 2      # 可视化时固定展示 2 个机器人
INTERP_STEPS = 10  # 连续滑动插值帧数

today = datetime.now().strftime("%Y-%m-%d")
# 自动匹配你刚刚训练出的多地图轻量级模型
MODEL_PATH = os.path.join(root_dir, 'STGNN_part', "stgnn", '多地图策略', f"2026-03-16_MultiMap_Curriculum_KL_STGNN.pth")

if not os.path.exists(MODEL_PATH):
    print(f"❌ 未找到模型: {MODEL_PATH}")
    print("请确认你已经运行完了刚刚的 KL.py，或者检查文件名日期是否匹配！")
    exit()

# ==========================================
# 2. JSON 坐标解析与图片特征提取
# ==========================================
map_dir = os.path.join(root_dir, 'map')
coords_path = os.path.join(map_dir, 'top_100_coordinates.json')
img_path = os.path.join(map_dir, 'images', f"{ENV_NAME}.png")

if not os.path.exists(coords_path) or not os.path.exists(img_path):
    print(f"❌ 未找到坐标 JSON ({coords_path}) 或 图像文件 ({img_path})！")
    exit()

# 读取 JSON 数据
with open(coords_path, 'r', encoding='utf-8') as f:
    all_coords = json.load(f)

if ENV_NAME not in all_coords:
    print(f"❌ 地图 {ENV_NAME} 不在 JSON 文件中！")
    exit()

map_info = all_coords[ENV_NAME]
STATE_SPACE = map_info["num_rooms"]
# JSON 里的 key 是字符串，需要转回 int
pos = {int(k): v for k, v in map_info["coordinates"].items()}

# 读取地图图像并提取可行走掩码 (白底黑边结构，提取高亮区域)
map_img_gray = np.array(Image.open(img_path).convert('L'))
walkable_mask = map_img_gray > 128  
img_h, img_w = map_img_gray.shape

# ==========================================
# 3. 核心坐标映射引擎 (Shapely -> Pixel)
# ==========================================
print(f"正在构建 {ENV_NAME} 空间映射与高斯热力场...")

y_indices, x_indices = np.where(walkable_mask)
if len(x_indices) == 0:
    print("❌ 提取可行走区域失败！请检查图片特征。")
    exit()

# 1. 获取图像中真实建筑区域的边界 bounding box
img_min_x, img_max_x = x_indices.min(), x_indices.max()
img_min_y, img_max_y = y_indices.min(), y_indices.max()

# 2. 获取 JSON 科学坐标系的边界 bounding box
raw_xs = [p[0] for p in pos.values()]
raw_ys = [p[1] for p in pos.values()]
raw_min_x, raw_max_x = min(raw_xs), max(raw_xs)
raw_min_y, raw_max_y = min(raw_ys), max(raw_ys)

# 3. 动态自适应缩放 (0.95 用于兼容 Matplotlib 保存时 bbox_inches 的微小白边)
scale_x = (img_max_x - img_min_x) / (raw_max_x - raw_min_x) if raw_max_x > raw_min_x else 1
scale_y = (img_max_y - img_min_y) / (raw_max_y - raw_min_y) if raw_max_y > raw_min_y else 1
scale = min(scale_x, scale_y) * 0.95  

raw_cx, raw_cy = (raw_min_x + raw_max_x) / 2, (raw_min_y + raw_max_y) / 2
img_cx, img_cy = (img_min_x + img_max_x) / 2, (img_min_y + img_max_y) / 2

# 4. 执行转换，将所有节点的坐标印在像素空间
pixel_pos = {}
for n, (lx, ly) in pos.items():
    px = img_cx + (lx - raw_cx) * scale
    # ⚠️ 必须翻转 Y 轴：因为计算机视觉 Y 轴向下，几何坐标系 Y 轴向上！
    py = img_cy - (ly - raw_cy) * scale  
    pixel_pos[n] = np.array([px, py])

# ==========================================
# 4. 空间网格划分与高斯热力场构建
# ==========================================
walkable_points = np.column_stack((x_indices, y_indices))
valid_nodes = sorted(list(pos.keys()))
node_centers = np.array([pixel_pos[n] for n in valid_nodes])

# 利用 KD-Tree 快速将所有像素划归为距离最近的房间 ID
kdtree = cKDTree(node_centers)
_, nearest_idx = kdtree.query(walkable_points)
room_assignment_map = np.zeros((img_h, img_w), dtype=int) - 1
room_assignment_map[y_indices, x_indices] = np.array(valid_nodes)[nearest_idx]

# 预计算所有节点的高斯热力基座 (自适应缩放方差)
X, Y = np.meshgrid(np.arange(img_w), np.arange(img_h))
node_gaussians = np.zeros((STATE_SPACE, img_h, img_w))
sigma = max(scale * 0.5, 6.0)  
for n in valid_nodes:
    cx, cy = pixel_pos[n]
    d2 = (X - cx)**2 + (Y - cy)**2
    node_gaussians[n] = np.exp(-d2 / (2 * sigma**2))

# ==========================================
# 5. 模型挂载与预演推演
# ==========================================
graph_ = graphdict_setup(ENV_NAME, STATE_SPACE)

model = STGNN(
    in_dim=2, 
    gat_hid=32,            
    gat_heads=4,          
    attn_hid=64,          
    num_attn_layers=2,    
    num_attn_heads=4,     
    out_dim=32,           
    seq_len=SEQ_LEN, 
    device=device
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 包含自环的安全边构建
edge_index = torch.tensor([[u, v] for u in graph_ for v in graph_[u]], dtype=torch.long).t().contiguous().to(device)
self_loops = torch.arange(STATE_SPACE, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
edge_index = torch.cat([edge_index, self_loops], dim=1) if edge_index.numel() > 0 else self_loops

def get_greedy_radar_action(robot_poses, history, ptb_pred, graph):
    actions = []
    current_team = set(robot_poses)
    chosen_targets = set() 

    for i, r_pos in enumerate(robot_poses):
        valid_moves = [n for n in graph[r_pos] if n != r_pos]
        if not valid_moves: 
            actions.append(r_pos)
            chosen_targets.add(r_pos)
            continue
            
        move_probs = np.array([ptb_pred[n] for n in valid_moves])
        move_probs += np.random.uniform(0, 1e-5, size=len(valid_moves))
        
        lookback = min(4, len(history))
        for step_back in range(1, lookback + 1):
            if history[-step_back][i] in valid_moves:
                idx = valid_moves.index(history[-step_back][i])
                move_probs[idx] -= (1.0 / (2**(step_back-1)))
                
        for j, n in enumerate(valid_moves):
            if n in current_team or n in chosen_targets: 
                move_probs[j] -= 5.0 
                
        best_action = valid_moves[np.argmax(move_probs)]
        actions.append(best_action)
        chosen_targets.add(best_action) 
        
    return actions

def manualInitEmm(graph, p_stay=0.1):
    lazy_gumma = np.zeros((STATE_SPACE, STATE_SPACE), dtype=float)
    p_move = 1.0 - p_stay
    for i in range(STATE_SPACE):
        if i in graph:
            neighbors = [n for n in graph[i] if n != i]
            lazy_gumma[i][i] = p_stay
            if neighbors:
                for j in neighbors: lazy_gumma[i][j] = p_move / len(neighbors)
            else: lazy_gumma[i][i] = 1.0
        else: lazy_gumma[i][i] = 1.0
    return lazy_gumma

transition_prob = manualInitEmm(graph_, p_stay=0.5)

# ==========================================
# 🌟 新增：自适应拓扑距离发牌器 (Spawning Engine)
# ==========================================
def get_hop_distance(graph, start, target):
    """BFS 获取两点在图中的最短跳数"""
    if start == target: return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        curr, dist = queue.popleft()
        if curr == target: return dist
        for neighbor in graph[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')

all_nodes = list(graph_.keys())
# 自适应设定：地图越庞大，要求开局探员和猎物离得越远 (至少拉开 3 个房间的距离，最多不超过 8 个房间)
MIN_HOPS = min(8, max(3, STATE_SPACE // 12)) 

rand_agent_poses, rand_target_pos = None, None

# 尝试最多 500 次找到满足距离要求的初始位置
for _ in range(500):
    spawn_points = random.sample(all_nodes, ROBOT_NUM + 1)
    agents = spawn_points[:ROBOT_NUM]
    target = spawn_points[ROBOT_NUM]
    
    # 确保所有的探员离猎物的拓扑距离都 >= MIN_HOPS
    if all(get_hop_distance(graph_, a, target) >= MIN_HOPS for a in agents):
        rand_agent_poses = agents
        rand_target_pos = target
        break

# 兜底防护：如果地图太小或结构太紧凑导致找不到，则自动放宽条件
if rand_agent_poses is None:
    print(f"⚠️ 地图拓扑紧凑，无法满足 {MIN_HOPS} 跳的安全距离，已自动放宽开局限制！")
    spawn_points = random.sample(all_nodes, ROBOT_NUM + 1)
    rand_agent_poses = spawn_points[:ROBOT_NUM]
    rand_target_pos = spawn_points[ROBOT_NUM]

print(f"🎬 初始化点位 -> 探员: {rand_agent_poses} | 猎物: {rand_target_pos} (最短物理相距 ≥ {MIN_HOPS} 个房间)")

# ==========================================
# 初始化推演环境
env = env_multi_robot(graph_, transition_prob, STATE_SPACE, rand_agent_poses, rand_target_pos, ROBOT_NUM)
one_hot, _, target_state = env.reset()
robot_poses = [np.argmax(p) for p in one_hot]

history = deque([robot_poses], maxlen=SEQ_LEN)
init_prob = torch.ones(STATE_SPACE, device=device)
for p in robot_poses: init_prob[p] = 0.0
prob_history = deque([init_prob / (init_prob.sum() + 1e-8)], maxlen=SEQ_LEN)

simulation_data = []

print("正在预计算推演轨迹...")
for step in range(150):
    with torch.no_grad():
        h_list = list(history)
        if len(h_list) < SEQ_LEN:
            h_list = [h_list[0]] * (SEQ_LEN - len(h_list)) + h_list
            
        n_feat = torch.zeros((SEQ_LEN, STATE_SPACE, 1), device=device)
        for t_idx, current_step_poses in enumerate(h_list):
            p_tensor = torch.tensor(current_step_poses, device=device).long()
            n_feat[t_idx, p_tensor, 0] = 1.0
        
        p_hist = list(prob_history)
        if len(p_hist) < SEQ_LEN:
            p_hist = [p_hist[0]] * (SEQ_LEN - len(p_hist)) + p_hist
        p_feat = torch.stack(p_hist).unsqueeze(-1)
        
        _, latest_probs, _ = model(n_feat, p_feat, edge_index)
        
        latest_probs = latest_probs / (latest_probs.sum() + 1e-8)
        ptb_pred = latest_probs.cpu().numpy()
        
    prob_history.append(latest_probs)
    simulation_data.append((list(robot_poses), target_state, ptb_pred))
    # ==========================================
    # 📊 实时状态监控与极值探针
    # ==========================================
    max_prob = np.max(ptb_pred)
    max_node = np.argmax(ptb_pred)
    min_prob = np.min(ptb_pred)
    min_node = np.argmin(ptb_pred)
        
    print(f"[Step {step+1:03d}] 🤖 探员: {robot_poses} | 🎯 真实: {target_state} | "
          f"预测Max: {max_prob:.5f} (节点 {max_node}) | Min: {min_prob:.5f} (节点 {min_node})")
    # ==========================================    
    actions = get_greedy_radar_action(robot_poses, history, ptb_pred, graph_)
    next_p, _, _, next_t, done = env.step(actions)
    robot_poses = [np.argmax(p) for p in next_p]
    history.append(robot_poses)
    target_state = next_t
    
    if done:
        simulation_data.append((list(robot_poses), target_state, ptb_pred))
        print(f"🎯 预演完成：目标在第 {step+1} 步被抓获！")
        break

# ==========================================
# 6. 高保真时空晕染渲染引擎
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

cmap_heatmap = plt.cm.YlOrRd
BG_COLOR = np.array([0.0, 0.0, 0.0, 1.0])            

image_layer = ax.imshow(np.zeros((img_h, img_w, 4)), origin='upper')
target_scatter = ax.scatter([], [], c='#00FF00', s=700, marker='*', edgecolors='white', linewidths=1.5, zorder=5)
agent_scatter = ax.scatter([], [], c='#00BFFF', s=200, marker='o', edgecolors='white', linewidths=2.0, zorder=4)

step_text = ax.text(30, 60, "", fontsize=24, color='#FFD700', weight='bold', family='monospace')
status_text = ax.text(img_w/2, img_h/2, "", ha='center', va='center', fontsize=40, color='#00FF00', weight='bold', alpha=0)

total_simulation_steps = len(simulation_data)
total_frames = (total_simulation_steps - 1) * INTERP_STEPS

def update(frame):
    step_idx = frame // INTERP_STEPS
    alpha = (frame % INTERP_STEPS) / float(INTERP_STEPS)
    
    if step_idx >= total_simulation_steps - 1:
        step_idx = total_simulation_steps - 2
        alpha = 1.0

    robots_t, target_t, prob_t = simulation_data[step_idx]
    robots_t1, target_t1, prob_t1 = simulation_data[step_idx + 1]

    interp_prob = prob_t * (1 - alpha) + prob_t1 * alpha
    H = np.tensordot(interp_prob, node_gaussians, axes=([0], [0]))
    
    valid_H = H[walkable_mask]
    vmax_val = np.max(valid_H) if len(valid_H) > 0 else 1.0
    vmin_val = np.min(valid_H) if len(valid_H) > 0 else 0.0
    if vmax_val == vmin_val:  
        vmax_val = vmin_val + 1e-5
        
    norm = mcolors.Normalize(vmin=vmin_val, vmax=vmax_val)
    color_mapped_img = cmap_heatmap(norm(H))
    
    frame_pixels = np.zeros((img_h, img_w, 4))
    frame_pixels[walkable_mask] = color_mapped_img[walkable_mask]
    frame_pixels[~walkable_mask] = BG_COLOR
    image_layer.set_data(frame_pixels)

    pos_target_t = pixel_pos[target_t]
    pos_target_t1 = pixel_pos[target_t1]
    interp_target = pos_target_t * (1 - alpha) + pos_target_t1 * alpha
    target_scatter.set_offsets(np.array([interp_target]))

    interp_agents = []
    for i in range(ROBOT_NUM):
        p_t = pixel_pos[robots_t[i]]
        p_t1 = pixel_pos[robots_t1[i]]
        interp_agents.append(p_t * (1 - alpha) + p_t1 * alpha)
    agent_scatter.set_offsets(np.array(interp_agents))

    step_text.set_text(f"SYSTEM STEP: {step_idx + 1}")
    if step_idx == total_simulation_steps - 2 and alpha == 1.0:
        status_text.set_text("TARGET CAUGHT!")
        status_text.set_bbox(dict(facecolor='black', alpha=0.8, edgecolor='#00FF00', boxstyle='round,pad=0.5'))
        status_text.set_alpha(1.0)

    ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0); ax.axis('off')

print(f"正在生成 {ENV_NAME} 高分辨率连续运动序列...")
ani = animation.FuncAnimation(fig, update, frames=total_frames + 20, interval=50) 
save_path = os.path.join(current_dir, f"{ENV_NAME}_robust_heatmap.gif")
ani.save(save_path, writer='pillow', fps=10)
print(f"✅ 成功生成终极渲染动图：{save_path}")