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
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

# ==========================================
# 1. 路径锚定与工程挂载
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from MADS_part.func import env_multi_robot
from map.env import graphdict_setup
from MADS_part.mads_network import MultiAgentManager 

# 指定预训练 STGNN 路径 (严格对齐你的目录结构)
stgnn_ckpt_path = os.path.join(root_dir, "STGNN_part", "stgnn", "多地图策略", "2026-03-19_MultiMap_Curriculum_MSE_STGNN.pth")

# ==========================================
# 2. 训练超参数与 MLOps 命名规则
# ==========================================
FIRST_MAP_STAY_LIMIT = 3000  
MAIN_MAP_STAY_LIMIT = 1000   

ROBOT_NUM = 5             
STATE_SPACE = 130         
SEQ_LEN = 30              
GAMMA = 0.95
STEP_LIMIT = 200          
k = 1.0                     # 异构探索度初始权重 (后续动态调整)
b = 0.5                     
BETA = 1 # PG Loss 和 CE Loss 的组合权重 (建议后期可引入动态退火)

log_dir_base = "./maddpg"
model_save_dir = "./mads_policy"
os.makedirs(log_dir_base, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 3. 地图库加载与物理难度排序
# ==========================================
json_path = os.path.join(root_dir, 'map', 'top_100_edges.json')
with open(json_path, 'r', encoding='utf-8') as f:
    maps_dict = json.load(f)

full_map_curriculum = []
for m_name, edges in maps_dict.items():
    if not edges: continue
    num_nodes = max([max(edge[0], edge[1]) for edge in edges]) + 1
    full_map_curriculum.append((m_name, num_nodes))

full_map_curriculum.sort(key=lambda x: x[1])
map_curriculum = full_map_curriculum[-50:]
MAP_COUNT = len(map_curriculum)

EPISODE_LIMIT = FIRST_MAP_STAY_LIMIT + (MAP_COUNT - 1) * MAIN_MAP_STAY_LIMIT 

current_time = datetime.now().strftime("%Y-%m-%d")
run_name = f"{current_time}_{ROBOT_NUM}agents_{EPISODE_LIMIT}episodes_GNN_Policy"

# ==========================================
# 4. 模型与优化器初始化 (对齐 GNN 架构)
# ==========================================
manager = MultiAgentManager(
    num_agents=ROBOT_NUM,
    stgnn_in_dim=2, 
    stgnn_out_dim=32,
    num_nodes=STATE_SPACE,
    hidden_dim=128,
    seq_len=SEQ_LEN,
    freeze_stgnn=True,   # 🛡️ STGNN 冻结保护开启
    device=device
).to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, manager.parameters()), lr=1e-4)

if os.path.exists(stgnn_ckpt_path):
    ckpt = torch.load(stgnn_ckpt_path, map_location=device)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    manager.shared_stgnn.load_state_dict(sd, strict=False)
    print(f"✅ [Load] 已注入预训练 STGNN 雷达: {stgnn_ckpt_path}")
else:
    print(f"⚠️ [Warning] 未找到模型 {stgnn_ckpt_path}，雷达将随机初始化！")

# ==========================================
# 5. 核心功能函数
# ==========================================
def get_map_sequential_v2(current_ep, sorted_map_list):
    if current_ep < FIRST_MAP_STAY_LIMIT:
        idx = 0
    else:
        ep_in_main = current_ep - FIRST_MAP_STAY_LIMIT
        idx = 1 + (ep_in_main // MAIN_MAP_STAY_LIMIT) 
    
    idx = min(idx, len(sorted_map_list) - 1)
    chosen_map, chosen_nodes = sorted_map_list[idx]
    return chosen_map, chosen_nodes, idx

# 🌟 注意：如果已经在 env_multi_robot 内重构了目标的游走逻辑，这里的 p_stay 参数可配合调整
def manualInitEmm(graph, current_state_space, p_stay=0.95):
    lazy_gumma = np.zeros((current_state_space, current_state_space), dtype=float)
    p_move = 1.0 - p_stay
    for i in range(current_state_space):
        if i in graph:
            neighbors = [n for n in graph[i] if n != i]
            lazy_gumma[i][i] = p_stay
            if neighbors:
                for j in neighbors: lazy_gumma[i][j] = p_move / len(neighbors)
            else: lazy_gumma[i][i] = 1.0
        else:
            lazy_gumma[i][i] = 1.0
    return lazy_gumma

def get_hop_distance(graph, start, target):
    if start == target: return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        curr, dist = queue.popleft()
        if curr == target: return dist
        for neighbor in graph.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')

# ==========================================
# 6. 主训练循环
# ==========================================
def main():
    writer = SummaryWriter(os.path.join(log_dir_base, run_name))
    recent_success = deque(maxlen=100)
    recent_steps = deque(maxlen=100)
    
    current_map_name, current_map_nodes, map_idx = get_map_sequential_v2(0, map_curriculum)
    graph_ = graphdict_setup(current_map_name, STATE_SPACE)
    
    print(f"🚀 启动 {EPISODE_LIMIT} Episode GNN拓扑感知训练。实验名称: {run_name}")
    print(f"🗺️ 挂载后 {MAP_COUNT} 张地图。起始难度: {map_curriculum[0][1]} 节点, 终极难度: {map_curriculum[-1][1]} 节点")
    
    last_map_idx = map_idx

    for i_ep in range(EPISODE_LIMIT):
        # --- 课程与地图切换 ---
        if i_ep > 0:
            target_map_name, target_map_nodes, target_map_idx = get_map_sequential_v2(i_ep, map_curriculum)
            if target_map_idx != last_map_idx:
                current_map_name = target_map_name
                current_map_nodes = target_map_nodes
                graph_ = graphdict_setup(current_map_name, STATE_SPACE)
                
                stay_limit = FIRST_MAP_STAY_LIMIT if target_map_idx == 1 else MAIN_MAP_STAY_LIMIT
                print(f"🌍 [地图切换] Ep {i_ep}: -> 锁定第 {target_map_idx + 1}/{MAP_COUNT} 张地图: {current_map_name} (节点数: {current_map_nodes})")
                last_map_idx = target_map_idx
            
        # 环境构建
        trans_prob = manualInitEmm(graph_, STATE_SPACE, p_stay=0.95)
        valid_nodes = list(graph_.keys())
        adj_mask = torch.zeros((STATE_SPACE, STATE_SPACE), device=device)
        for u in valid_nodes:
            adj_mask[u, graph_[u]] = 1.0
        
        # 🌟 [架构师级修复]：强制确保 edge_index 为 long 类型，完美匹配 GATConv 需求
        edge_index = torch.tensor([[u, v] for u in graph_ for v in graph_[u]], 
                                  dtype=torch.long, device=device).t().contiguous()

        # 重置环境
        s_target = random.choice(valid_nodes)
        s_robot = []
        min_hops = min(8, max(2, len(valid_nodes) // 15)) 
        
        for _ in range(ROBOT_NUM):
            candidate = random.choice(valid_nodes)
            attempts = 0
            while get_hop_distance(graph_, candidate, s_target) < min_hops and attempts < 100:
                candidate = random.choice(valid_nodes)
                attempts += 1
            if candidate == s_target:
                safe_nodes = [n for n in valid_nodes if n != s_target]
                candidate = random.choice(safe_nodes) if safe_nodes else s_target
            s_robot.append(candidate)
            
        env = env_multi_robot(graph_, trans_prob, STATE_SPACE, s_robot, s_target, ROBOT_NUM)
        obs_robot, _, _ = env.reset()
        
        # 维护 30 步历史序列
        cur_poses = [int(np.argmax(p)) for p in obs_robot]
        history_poses = deque([cur_poses] * SEQ_LEN, maxlen=SEQ_LEN)
        init_prob = torch.zeros(STATE_SPACE, 1, device=device)
        init_prob[valid_nodes] = 1.0 / len(valid_nodes)
        history_target_probs = deque([init_prob] * SEQ_LEN, maxlen=SEQ_LEN)
        
        log_probs_list, rewards_list, ce_loss_list = [], [], []
        visited = set(cur_poses)

        # --- Step 循环 ---
        for t in range(STEP_LIMIT):
            # 动态 K 值计算
            k_val = k * np.exp(- 0.5 * (len(visited) / len(valid_nodes)))
            
            # 构造 Tensor 输入
            seq_feat = torch.zeros((SEQ_LEN, STATE_SPACE, 1), device=device)
            for time_idx, step_poses in enumerate(history_poses):
                seq_feat[time_idx, torch.tensor(step_poses).long(), 0] = 1.0
            seq_probs = torch.stack(list(history_target_probs))
            
            # 🌟 完美对齐：传入 edge_index，激活网络内部的 GNN 拓扑弥漫
            logits, latest_probs, _ = manager(k_val, seq_feat, seq_probs, edge_index, cur_poses)
            
            # Logits 掩码处理与动作采样
            mask = adj_mask[cur_poses]
            probs = F.softmax(logits + (mask - 1) * 1e9, dim=1)
            dist = Categorical(probs)
            actions = dist.sample()
            
            # CE 互斥 Loss (鼓励探索不同路线)
            step_ce = 0.0
            for i in range(ROBOT_NUM):
                for j in range(ROBOT_NUM):
                    if i != j:
                        step_ce += (probs[i] * torch.log(probs[j].detach() + 1e-8)).sum()
            ce_loss_list.append(step_ce)
            
            # 环境交互 (已挂载包含懒惰惩罚和探索奖励的新 Env)
            next_obs, _, rewards, next_target, done = env.step(actions.cpu().numpy())
            
            # 更新历史与状态
            cur_poses = [int(np.argmax(p)) for p in next_obs]
            history_poses.append(cur_poses)
            history_target_probs.append(latest_probs.unsqueeze(-1))
            visited.update(cur_poses)
            
            log_probs_list.append(dist.log_prob(actions))
            rewards_list.append(torch.tensor(rewards, device=device, dtype=torch.float))
            
            if done: break
        
        # --- 梯度更新 ---
        if len(log_probs_list) > 0:
            R, returns = 0, []
            for r in reversed(rewards_list):
                R = r + GAMMA * R
                returns.insert(0, R)
            returns = torch.stack(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            pg_loss = -(torch.stack(log_probs_list) * returns).sum()
            a = pg_loss * BETA
            
            if len(ce_loss_list) > 0:
                mean_ce_over_T = torch.stack(ce_loss_list).mean() 
                b = mean_ce_over_T * ((1.0 - BETA) / (ROBOT_NUM - 1))
            else:
                b = torch.tensor(0.0, device=device)
            
            total_loss = a + b
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(manager.parameters(), 0.5)
            optimizer.step()

        # --- 监控面板 ---
        recent_success.append(1 if done else 0)
        recent_steps.append(t + 1)
        
        if i_ep > 0 and i_ep % 100 == 0:
            avg_win = (sum(recent_success) / len(recent_success)) * 100
            avg_s = sum(recent_steps) / len(recent_steps)
            print(f"Ep {i_ep:05d} | Win: {avg_win:3.0f}% | Steps: {avg_s:5.1f} | Map: {current_map_name} | Nodes: {current_map_nodes}")
            
            writer.add_scalar('Train/WinRate', avg_win, i_ep)
            writer.add_scalar('Train/AvgSteps', avg_s, i_ep)
            writer.add_scalar('Loss/PG_Loss', a.item(), i_ep)
            writer.add_scalar('Loss/CE_Loss', b.item(), i_ep)

        # 保存断点
        if i_ep > 0 and i_ep % 5000 == 0:
            save_path = os.path.join(model_save_dir, f"{run_name}.pkl")
            torch.save(manager.state_dict(), save_path)

    writer.close()
    
    # 最终模型保存
    final_save_path = os.path.join(model_save_dir, f"{run_name}.pkl")
    torch.save(manager.state_dict(), final_save_path)
    print(f"✅ 训练圆满完成。最终模型已保存至: {final_save_path}")

if __name__ == '__main__':
    main()
