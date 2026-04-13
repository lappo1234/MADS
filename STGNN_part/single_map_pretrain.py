# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime
from tensorboardX import SummaryWriter

# ==========================================
# 1. 环境挂载
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 确保这里的导入路径与你的工程结构一致
from MADS_part.func import env_multi_robot
from map.env import graphdict_setup
from MADS_part.mads_network import STGNN

# ==========================================
# 2. 自动化配置与参数
# ==========================================
ENV_NAME = "museum"  
STATE_SPACE = 71
SEQ_LEN = 30
PRETRAIN_EPISODES = 80000
ROBOT_NUM_MIN = 1
ROBOT_NUM_MAX = 8

# 核心混合损失权重
LAMBDA_VISITED, LAMBDA_FLOW, LAMBDA_SPATIAL = 5.0, 10.0, 0.5

today = datetime.now().strftime("%Y-%m-%d")
run_name = f"{today}_{ENV_NAME}_STGNN_Robust"

MODEL_SAVE_DIR = os.path.join(current_dir, "stgnn", '单地图策略')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{run_name}.pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=os.path.join(root_dir, "STGNN_part", "stgnn_log", 'singlemap', run_name))

graph_ = graphdict_setup(ENV_NAME, STATE_SPACE)
edge_index = torch.tensor([[u, v] for u in graph_ for v in graph_[u]], dtype=torch.long).t().contiguous().to(device)

# ==========================================
# 3. 核心功能组件 (已恢复正确的 Padding 逻辑)
# ==========================================

def build_node_feature_seq(history):
    """
    处理变长机器人位置，必须使用 Padding 保证 Transformer 序列长度稳定
    """
    feat = torch.zeros((SEQ_LEN, STATE_SPACE, 1), device=device)
    poses_list = list(history)
    
    # 【核心保留】序列填充：用最早的一帧填满不足的序列长度
    if len(poses_list) < SEQ_LEN:
        poses_list = [poses_list[0]] * (SEQ_LEN - len(poses_list)) + poses_list
    
    for t_idx, current_step_poses in enumerate(poses_list):
        p_tensor = torch.tensor(current_step_poses, device=device).long()
        feat[t_idx, p_tensor, 0] = 1.0
    return feat

def build_prob_feature_seq(prob_history):
    probs = list(prob_history)
    # 【核心保留】概率序列的 Padding 逻辑
    if len(probs) < SEQ_LEN:
        probs = [probs[0]] * (SEQ_LEN - len(probs)) + probs
    return torch.stack(probs).unsqueeze(-1)

def manualInitEmm(graph, p_stay=0.5):
    lazy_gumma = np.zeros((STATE_SPACE, STATE_SPACE), dtype=float)
    p_move = 1.0 - p_stay
    for i in range(STATE_SPACE):
        neighbors = [n for n in graph[i] if n != i]
        lazy_gumma[i][i] = p_stay
        if neighbors:
            for j in neighbors: lazy_gumma[i][j] = p_move / len(neighbors)
        else: lazy_gumma[i][i] = 1.0
    return lazy_gumma

# ==========================================
# 4. 主训练循环
# ==========================================

def main():
    print(f"📡 启动鲁棒增强预训练 (已恢复 MSE 机制): {run_name}")
    print(f"🤖 机器人规模范围: {ROBOT_NUM_MIN} - {ROBOT_NUM_MAX}")
    
    model = STGNN(in_dim=2, out_dim=32, seq_len=SEQ_LEN, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    window_loss = 0.0

    for i_ep in range(1, PRETRAIN_EPISODES + 1):
        if (i_ep - 1) % 100 == 0:
            window_loss = 0.0

        # --- 随机化当前回合参数 ---
        current_robot_num = random.randint(ROBOT_NUM_MIN, ROBOT_NUM_MAX)
        p_stay = random.uniform(0.3, 0.95)
        transition_prob = manualInitEmm(graph_, p_stay=p_stay)
        
        # --- 随机起点采样 (Agent + Target) ---
        all_nodes = list(range(STATE_SPACE))
        spawn_points = random.sample(all_nodes, current_robot_num + 1)
        rand_agent_poses = spawn_points[:current_robot_num]
        rand_target_pos = spawn_points[current_robot_num]

        # 初始化环境
        env = env_multi_robot(graph_, transition_prob, STATE_SPACE, 
                               rand_agent_poses, rand_target_pos, current_robot_num)
        
        one_hot, _, target_state = env.reset()
        robot_poses = [np.argmax(p) for p in one_hot] 
        
        # 初始化历史
        history = deque([robot_poses], maxlen=SEQ_LEN)
        
        # 贝叶斯屏蔽初始化：将机器人所在节点的先验概率置为 0
        init_prob = torch.ones(STATE_SPACE, device=device)
        for p in robot_poses:
            init_prob[p] = 0.0
        init_prob = init_prob / init_prob.sum() 
        prob_history = deque([init_prob], maxlen=SEQ_LEN)

        visited_mask = torch.zeros(STATE_SPACE, device=device)
        for p in robot_poses: visited_mask[p] = 1.0

        total_ep_loss = 0
        
        for t in range(120):
            node_feat = build_node_feature_seq(history)
            prob_feat = build_prob_feature_seq(prob_history)

            _, latest_probs, logits = model(node_feat, prob_feat, edge_index)
            
            # 1. 直接从你环境里取出预计算好的“目标到所有节点的最短距离”
            # env.dist_matrix_ 的形状是 (N, N)，所以取第 target_state 行
            dist_array = env.dist_matrix_[target_state] 
            dist_tensor = torch.tensor(dist_array, dtype=torch.float, device=device)

            # 2. 构造高斯热力场 (Ground Truth)
            # sigma 控制扩散的范围，sigma=1.0 代表一环外衰减很快，sigma=2.0 衰减较慢
            sigma = 1.5 
            soft_gt = torch.exp(-(dist_tensor ** 2) / (2 * sigma ** 2))

            # 3. 把 soft_gt 喂给流损失 (MSE)
            # 此时网络会受到非常平滑的引导，而不是突兀的悬崖惩罚
            loss_f = ((latest_probs - soft_gt) ** 2 * (1 - visited_mask)).mean()

            loss_v = (latest_probs ** 2 * visited_mask).mean() # 惩罚在已访问区域预测高概率
            loss_s = (latest_probs[edge_index[0]] - latest_probs[edge_index[1]]).pow(2).mean() # 空间平滑

            total_loss = (LAMBDA_VISITED*loss_v + LAMBDA_FLOW*loss_f + 
                          LAMBDA_SPATIAL*loss_s )
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # 基础的梯度裁剪防爆
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
            optimizer.step()
            
            # 更新概率历史 (detach 以防计算图跨时间步无端累积)
            prob_history.append(latest_probs.detach())
            
            # 仿真推演：纯粹无偏见的随机游走
            actions = [random.choice(graph_[p]) for p in robot_poses]
            next_p, _, _, next_t, done = env.step(actions)
            robot_poses = [np.argmax(p) for p in next_p]
            
            history.append(robot_poses)
            target_state = next_t
            for p in robot_poses: visited_mask[p] = 1.0
            
            total_ep_loss += total_loss.item()
            if done: break
            
        window_loss += (total_ep_loss / (t + 1))
            
        # --- 日志与保存 ---
        if i_ep % 100 == 0:
            avg_loss = window_loss / 100.0
            print(f"Ep {i_ep:05d} | Map: {ENV_NAME:<8} | Robots: {current_robot_num} | Loss: {avg_loss:.5f}")
            writer.add_scalar('pretrain/loss_avg100', avg_loss, i_ep)

        if i_ep % 1000 == 0:
            torch.save(model.state_dict(), SAVE_PATH)

    writer.close()

if __name__ == '__main__':
    main()