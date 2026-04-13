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
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

# ==========================================
# 1. 环境挂载与模块导入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 确保路径匹配你的工程结构
from MADS_part.func import env_multi_robot
from map.env import graphdict_setup
from MADS_part.mads_network import MultiAgentManager # 引入主控 Manager

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==============================================================================
# 2. 基础配置与 MLOps 规范命名
# ==============================================================================
log_dir_base = "./maddpg"
model_save_dir = "./mads_policy"
os.makedirs(log_dir_base, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 环境与超参数定义
episode_limit = 20000
# [架构师修正] 设为 'random' 以匹配 env.py 的图生成逻辑，避免 NameError
env_name = "random" 
robot_num = 4
STATE_SPACE = 71 
gamma = 0.95
limit = 200     
seq_len = 30

# 动态 k 值退火参数
k0 = 1          
k_alpha = 0.5   

# 动态生成高辨识度的 Run ID
current_time = datetime.now().strftime("%Y-%m-%d")
run_name = f"{current_time}_{env_name}_{robot_num}agents"

# 模型保存与日志路径
SAVE_MODEL_PATH = os.path.join(model_save_dir, f"{run_name}.pkl")
tb_log_dir = os.path.join(log_dir_base, run_name)

CONTINUE_TRAINING = False  
LOAD_MODEL_PATH = os.path.join(model_save_dir, "your_history_model_name.pkl") 

# ==============================================================================
# 3. 初始化网络与图结构
# ==============================================================================
manager = MultiAgentManager(
    num_agents=robot_num,
    stgnn_in_dim=2, # [架构师断言] 必须为 2：自身位置特征 + 雷达概率特征
    stgnn_out_dim=32,
    num_nodes=STATE_SPACE,
    hidden_dim=128,
    seq_len=seq_len,
    device=device
).to(device)

optimizer = optim.Adam(manager.parameters(), lr=0.001)

start_episode = 0
if CONTINUE_TRAINING and os.path.exists(LOAD_MODEL_PATH):
    print(f"Loading history model from {LOAD_MODEL_PATH}...")
    checkpoint = torch.load(LOAD_MODEL_PATH, map_location=device)
    manager.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['episode'] + 1
    print(f"Resuming from episode {start_episode}")
else:
    print("Starting training from scratch with RADAR Early-Fusion Head.")

writer = SummaryWriter(tb_log_dir)

# 初始化图结构
graph_ = graphdict_setup(env_name, STATE_SPACE)

def build_edge_index(graph):
    edges = [[u, v] for u in graph for v in graph[u]]
    return torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

edge_index = build_edge_index(graph_)

# 预渲染全局动作掩码
global_adj_mask = torch.zeros((STATE_SPACE, STATE_SPACE), device=device)
for i in range(STATE_SPACE):
    if i in graph_:
        global_adj_mask[i, graph_[i]] = 1.0

def compute_dist_matrix(graph, N):
    dist = np.full((N, N), np.inf)
    for s in range(N):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in graph.get(u, []):
                if dist[s][v] == np.inf:
                    dist[s][v] = dist[s][u] + 1
                    q.append(v)
    return dist

dist_matrix = compute_dist_matrix(graph_, STATE_SPACE)
finite_dists = dist_matrix[np.isfinite(dist_matrix)]
max_dist_val = int(np.nanmax(finite_dists)) if len(finite_dists) > 0 else 1

# ==============================================================================
# 4. 核心调度与特征计算
# ==============================================================================
def get_robot_idx(one_hot_obs):
    return int(np.argmax(one_hot_obs))

def build_node_feature_seq(history_poses):
    poses_tensor = torch.tensor(list(history_poses), dtype=torch.long, device=device)
    actual_len = poses_tensor.shape[0] 
    
    # 构建占用栅格特征 [seq_len, STATE_SPACE, 1]
    feat = torch.zeros((actual_len, STATE_SPACE, 1), device=device)
    feat.scatter_(1, poses_tensor.unsqueeze(-1), 1.0)
    
    return feat

def select_action(seq_features, seq_target_probs, robot_poses, current_k_val):
    manager.train()
    
    # [架构师修正] 对齐 mads_network.py 的签名：传入 seq_target_probs
    # 接收：动作logits，指数化还原后的概率，底层对数概率(用于算Loss)
    action_logits, latest_probs, target_log_probs = manager(
        current_k_val, seq_features, seq_target_probs, edge_index, robot_poses
    )
    
    with torch.no_grad():
        mask = global_adj_mask[robot_poses] 
        
    masked_logits = action_logits + (mask - 1) * 1e9
    probs = F.softmax(masked_logits, dim=1)
    dist = Categorical(probs)
    actions = dist.sample()
    
    # 返回分离后的 latest_probs 给外部放入时序窗口
    return actions.cpu().numpy(), dist.log_prob(actions), target_log_probs, latest_probs.detach()

def finish_episode(log_probs_list, rewards_list, target_loss_list):
    R = 0
    returns = []
    with torch.no_grad():
        for r in reversed(rewards_list):
            R = r + gamma * R
            returns.insert(0, R)
        returns_tensor = torch.stack(returns).to(device)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            
    log_probs_tensor = torch.stack(log_probs_list) 
    
    # 1. 策略梯度损失 (RL Loss)
    rl_loss = -(log_probs_tensor * returns_tensor).sum()
    
    # 2. 雷达预测辅助损失 (Prediction Loss)
    pred_loss = torch.stack(target_loss_list).mean()
    
    # 3. 混合梯度反传
    total_loss = rl_loss + 0.5 * pred_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(manager.parameters(), max_norm=0.5)
    optimizer.step()
    
    return total_loss.item(), pred_loss.item()

def manualInitEmm(graph):
    lazy_gumma = np.zeros((STATE_SPACE, STATE_SPACE), dtype=float)
    for i in range(len(lazy_gumma)):
        if i in graph:
            size = len(graph[i])
            for j in range(len(lazy_gumma)):
                if i == j: lazy_gumma[i][j] = 0.9
                elif j in graph[i]: lazy_gumma[i][j] = 0.1 / (size - 1)
                else: lazy_gumma[i][j] = 0.
        else:
            lazy_gumma[i][i] = 1.0
    return lazy_gumma

# ==============================================================================
# 5. 主训练循环 (Early Fusion 队列管理)
# ==============================================================================
def main():
    transition_prob = manualInitEmm(graph_)
    dequelen = 30
    recent_steps = deque(maxlen=dequelen)
    recent_explored = deque(maxlen=dequelen)
    recent_rewards = deque(maxlen=dequelen)
    recent_success = deque(maxlen=dequelen)
    recent_rl_loss = deque(maxlen=dequelen)
    recent_pred_loss = deque(maxlen=dequelen) 
    
    for i_episode in range(start_episode, episode_limit):
        progress = i_episode / episode_limit
        min_dist_threshold = max(3, int(progress * max_dist_val))
        
        valid_pairs = [(s, t) for s in range(STATE_SPACE) for t in range(STATE_SPACE) 
                       if s != t and dist_matrix[s][t] >= min_dist_threshold]
        
        initRobotState, initTargetState = random.choice(valid_pairs) if valid_pairs else (0, 10)
        
        env = env_multi_robot(graph_, transition_prob, STATE_SPACE, initRobotState, initTargetState, robot_num)
        one_hot_pose, _, _ = env.reset()
        
        robot_poses = [get_robot_idx(p) for p in one_hot_pose]
        history_poses = deque([robot_poses], maxlen=seq_len)
        visited_nodes = set(robot_poses)
        
        # [架构师修正] 构建与历史位置同步的目标概率记忆队列 (Initial uniform distribution)
        initial_probs = torch.ones(STATE_SPACE, 1, device=device) / STATE_SPACE
        history_target_probs = deque([initial_probs], maxlen=seq_len)
        
        log_probs_list, rewards_list, target_loss_list = [], [], []
        steps = 0
        current_k = k0 
        
        for t in range(limit):
            steps += 1
            
            explored_area = len(visited_nodes)
            unexplored_area = max(STATE_SPACE - explored_area, 1) 
            current_k = k0 * np.exp(-k_alpha * (explored_area / unexplored_area))
            
            # --- 序列化双模态特征 ---
            seq_feat = build_node_feature_seq(history_poses)
            seq_target_probs_tensor = torch.stack(list(history_target_probs)) # [seq_len, N, 1]
            
            # --- 核心前向传递 ---
            actions, log_p, target_log_probs, latest_probs = select_action(
                seq_feat, seq_target_probs_tensor, robot_poses, current_k
            )
            
            next_one_hot, _, rewards, next_target, done = env.step(actions)
            
            # [架构师修正] 网络侧使用了 F.log_softmax 限制下界，这里绝对不能用 CrossEntropy！
            # 必须使用负对数似然(NLLLoss) 接收网络直出的 log_probs 算交叉熵
            true_target_tensor = torch.tensor([next_target], dtype=torch.long, device=device)
            step_pred_loss = F.nll_loss(target_log_probs.unsqueeze(0), true_target_tensor)
            target_loss_list.append(step_pred_loss)
            
            # --- 更新环境与双重历史队列 ---
            robot_poses = [get_robot_idx(p) for p in next_one_hot]
            history_poses.append(robot_poses)
            history_target_probs.append(latest_probs.unsqueeze(-1)) # 压入预测出的最新概率图
            visited_nodes.update(robot_poses)
            
            log_probs_list.append(log_p)
            rewards_list.append(torch.tensor(rewards, device=device))
            
            if done: 
                break
                
        recent_success.append(1 if done else 0) 
        recent_steps.append(steps)
        recent_explored.append(len(visited_nodes))
        
        if len(log_probs_list) > 0:
            total_loss_val, pred_loss_val = finish_episode(log_probs_list, rewards_list, target_loss_list)
            total_r = sum([r.sum().item() for r in rewards_list])
            
            recent_rewards.append(total_r)
            recent_rl_loss.append(total_loss_val)
            recent_pred_loss.append(pred_loss_val)
            
            writer.add_scalar('train/RL_Total_Loss', total_loss_val, i_episode)
            writer.add_scalar('train/Radar_Pred_Loss', pred_loss_val, i_episode)
            writer.add_scalar('train/reward', total_r, i_episode)
            writer.add_scalar('train/steps', steps, i_episode)
            writer.add_scalar('train/dynamic_k', current_k, i_episode)
            
            if i_episode % 10 == 0:
                avg_steps = sum(recent_steps) / len(recent_steps)
                avg_exp = sum(recent_explored) / len(recent_explored)
                avg_rew = sum(recent_rewards) / len(recent_rewards)
                avg_rl_loss = sum(recent_rl_loss) / len(recent_rl_loss)
                avg_pred_loss = sum(recent_pred_loss) / len(recent_pred_loss)
                success_rate = sum(recent_success) / len(recent_success) * 100
                
                print(f"Ep {i_episode:05d} | Win: {success_rate:3.0f}% | "
                      f"Rwd: {avg_rew:5.1f} | Steps: {avg_steps:5.1f} | "
                      f"Exp: {avg_exp:4.1f}/{STATE_SPACE} | "
                      f"Loss: {avg_rl_loss:5.2f} | PredLoss: {avg_pred_loss:5.2f}")
            
            if i_episode > 0 and i_episode % 100 == 0:
                torch.save({
                    'episode': i_episode,
                    'model_state_dict': manager.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, SAVE_MODEL_PATH)
                print(f"--> [Model Saved] Updated Policy with Radar Head to {SAVE_MODEL_PATH}")

    writer.close()

if __name__ == '__main__':
    main()