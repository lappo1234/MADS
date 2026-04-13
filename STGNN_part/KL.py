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
from MADS_part.mads_network import STGNN

# 🌟 核心修改：直接从我们刚刚改好的 env 模块中，导入解析完毕的 JSON 数据！
from map.env import graphdict_setup, top_100_maps_edges

# ==========================================
# 2. 自动化配置与参数
# ==========================================
SEQ_LEN = 30
PRETRAIN_EPISODES = 30000  # 锁定 50,000 回合
ROBOT_NUM_MIN = 1
ROBOT_NUM_MAX = 5  

# 核心混合损失权重 (针对 KL 散度重新平衡)
LAMBDA_VISITED = 5.0   # 严厉打击走过的回头路 (线性压榨)
LAMBDA_FLOW = 1.0      # 目标牵引 (KL散度，自带极强梯度，权重 1.0 即可)
LAMBDA_SPATIAL = 0.1   # 空间平滑 (适当降低，避免干扰 KL 散度的峰值塑造)

today = datetime.now().strftime("%Y-%m-%d")
run_name = f"{today}_MultiMap_Curriculum_KL_STGNN"

MODEL_SAVE_DIR = os.path.join(current_dir, "stgnn", '多地图策略')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{run_name}.pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=os.path.join(root_dir, "STGNN_part", "stgnn_log", 'multimap', run_name))

# ==========================================
# 3. 核心功能组件
# ==========================================

def build_node_feature_seq(history, current_state_space):
    """动态尺寸的节点特征构造"""
    feat = torch.zeros((SEQ_LEN, current_state_space, 1), device=device)
    poses_list = list(history)
    
    if len(poses_list) < SEQ_LEN:
        poses_list = [poses_list[0]] * (SEQ_LEN - len(poses_list)) + poses_list
    
    for t_idx, current_step_poses in enumerate(poses_list):
        p_tensor = torch.tensor(current_step_poses, device=device).long()
        feat[t_idx, p_tensor, 0] = 1.0
    return feat

def build_prob_feature_seq(prob_history):
    """概率特征构造 (长度由模型输出决定，自动适配)"""
    probs = list(prob_history)
    if len(probs) < SEQ_LEN:
        probs = [probs[0]] * (SEQ_LEN - len(probs)) + probs
    return torch.stack(probs).unsqueeze(-1)

def manualInitEmm(graph, current_state_space, p_stay=0.5):
    """动态尺寸的状态转移矩阵"""
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

# ==========================================
# 4. 主训练循环 (课程学习机制)
# ==========================================

def main():
    print(f"📡 启动多地图课程学习 (Curriculum Learning KL): {run_name}")
    print(f"🤖 机器人规模范围: {ROBOT_NUM_MIN} - {ROBOT_NUM_MAX}")
    
    # --- 🌟 极简课程构建器 ---
    map_curriculum = []
    
    # 防护：如果 JSON 未能成功加载，中断程序并给出提示
    if not top_100_maps_edges:
        raise ValueError("❌ 无法获取地图数据，请确保 map 目录下存在 top_100_edges.json 文件！")
        
    for m_name, edges in top_100_maps_edges.items():
        if not edges: continue
        # 解析 JSON 后的边通常是 list 格式 (如 [0, 93])，提取最大节点 ID 以推断空间大小
        num_nodes = max([max(edge[0], edge[1]) for edge in edges]) + 1
        map_curriculum.append((m_name, num_nodes))
        
    # JSON 数据的顺序就是你当时提取的由难到简，反转它实现由简到难！
    map_curriculum.reverse()
    print(f"🗺️ 成功挂载 {len(map_curriculum)} 张真实建筑地图。起始难度: {map_curriculum[0][1]} 节点, 终极难度: {map_curriculum[-1][1]} 节点")

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
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    window_loss = 0.0

    current_env_name = ""
    current_state_space = 0
    graph_ = None
    edge_index = None

    for i_ep in range(1, PRETRAIN_EPISODES + 1):
        if (i_ep - 1) % 100 == 0:
            window_loss = 0.0

        # ==========================================
        # 🔄 课程学习地图切换逻辑 (Curriculum Switcher)
        # ==========================================
        if i_ep <= 49000:
            # 1-49000 轮：平均分配给挂载的地图（防呆设计：自适应地图数量）
            map_idx = min((i_ep - 1) // (49000 // len(map_curriculum)), len(map_curriculum) - 1)
            target_env_name, target_state_space = map_curriculum[map_idx]
        else:
            # 49001-50000 轮：终极泛化期
            if i_ep == 49001:
                print(f"\n🔥 阶段 2: 终极泛化期 (49001-50000) -> 启动真实地图随机抽样验证!")
            
            # 为了防止底层频繁重建消耗算力，每 10 个回合随机抽样切换一次
            if (i_ep - 1) % 10 == 0:
                target_env_name, target_state_space = random.choice(map_curriculum)
            else:
                target_env_name, target_state_space = current_env_name, current_state_space

        # 触发底层图结构更换
        if current_env_name != target_env_name:
            current_env_name = target_env_name
            current_state_space = target_state_space
            graph_ = graphdict_setup(current_env_name, current_state_space)
            
            # 构建 PyTorch Edge Index
            edge_index = torch.tensor([[u, v] for u in graph_ for v in graph_[u]], dtype=torch.long).t().contiguous().to(device)
            
            # 🛡️ 架构师级防护：强制注入自环
            self_loops = torch.arange(current_state_space, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
            if edge_index.numel() > 0:
                edge_index = torch.cat([edge_index, self_loops], dim=1)
            else:
                edge_index = self_loops
                
            if i_ep <= 49000:
                print(f"🌍 [课程升级] Ep {i_ep}: 切换至 {current_env_name}, 尺寸: {current_state_space}")

        # --- 随机化当前回合参数 ---
        current_robot_num = random.randint(ROBOT_NUM_MIN, ROBOT_NUM_MAX)
        p_stay = random.uniform(0.3, 0.95)
        transition_prob = manualInitEmm(graph_, current_state_space, p_stay=p_stay)
        
        # 安全抽取生成点：直接从建好的真实图节点里抽签
        all_nodes = list(graph_.keys())
        spawn_points = random.sample(all_nodes, current_robot_num + 1)
        rand_agent_poses = spawn_points[:current_robot_num]
        rand_target_pos = spawn_points[current_robot_num]

        # 初始化环境
        env = env_multi_robot(graph_, transition_prob, current_state_space, 
                               rand_agent_poses, rand_target_pos, current_robot_num)
        
        one_hot, _, target_state = env.reset()
        robot_poses = [np.argmax(p) for p in one_hot] 
        
        # 初始化历史
        history = deque([robot_poses], maxlen=SEQ_LEN)
        
        init_prob = torch.ones(current_state_space, device=device)
        for p in robot_poses:
            init_prob[p] = 0.0
        init_prob = init_prob / init_prob.sum() 
        prob_history = deque([init_prob], maxlen=SEQ_LEN)

        total_ep_loss = 0
        
        for t in range(120):
            node_feat = build_node_feature_seq(history, current_state_space)
            prob_feat = build_prob_feature_seq(prob_history)

            # 前向传播
            _, latest_probs, raw_log_probs = model(node_feat, prob_feat, edge_index)
            
            # --- 🌟 动态记忆掩码 ---
            visited_mask = torch.zeros(current_state_space, device=device)
            for past_poses in history: 
                for p in past_poses:
                    visited_mask[p] = 1.0

            # --- 🌟 动态高斯软标签 ---
            dist_array = env.dist_matrix_[target_state] 
            dist_tensor = torch.tensor(dist_array, dtype=torch.float, device=device)
            
            # 距离截断防护 (图可能不全连通)
            dist_tensor = torch.clamp(dist_tensor, max=1e4)
            sigma = 1.5 
            soft_gt = torch.exp(-(dist_tensor ** 2) / (2 * sigma ** 2))
            
            # 剔除走过区域的气味，防止 KL 散度与 visited_mask 惩罚互斥
            soft_gt = soft_gt * (1 - visited_mask) + 1e-8
            
            # 防止软标签全军覆没导致梯度消失或数值异常
            gt_sum = soft_gt.sum()
            if gt_sum < 1e-6:
                soft_gt = torch.ones(current_state_space, device=device) / current_state_space
            else:
                soft_gt = soft_gt / gt_sum

            # --- 🌟 安全混合损失 ---
            # 1. 目标牵引: KL 散度
            loss_f = F.kl_div(raw_log_probs, soft_gt, reduction='sum') / current_state_space
            # 2. 排雷清理: 线性压榨
            loss_v = (latest_probs * visited_mask).sum() / (visited_mask.sum() + 1e-8)
            # 3. 空间平滑
            loss_s = (latest_probs[edge_index[0]] - latest_probs[edge_index[1]]).pow(2).mean() 

            total_loss = LAMBDA_VISITED*loss_v + LAMBDA_FLOW*loss_f + LAMBDA_SPATIAL*loss_s
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
            optimizer.step()
            
            prob_history.append(latest_probs.detach())
            
            # 仿真推演
            actions = [random.choice(graph_[p]) for p in robot_poses]
            next_p, _, _, next_t, done = env.step(actions)
            robot_poses = [np.argmax(p) for p in next_p]
            
            history.append(robot_poses)
            target_state = next_t
            
            total_ep_loss += total_loss.item()
            if done: break
            
        window_loss += (total_ep_loss / (t + 1))
            
        # --- 日志与保存 ---
        if i_ep % 100 == 0:
            avg_loss = window_loss / 100.0
            print(f"Ep {i_ep:05d} | MapNodes: {current_state_space:<3} | Robots: {current_robot_num} | Loss: {avg_loss:.5f}")
            writer.add_scalar('pretrain/loss_avg100', avg_loss, i_ep)
            writer.add_scalar('pretrain/map_size', current_state_space, i_ep)

        if i_ep % 1000 == 0:
            torch.save(model.state_dict(), SAVE_PATH)

    writer.close()

if __name__ == '__main__':
    main()