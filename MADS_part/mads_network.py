# -*- coding: utf-8-sig -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv 

class STGNN(nn.Module):
    # 注意：在实例化时，in_dim 必须传入 2 (自身位置特征 + 雷达概率特征)
    def __init__(self, in_dim=2, gat_hid=32, gat_heads=4, attn_hid=64, num_attn_layers=2, 
                 num_attn_heads=4, out_dim=None, seq_len=10, device=torch.device('cuda')):
        super().__init__()
        self.seq_len = seq_len
        self.device = device
        
        # O(1) 并发图注意力层
        self.gat1 = GATConv(in_channels=in_dim, out_channels=gat_hid, heads=gat_heads, concat=True)
        self.gat2 = GATConv(in_channels=gat_hid * gat_heads, out_channels=gat_hid, heads=1, concat=False)
        self.gat_out_dim = gat_hid
        
        # 时序 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.gat_out_dim, nhead=num_attn_heads, dim_feedforward=attn_hid,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.temporal_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)
        
        self.out_dim = out_dim if out_dim is not None else self.gat_out_dim
        self.output_proj = nn.Linear(self.gat_out_dim, self.out_dim) if self.out_dim != self.gat_out_dim else nn.Identity()

        # ==========================================================
        # 目标概率预测头
        # 此时 gat_out_dim 已经包含了丰富的概率历史与轨迹交互信息
        # ==========================================================
        self.target_predictor = nn.Sequential(
            nn.Linear(self.gat_out_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, seq_node_features, seq_target_probs, edge_index):

        seq_len, N, _ = seq_node_features.shape

        # 早期特征融合 (Early Fusion)
        # 拼接后的特征维度为 [seq_len, N, 2]
        combined_features = torch.cat([seq_node_features, seq_target_probs], dim=-1)

        # 展开为平面节点序列进行 O(1) 图卷积
        flat_node_features = combined_features.view(seq_len * N, -1)
        offsets = (torch.arange(seq_len, device=self.device) * N).view(1, 1, seq_len)
        batched_edge_index = (edge_index.unsqueeze(2) + offsets).view(2, -1)

        h = F.elu(self.gat1(flat_node_features, batched_edge_index))
        h = F.elu(self.gat2(h, batched_edge_index))
        gat_stack = h.view(seq_len, N, -1).transpose(0, 1)

        # 提取时序特征
        attended_features = self.temporal_attention(gat_stack)
        final_node_embeddings = attended_features[:, -1, :] # [N, gat_out_dim]
        
        # 预测最新时刻的概率
        target_logits = self.target_predictor(final_node_embeddings).squeeze(-1) # [N]
        
        # 使用原生 log_softmax 免疫 NaN，并截断过度自信引发的 -inf
        log_probs = F.log_softmax(target_logits, dim=0) 
        log_probs = torch.clamp(log_probs, min=-20.0)
        
        # 还原概率分布用于 Policy 输入
        latest_probs = torch.exp(log_probs) 
        
        return self.output_proj(final_node_embeddings), latest_probs, log_probs


class MADSPolicyGNN(nn.Module):
    def __init__(self, stgnn_out_dim, hidden_dim=128, device=torch.device('cuda')):
        """
        拓扑感知型动态尺寸适配器 (Topology-Aware Dynamic Adapter)
        融合了 GAT (Graph Attention) 与全局状态广播。
        """
        super().__init__()
        self.device = device
        
        # ==========================================================
        # 1. 拓扑特征聚合层 (Message Passing)
        # 输入：STGNN 节点特征 (stgnn_out_dim) + 缩放后的目标概率 (1)
        # ==========================================================
        gnn_in_dim = stgnn_out_dim + 1
        
        # 使用 2 层 GAT。heads=2 增加多头注意力，提取不同的寻路维度
        self.gat1 = GATConv(in_channels=gnn_in_dim, out_channels=hidden_dim // 2, heads=2, concat=True)
        self.gat2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=1, concat=False)
        
        # ==========================================================
        # 2. 节点级视野评估 (Node-Centric Scoring)
        # 输入：拓扑感知后的 Agent 特征(hidden_dim) + 候选节点特征(hidden_dim)
        # ==========================================================
        mlp_in_dim = hidden_dim * 2
        
        self.fc1 = nn.Linear(mlp_in_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1) # 对单个节点输出一个 Logit 评分

    def forward(self, shared_node_emb, current_node_idx, target_probs, edge_index):
        """
        前向传播
        ⚠️ 注意：这里必须传入当前地图的 edge_index [2, E]
        """
        N = shared_node_emb.size(0)
        
        # ---------------------------------------------------------
        # Step 1: 解决“量级碾压” —— 概率特征对数化放缩
        # ---------------------------------------------------------
        # 将 [N, 1] 的概率加上微小常量防止 log(0)，放大微弱的高概率区信号
        scaled_probs = torch.log(target_probs.view(N, 1) + 1e-8)
        
        # 拼接基础节点特征: [N, stgnn_out_dim + 1]
        raw_node_features = torch.cat([shared_node_emb, scaled_probs], dim=-1)
        
        # ---------------------------------------------------------
        # Step 2: 拓扑信息弥漫 (Graph Message Passing)
        # ---------------------------------------------------------
        # 让节点顺着边“看”到邻居的特征和目标概率
        x = F.elu(self.gat1(raw_node_features, edge_index))
        topo_aware_embs = F.elu(self.gat2(x, edge_index)) # 输出形状: [N, hidden_dim]
        
        # ---------------------------------------------------------
        # Step 3: 全局上下文广播 (Broadcasting)
        # ---------------------------------------------------------
        # 提取当前 Agent 所在位置的“拓扑感知特征”
        agent_emb = topo_aware_embs[current_node_idx] # [hidden_dim]
        # 广播给所有的 N 个候选节点
        agent_emb_expanded = agent_emb.unsqueeze(0).expand(N, -1) # [N, hidden_dim]
        
        # 拼接信息源: [N, 2 * hidden_dim]
        policy_input = torch.cat([agent_emb_expanded, topo_aware_embs], dim=-1)
        
        # ---------------------------------------------------------
        # Step 4: 并发打分网络
        # ---------------------------------------------------------
        x = F.relu(self.fc1(policy_input))
        x = F.relu(self.fc2(x))
        node_logits = self.fc3(x) # 输出形状 [N, 1]
        
        # 转置为行向量 [1, N]，与掩码层和环境无缝对接
        return node_logits.view(1, N)

class MultiAgentManager(nn.Module):
    def __init__(self, num_agents, stgnn_in_dim, stgnn_out_dim, num_nodes, 
                 hidden_dim=128, seq_len=10, freeze_stgnn=False, device=torch.device('cuda')):
        super().__init__()
        self.num_agents = num_agents
        self.device = device
        self.num_nodes = num_nodes # 仅保留用于外部调用者的兼容性
        self.freeze_stgnn = freeze_stgnn

        # STGNN 强制 in_dim 为 2
        self.shared_stgnn = STGNN(in_dim=2, out_dim=stgnn_out_dim, seq_len=seq_len, device=device)
        
        # 🌟 [架构师改造 1] 挂载全新的拓扑感知 GNN 策略网络 (MADSPolicyGNN)
        self.homogeneous_policy = MADSPolicyGNN(stgnn_out_dim, hidden_dim, device)
        self.hetero_policies = nn.ModuleList([
            MADSPolicyGNN(stgnn_out_dim, hidden_dim, device) for _ in range(num_agents)
        ])

        if freeze_stgnn:
            for param in self.shared_stgnn.parameters():
                param.requires_grad = False
            print("--> [INFO] 🛡️ STGNN Radar is FROZEN. Gradient Isolation Active.")

    def forward(self, k_val, seq_node_features, seq_target_probs, edge_index, agent_current_nodes):
        # 1. 呼叫雷达，获取双重时序特征与目标概率热力图
        shared_node_emb, latest_probs, target_logits = self.shared_stgnn(
            seq_node_features, seq_target_probs, edge_index
        )

        # 🌟 [架构师改造 2] 梯度物理隔断 (Gradient Detachment)
        # 即使 params 设置了 requires_grad=False，在计算图中打断张量流是更安全、更省显存的工业级做法
        if self.freeze_stgnn:
            shared_node_emb = shared_node_emb.detach()
            latest_probs = latest_probs.detach()

        all_action_logits = []
        for i in range(self.num_agents):
            idx = agent_current_nodes[i]
            
            # 🌟 [架构师改造 3] 注入 edge_index，彻底激活 GAT 层的路网感知能力
            homo_logits = self.homogeneous_policy(shared_node_emb, idx, latest_probs, edge_index)
            hetero_logits = self.hetero_policies[i](shared_node_emb, idx, latest_probs, edge_index)
            
            # 异构探索度融合 (K_val 动态退火)
            final_logits = homo_logits + k_val * hetero_logits
            all_action_logits.append(final_logits)

        # 返回 latest_probs 便于主循环将其推入下一个时间步的历史队列中
        return torch.cat(all_action_logits, dim=0), latest_probs, target_logits
