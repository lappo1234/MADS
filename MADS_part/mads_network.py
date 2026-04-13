# -*- coding: utf-8-sig -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv 

class STGNN(nn.Module):
    # 注意：在实例化时，in_dim 应该传入 2
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

        # [核心优化] 早期特征融合 (Early Fusion)
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
        
        #        【架构师神级修复】：必须使用原生 log_softmax，初步免疫 NaN！
        log_probs = F.log_softmax(target_logits, dim=0) 
        
        # ? 【最后一道护城河】：限制对数概率下限，彻底斩断过度自信引发的 -inf！
        log_probs = torch.clamp(log_probs, min=-20.0)
        
        # 外部依然需要正常的概率做掩码计算，所以 exp 还原一下
        latest_probs = torch.exp(log_probs) 
        
        # 注意返回值：把 logits 替换成了 log_probs，传给外部算 Loss
        return self.output_proj(final_node_embeddings), latest_probs, log_probs


class MADSPolicy(nn.Module):
    def __init__(self, stgnn_out_dim, num_nodes, hidden_dim=128, device=torch.device('cuda')):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device
        
        # ==========================================================
        # [架构师核心科技 2]：决策层视野扩充
        # 输入：STGNN节点特征 + 机器人自身位置One-hot + 全图目标概率雷达
        # ==========================================================
        self.fc1 = nn.Linear(stgnn_out_dim + num_nodes + num_nodes, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_nodes)

    def forward(self, shared_node_emb, current_node_idx, target_probs):
        # current_node_idx: scalar int
        agent_emb = shared_node_emb[current_node_idx].unsqueeze(0) # [1, emb]
        
        pos_vec = torch.zeros(1, self.num_nodes, device=self.device)
        pos_vec[0, current_node_idx] = 1.0
        
        # 拼接三大信息源！
        policy_input = torch.cat([agent_emb, pos_vec, target_probs], dim=1)
        x = F.relu(self.fc1(policy_input))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MultiAgentManager(nn.Module):
    def __init__(self, num_agents, stgnn_in_dim, stgnn_out_dim, num_nodes, 
                 hidden_dim=128, seq_len=10, freeze_stgnn=False, device=torch.device('cuda')):
        super().__init__()
        self.num_agents = num_agents
        self.device = device
        self.num_nodes = num_nodes

        # 强制 STGNN 的 in_dim 为 2 (因为我们拼接了 1维位置 + 1维概率)
        self.shared_stgnn = STGNN(in_dim=2, out_dim=stgnn_out_dim, seq_len=seq_len, device=device)
        self.homogeneous_policy = MADSPolicy(stgnn_out_dim, num_nodes, hidden_dim, device)
        self.hetero_policies = nn.ModuleList([
            MADSPolicy(stgnn_out_dim, num_nodes, hidden_dim, device) for _ in range(num_agents)
        ])

        if freeze_stgnn:
            for param in self.shared_stgnn.parameters():
                param.requires_grad = False
            print("?? [INFO] STGNN Radar is FROZEN. Only training RL Policies.")

    def forward(self, k_val, seq_node_features, seq_target_probs, edge_index, agent_current_nodes):
        # 接收并传递双重时序特征序列
        shared_node_emb, latest_probs, target_logits = self.shared_stgnn(
            seq_node_features, seq_target_probs, edge_index
        )
        
        # policy 依然接收最新时刻的雷达图
        target_probs_for_policy = latest_probs.unsqueeze(0)

        all_action_logits = []
        for i in range(self.num_agents):
            idx = agent_current_nodes[i]
            homo_logits = self.homogeneous_policy(shared_node_emb, idx, target_probs_for_policy)
            hetero_logits = self.hetero_policies[i](shared_node_emb, idx, target_probs_for_policy)
            final_logits = homo_logits + k_val * hetero_logits
            all_action_logits.append(final_logits)

        # 返回 latest_probs，便于主循环将其推入下一个时间步的历史队列中
        return torch.cat(all_action_logits, dim=0), latest_probs, target_logits