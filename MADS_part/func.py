# -*- coding: utf-8 -*-
import numpy as np
from random import choice
from collections import deque

class env_multi_robot:
    def __init__(
            self,
            graph,
            trans_prob_matrix,
            stateSpace,
            robotState,
            targetState,
            robotNum,
    ):
        self.reset_robot_state_ = robotState
        self.reset_target_state_ = targetState
        self.robot_num_ = robotNum
        self.graph_ = graph
        self.trans_prob_matrix = trans_prob_matrix
        self.state_space_ = stateSpace
        self.robot_states_ = [robotState for i in range(robotNum)]
        self.target_state_ = targetState
        self.done_ = False
        self.eta_fp_ = 0.0
        self.eta_fn_ = 0.0

        # === 新增：为每个机器人维护最近位置历史（用于徘徊检测）===
        self.robot_histories_ = [
            deque([robotState], maxlen=5) for _ in range(robotNum)
        ]
        # =====================================================

        # === 内部计算距离矩阵（用于位移判断）===
        self.dist_matrix_ = self._compute_dist_matrix()

    def _compute_dist_matrix(self):
        """计算全图最短路径距离矩阵"""
        N = self.state_space_
        dist = np.full((N, N), np.inf)
        for s in range(N):
            dist[s][s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in self.graph_.get(u, []):
                    if dist[s][v] == np.inf:
                        dist[s][v] = dist[s][u] + 1
                        q.append(v)
        # 将不可达节点距离设为一个大数（实际 museum 图是连通的）
        dist[dist == np.inf] = N
        return dist
    def compute_dist_matrix(graph, N):
        dist = np.full((N, N), np.inf)
        for s in range(N):
            dist[s][s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in graph[u]:
                    if dist[s][v] == np.inf:
                        dist[s][v] = dist[s][u] + 1
                        q.append(v)
        return dist
    def is_loitering(self, robot_idx, window=8, max_unique=3):
        
        hist_deque = self.robot_histories_[robot_idx]
        if len(hist_deque) < window:
            return False
    
       
        hist_list = list(hist_deque)
        recent_hist = hist_list[-window:]  
    
        unique_nodes = len(set(recent_hist))
    
        return unique_nodes <= max_unique

    def hmm_simulator(self):
        hmm_state = np.zeros(self.state_space_, dtype=float, order='C')
        for i in range(self.state_space_):
            if i == self.target_state_:
                hmm_state[i] = 0.5
            else:
                hmm_state[i] = 0.5 / (self.state_space_ - 1)
        return hmm_state

    def update_env(self, action):

        rewards = np.zeros(self.robot_num_, dtype=float)

        # Step 1: 执行所有动作，更新机器人位置
        for i in range(self.robot_num_):
            self.robot_states_[i] = action[i]
            self.robot_histories_[i].append(self.robot_states_[i])
            # else: 非法动作（policy 已 mask，通常不会发生，保留默认 reward=0）

        # Step 2: 检查是否有任意机器人成功找到目标（全局成功）
        found_by_any = any(
            self.robot_states_[i] == self.target_state_ for i in range(self.robot_num_)
        )

        # Step 3: 为每个机器人计算奖励
        if found_by_any:
            # 所有机器人获得基于距离的奖励：越近越高
            for i in range(self.robot_num_):
                dist = self.dist_matrix_[self.robot_states_[i], self.target_state_]
                rewards[i] = 10/(1+dist)  # dist=0 → 20.0, dist=1 → 10.0, etc.
        else:
            # 未成功：按个体行为给惩罚
            for i in range(self.robot_num_):
                if self.is_loitering(i):
                    rewards[i] = -3.0
                else:
                    rewards[i] = -0.5

        # Step 4: 更新目标位置（lazy random walk）
        traget_next_state = np.random.choice(
            range(len(self.trans_prob_matrix[self.target_state_])),
            p=self.trans_prob_matrix[self.target_state_]
        )
        if traget_next_state != 0:
            self.target_state_ = traget_next_state

        return rewards

    def generate_robot_obs(self):
        obs_robot = []
        for i in range(self.robot_num_):
            cur = np.zeros(self.state_space_, dtype=float, order='C')
            cur[self.robot_states_[i]] = 1.0
            obs_robot.append(cur)
        return obs_robot

    def update_env_obs(self):
        obs_env = []
        for robot_state in self.robot_states_:
            if robot_state == self.target_state_:
                a = np.random.rand()
                if a < self.eta_fn_:
                    obs_env.append(0)
                else:
                    obs_env.append(1)
            else:
                a = np.random.rand()
                if a < self.eta_fp_:
                    obs_env.append(1)
                else:
                    obs_env.append(0)
        return obs_env

    def step(self, action):
        reward = self.update_env(action)  # 返回新奖励
        obs_robot = self.generate_robot_obs()
        obs_env = self.update_env_obs()
        for i in range(len(obs_env)):
            if obs_env[i] == 1 and self.robot_states_[i] == self.target_state_:
                self.done_ = True
        return obs_robot, obs_env, reward, self.target_state_, self.done_

    def reset(self):
        self.done_ = False
        self.robot_states_ = [self.reset_robot_state_ for i in range(self.robot_num_)]
        self.target_state_ = self.reset_target_state_
        
        # === 重置位置历史（关键！）===
        self.robot_histories_ = [
            deque([self.reset_robot_state_], maxlen=5) 
            for _ in range(self.robot_num_)
        ]
        # ===========================

        true_hmm = self.hmm_simulator()
        obs_robot = self.generate_robot_obs()
        obs_env = self.update_env_obs()
        return obs_robot, obs_env, self.target_state_