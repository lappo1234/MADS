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
        
        # 支持列表/数组形式的初始位置
        if isinstance(robotState, (list, tuple, np.ndarray)):
            self.robot_states_ = list(robotState)
        else:
            self.robot_states_ = [robotState for _ in range(robotNum)]
            
        self.target_state_ = targetState
        self.done_ = False
        self.eta_fp_ = 0.0
        self.eta_fn_ = 0.0

        # 🌟 [架构师优化] 1. 引入精确的“连续挂机计数器”和“全局探索记忆”
        self.stay_counters_ = np.zeros(self.robot_num_, dtype=int)
        self.global_visited_ = set(self.robot_states_)

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
        dist[dist == np.inf] = N
        return dist
        
    # (原有的 is_loitering 函数已被更精确的 stay_counters_ 替代，已删除以保持代码整洁)

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

        prev_robot_states = self.robot_states_.copy()
        prev_target_state = int(self.target_state_)

        is_new_node = [False] * self.robot_num_

        # Step 1: 执行动作，更新机器人位置，并检测行为特征
        for i in range(self.robot_num_):
            self.robot_states_[i] = int(action[i]) 
            
            # 🌟 [架构师优化] 2. 精准追踪连续挂机行为
            if self.robot_states_[i] == prev_robot_states[i]:
                self.stay_counters_[i] += 1
            else:
                self.stay_counters_[i] = 0  # 一旦移动，清空挂机计数
                
            # 🌟 [架构师优化] 3. 检测是否踩亮了地图上的新节点（战争迷雾系统）
            if self.robot_states_[i] not in self.global_visited_:
                is_new_node[i] = True
                self.global_visited_.add(self.robot_states_[i])

        # Step 2: 提前推演目标的下一步位置（同步移动）
        target_next_state = int(np.random.choice(
            range(len(self.trans_prob_matrix[prev_target_state])),
            p=self.trans_prob_matrix[prev_target_state]
        ))
        if target_next_state == 0:  
            target_next_state = prev_target_state

        # Step 3: 检查是否有机器人成功找到目标
        found_by_any = False
        for i in range(self.robot_num_):
            curr_robot_state = self.robot_states_[i]
            prev_robot_state = prev_robot_states[i]
            
            meet_at_node = (curr_robot_state == prev_target_state) or (curr_robot_state == target_next_state)
            cross_paths = (prev_robot_state == target_next_state) and (curr_robot_state == prev_target_state)
            
            if meet_at_node or cross_paths:
                found_by_any = True
                break

        # 正式更新全局目标位置
        self.target_state_ = target_next_state

        # Step 4: 为每个机器人计算极具侵略性的 Reward 体系
        for i in range(self.robot_num_):
            if found_by_any:
                rewards[i] = 50.0  # 团队胜利，共享高额赏金
            else:
                # 基础时间税：逼迫他们快点结束战斗
                step_r = -0.5
                
                # 💥 惩罚项：如果连续挂机大于等于 5 步，追加严重扣分！
                if self.stay_counters_[i] >= 5:
                    step_r -= 1.0
                    
                # 🍬 奖励项：如果开辟了新节点，给予正向激励！(抵消时间税还有盈余)
                if is_new_node[i]:
                    step_r += 1.0
                    
                rewards[i] = step_r

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
        reward = self.update_env(action)  
        obs_robot = self.generate_robot_obs()
        obs_env = self.update_env_obs()
        
        # 判断本局是否结束
        for i in range(len(obs_env)):
            if obs_env[i] == 1 and self.robot_states_[i] == self.target_state_:
                self.done_ = True
                
        return obs_robot, obs_env, reward, self.target_state_, self.done_

    def reset(self):
        self.done_ = False
        
        if isinstance(self.reset_robot_state_, (list, tuple, np.ndarray)):
            self.robot_states_ = list(self.reset_robot_state_)
        else:
            self.robot_states_ = [self.reset_robot_state_ for _ in range(self.robot_num_)]
            
        self.target_state_ = self.reset_target_state_
        
        # 🌟 [架构师优化] 4. Episode 重置时，清空探索记忆和挂机计数
        self.stay_counters_ = np.zeros(self.robot_num_, dtype=int)
        self.global_visited_ = set(self.robot_states_)

        true_hmm = self.hmm_simulator()
        obs_robot = self.generate_robot_obs()
        obs_env = self.update_env_obs()
        return obs_robot, obs_env, self.target_state_
