import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture:
    Splits Value (V) and Advantage (A) streams.
    Input: State Vector
    Output: Q-Values for Actions
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DuelingDQN, self).__init__()
        
        # Feature Layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU()
        )
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) Buffer
    Reduced size for 1-5m framework (~1000)
    """
    def __init__(self, capacity=1000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
    
    def push(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.buffer else 1.0
        
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = max_prio # Deque doesn't support index set easily if rotating, but here we treat priorities as parallel list logic? 
            # Simplified: Just use a list for priorities for correct indexing
            
        self.pos = (self.pos + 1) % self.capacity
        
        # Maintain priorities as list for easy indexing
        if isinstance(self.priorities, deque):
             self.priorities = list(self.priorities)
        if len(self.priorities) < len(self.buffer):
             self.priorities.append(max_prio)
        else:
             self.priorities[self.pos-1 if self.pos > 0 else self.capacity-1] = max_prio

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []
            
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = float(prio)

class DQNAgent:
    def __init__(self, input_dim, action_dim=3, lr=1e-4, gamma=0.99, device="cpu", model_path=None):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        self.model_path = model_path
        
        self.policy_net = DuelingDQN(input_dim, action_dim).to(device)
        self.target_net = DuelingDQN(input_dim, action_dim).to(device)
        
        if model_path and os.path.exists(model_path):
            try:
                self.policy_net.load_state_dict(torch.load(model_path, map_location=device))
                print(f"✅ Loaded DQN weights from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load DQN weights: {e}")
                
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(capacity=1000)
        
        self.batch_size = 32
        self.beta = 0.4

    def predict(self, state):
        """
        Return (max_q, action_idx, q_values_numpy) without exploration.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        max_q, action_idx = torch.max(q_values, dim=1)
        return max_q.item(), action_idx.item(), q_values.detach().cpu().numpy()[0]
        
    def act(self, state, epsilon=0.05):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
            
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()
        
    def get_confidence(self, state):
        """Return max Q-value as confidence proxy"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.max().item()

    def learn(self):
        if len(self.memory.buffer) < self.batch_size:
            return
            
        samples, indices, weights = self.memory.sample(self.batch_size, self.beta)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Double DQN Logic
        # 1. Select action using Policy Net
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        # 2. Evaluate action using Target Net
        next_q_values = self.target_net(next_states).gather(1, next_actions)
        
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        curr_q_values = self.policy_net(states).gather(1, actions)
        
        # Weighted MSE Loss
        loss = (weights * (curr_q_values - expected_q_values.detach()).pow(2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update Priorities
        errors = (curr_q_values - expected_q_values.detach()).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, (errors + 1e-5).flatten())
        
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

