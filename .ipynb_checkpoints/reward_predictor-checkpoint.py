import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardPredictorNetwork(nn.Module):
    """
    Predict the reward that a human would assign to each frame of
    the input trajectory, trained using the human's preferences between
    pairs of trajectories.
    Network inputs:
    - s1/s2     Trajectory pairs
    - pref      Preferences between each pair of trajectories
    Network outputs:
    - r1/r2     Reward predicted for each frame
    - rs1/rs2   Reward summed over all frames for each trajectory
    - pred      Predicted preference
    """
    
    def __init__(self, env):
        super(RewardPredictorNetwork, self).__init__()
        nh, nw, nc = env.observation_space.shape
        nact = env.action_space.n
        
        self.conv1 = nn.Conv2d(nc, 8, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(32)
        
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(nw))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(nh))))
        
        linear_input_size = convw * convh * 32
        
        self.head1 = nn.Linear(linear_input_size, 16)
        self.head2 = nn.Linear(16, 1)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, s1s, s2s):        
        assert s1s.shape == s2s.shape, "segments should be the same shape"
        
        shape = s1s.shape
        batch_size = shape[0]
        segment_length = shape[1]
        
        #wrapping segment_length and batch_size, so size[0] is 'batchsize * segment_length'
        s1s = torch.reshape(s1s, ([-1] + list(s1s.shape[2:])))
        s2s = torch.reshape(s2s, ([-1] + list(s2s.shape[2:])))
        
        #stack segments and batchsize together, so new shape is '2 * batchsize * segment_length'
        x = torch.cat((s1s, s2s), axis = 0)
        
        #pass segment data through convolutional model
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.head1(x))
        x = self.head2(x)
    
        #Split rewards at each frame back into individual segments
        #where shape is 'batchsize * segment_length'
        r1s, r2s = torch.split(x, batch_size * segment_length)
        
        #unwrapping segment_length and batch_size
        r1s =  torch.reshape(r1s, [batch_size, segment_length, -1])
        r2s =  torch.reshape(r2s, [batch_size, segment_length, -1])
        
        #Sum over all the segment_length to get a shape of 'batch_size'
        r1 = torch.sum(r1s, axis = 1)
        r2 = torch.sum(r2s, axis = 1)
        
        #Predict human preference for each segment
        rs = torch.cat((r1, r2), axis = 1)
        pred = self.softmax(rs)
        
        return r1s, r2s, pred
    
    def predict_single(self, s1s):
        
        #pass segment data through convolutional model
        x = F.relu(self.bn1(self.conv1(s1s)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.head1(x))
        r1s = self.head2(x)
        
        return r1s
    
class RewardPredictorEnsemble(nn.Module):
    def __init__(self,
                 env,
                 lr=3e-4,
                 n_preds=1):
        
        super(RewardPredictorEnsemble, self).__init__()
        self.n_preds = n_preds
        self.rps = []
        for pred_n in range(n_preds):
            rp = RewardPredictorNetwork(env)
            self.rps.append(rp)
    
        self.n_steps = 0
        
    def forward(self, s1s, s2s):
        """
        Return concatenation of all the reward networks outputs
        """
        return torch.stack([rp(s1s.clone(), s2s.clone()) for rp in self.rps], dim=0)
    
    def raw_reward(self, x):
        """
        Returns reward at each frame for a batch of clips
        """
        stack = torch.stack([rp.predict_single(x.clone()) for rp in self.rps], dim=0)
        return torch.squeeze(stack)
    
    def reward(self, x):
        """
        Return (normalized) reward for each frame of a single segment.
        (Normalization involves normalizing the rewards from each member of the
        ensemble separately, then averaging the resulting rewards across all
        ensemble members.)
        """
        with torch.no_grad():
            ensemble_rewards = self.raw_reward(x).permute(1,0)
            
        ensemble_rewards -= ensemble_rewards.mean(axis = 0)
        ensemble_rewards /= (ensemble_rewards.std(axis = 0)+1e-12)
        ensemble_rewards *= 0.05
        
        rs = torch.mean(ensemble_rewards, axis=1)
        
        return rs
    
    def preferences(self, s1s, s2s):
        """
        Predict probability of human preferring one segment over another
        for each segment in the supplied batch of segment pairs.
        """
        #with torch.no_grad():
        ensemble_rewards = self.forward(s1s, s2s)
        return ensemble_rewards
    
    def train(self, prefs_train, prefs_val, val_interval):
        """
        Train all ensemble members for one epoch
        """
        
        print("Training with {} prefrences and {} Test".format(len(prefs_train), len(prefs_val)))
        
        start_steps = self.n_steps
        start_time = time.time()
        
        