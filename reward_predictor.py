import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reward_Predictor(nn.Module):
    def __init__(self, obs_shape, base=None, base_kwargs=None):
        
        #self.obs_shape = (1, obs_shape[1], b[2])
        
        super(Reward_Predictor, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        #self.base = base(obs_shape[0], **base_kwargs)
        self.base = base(1, **base_kwargs)
        
        self.softmax = nn.Softmax(dim=1)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.base.parameters(), lr=3e-4)
    
    def forward(self, s1s, s2s):
        assert s1s.shape == s2s.shape, "segments should be the same shape"
        
        shape = s1s.shape
        batch_size = shape[0]
        segment_length = shape[1]
        
        #print(f'batchsize: {batch_size}')
        #print(f'seg len: {segment_length}')
        
        #wrapping segment_length and batch_size, so size[0] is 'batchsize * segment_length'
        s1s = torch.reshape(s1s, ([-1] + list(s1s.shape[2:])))
        s2s = torch.reshape(s2s, ([-1] + list(s2s.shape[2:])))
        
        #stack segments and batchsize together, so new shape is '2 * batchsize * segment_length'
        x = torch.cat((s1s, s2s), axis = 0)
        
        #print(f'shape: {x.shape}')
        
        x = self.base(x)
        
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
        
        return pred
    
    def reward(self, obs):
        """
        Return (normalized) reward for each frame of a single segment.
        (Normalization involves normalizing the rewards from each member of the
        ensemble separately, then averaging the resulting rewards across all
        ensemble members.)
        """

        reward = self.base(obs)
        reward -= reward.mean()
        reward /= (reward.std() + 1e-12)
        reward *= -0.05
        return reward

    #def train(self, pref_buffer):
    def train(self, train_db, val_db, device):
        #train_db, val_db = pref_buffer.get_dbs()
        train_loader = torch.utils.data.DataLoader(
            train_db,
            batch_size=32,
            shuffle=False,
            num_workers=8
        )
        val_loader = torch.utils.data.DataLoader(
            val_db,
            batch_size=32,
            shuffle=False,
            num_workers=8
        )

        for batch in train_loader:
            s1s, s2s, prefs = batch
            
            self.optimizer.zero_grad()
            
            preds = self(s1s.to(device), s2s.to(device))
            
            loss = self.criterion(preds, prefs.to(device))
            loss.backward()
            self.optimizer.step()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                s1s, s2s, prefs = batch
                preds = self(s1s.to(device), s2s.to(device))
                
                predicted = torch.max(preds.data, 1)[1]

                total += predicted.size(0)
                correct += (predicted == prefs.to(device)).sum().item()

        print(f'accuracy of reward on {total} trajectories: {(100 * correct / total)}')




class NNBase(nn.Module):
    def __init__(self, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size
    

class CNNBase(NNBase):
    def __init__(self, num_inputs, hidden_size=512):
        super(CNNBase, self).__init__(num_inputs)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs):
        x = self.main(inputs / 255.0)

        return self.critic_linear(x)


class MLPBase(NNBase):
    def __init__(self, num_inputs, hidden_size=64):
        super(MLPBase, self).__init__(num_inputs, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs):
        x = inputs

        hidden_critic = self.critic(x)

        return self.critic_linear(hidden_critic)