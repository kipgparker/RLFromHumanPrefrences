import copy
import glob
import os
import time
from collections import deque

from torch import roll

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from reward_predictor import Reward_Predictor
from pref_db import PrefBuffer, PrefDB, Segment

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.a2c_acktr import A2C_ACKTR
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

import torch.multiprocessing as mp

import garner as g

#mp.set_start_method("spawn")
#from multiprocessing import Process


def main():
    args = get_args()

    mp.set_start_method("spawn")

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)

    #if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    #    torch.backends.cudnn.benchmark = False
    #    torch.backends.cudnn.deterministic = True

    #torch.set_num_threads(1)  

    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, device, False)

    reward_predictor = Reward_Predictor(
        envs.observation_space.shape
    )
    reward_predictor.to(device)
    reward_predictor.base.share_memory()

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    g.login('kipgparker@gmail.com')
    g.select_pool('Deep reinforcement learning from human prefrences')


    pref_db_train = PrefDB(maxlen=30000)
    pref_db_val = PrefDB(maxlen=5000)
    pref_buffer = PrefBuffer(garner = g, db_train=pref_db_train,
                    db_val=pref_db_val, maxlen=30)

                    

    processes = []
    
    p = mp.Process(target=train_policy, args=(args, actor_critic, reward_predictor))
    p.start()
    processes.append(p)

    p = mp.Process(target=train_reward_predictor, args=(args, reward_predictor, pref_db_train, pref_db_val))
    p.start()
    processes.append(p)
    
    for p in processes:
        p.join()

    pref_buffer.start_thread()
    #f(args, agent, rollouts, envs, actor_critic, reward_predictor, device, eval_log_dir)
    


def train_reward_predictor(args, reward_predictor, db_train, db_val):

    device = torch.device("cuda:0" if args.cuda else "cpu")




    #Run forever
    while True:
        #db_train, db_val = pref_buffer.get_dbs()
        reward_predictor.train(db_train, db_val, device)

def train_policy(args, actor_critic, reward_predictor):

    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    #actor_critic.base.share_memory()

    agent = A2C_ACKTR(
        actor_critic,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        alpha=args.alpha,
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            #print(reward.shape)
            #print(reward)
            #break
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
                
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()


        #Switch environment reward for custom reward_predictor
        obs_shape = rollouts.obs.size()[3:]

        #Alternative with seq length
        #obs_shape = rollouts.obs.size()[2:]
        #rollouts.obs[:-1].view(-1, *obs_shape)
        

        with torch.no_grad():
            preds = reward_predictor.reward(
                rollouts.obs[:-1, :, 0].view(-1, 1, *obs_shape)
            )
            #print(rollouts.obs[:-1, :, 0].view(-1, 1, *obs_shape).shape)
            rollouts.rewards = preds.view(args.num_steps, args.num_processes, 1)


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, "a2c")
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                    args.num_processes, eval_log_dir, device)


    #mp.set_start_method('spawn')
    #proc = mp.Process(target=f)
    #proc.start()
    #proc.join()


if __name__ == "__main__":
    main()
