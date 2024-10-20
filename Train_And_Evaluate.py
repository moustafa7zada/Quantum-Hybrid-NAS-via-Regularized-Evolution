from Agent_with_pennylane import Agent_From_DNA
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import gym
import random

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode = 'rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
#100 - 128 - 8
def Train_and_Evaluate_fn(DNA , number_of_updates =100,  number_of_steps = 64 , number_of_environments = 8 ,  learning_rate = 2.5e-4 ,
                       device = 'cuda' , annealing = True , gae = True , number_of_epoches = 4 , gamma = 0.99 ,gae_lambda=0.98 ,
                         clipping_coeff = 0.2,value_clipping = True , gym_id ="CartPole-v1" , seed = 1  , capture_video = False
                       , max_grad_norm = 0.5 , value_coeff = 0.5 , entro_coeff = 0.01 , norm_adv = True
                       ) :


    arguments = locals()

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ########################### GPUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU ###############################

    #torch.backends.cudnn.deterministic = True
    #########
    # batch size = steps * env 
    #min batch size = batch size / epochs 
    batch_size = number_of_steps * number_of_environments
    minibatch_size = batch_size //  number_of_epoches

    
    global_step_count = 0
    start_time = time.time()
    run_name = f"{DNA}__{gym_id}__{seed}__{int(time.time())}"

    writer = SummaryWriter(f'individual_models/RUN-4/{run_name}')

    writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in arguments.items()])),
    )


    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id , seed + i, i, capture_video, run_name) for i in range(number_of_environments)]
    )
    
    
    agent = Agent_From_DNA(DNA)

    optimizer = optim.Adam(agent.parameters()  , lr = learning_rate , eps = 1e-5)


    #return 0 
    obs = torch.zeros((number_of_steps, number_of_environments) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((number_of_steps, number_of_environments) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((number_of_steps, number_of_environments)).to(device)
    rewards = torch.zeros((number_of_steps, number_of_environments)).to(device)
    dones= torch.zeros((number_of_steps, number_of_environments)).to(device)
    values = torch.zeros((number_of_steps, number_of_environments)).to(device)


    next_obs = torch.tensor(envs.reset()[0]).to(device) # these are the initial observation and done(now called terminated and truncated) , we called it next since after each iteration the new obs will be used in the begining of the new step
    next_done = torch.zeros(number_of_environments).to(device)

    default_time = time.time()
    for update in range(1 , number_of_updates + 1 ):
        if annealing == True :
            frac = 1.0 - (update - 1.0)/number_of_updates
            new_lr = learning_rate * frac
            optimizer.param_groups[0]["lr"] = new_lr


        for step in range(0 , number_of_steps) :
            global_step_count += 1* number_of_environments
            obs[step] = next_obs
            dones[step] = next_done
            #truncates[step] = next_truncated



            with torch.no_grad() :
                action , logprob ,  _ , value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, done, info , _ = envs.step(action.cpu().numpy()) #openai gym works only with numpy so you pass the action as a numpy array on cpu and the reward is an array so we transform it into a tensor
            rewards[step] = torch.tensor(reward).to(device).view(-1) #everything we got from the envirmonemt is as arrays so we have to transfer them into
            next_obs = torch.Tensor(next_obs).to(device)  # we dont put them in the array because we do that in the beginign of the for loop
            next_done = torch.Tensor(done).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1,-1)
            if gae == True :
                advantages = torch.zeros_like(rewards).to(device)
                last_gae_adv = 0
                for t in reversed(range(number_of_steps)) :
                    if t == number_of_steps - 1 :
                        next_not_terminal = 1.0 - next_done
                        nextvalues = next_value
                    else :
                        next_not_terminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]

                    td_error = rewards[t] + gamma * nextvalues * next_not_terminal - values[t]
                    advantages[t] = last_gae_adv = td_error + gae_lambda*last_gae_adv*next_not_terminal * gamma
                returns = advantages + values

            else :
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(number_of_steps)) :
                    if t == number_of_steps - 1 :
                        next_not_terminal = 1.0 - next_done
                        next_return = next_value
                    else :
                        next_not_terminal = 1 - dones[t+1]
                        next_return = returns[t+1]
                        returns[t] = rewards[t] + gamma * next_not_terminal * next_return
                    advantages = returns - values



        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)


        #optimizig timeee
        b_inds = np.arange( batch_size)
        clipfracs =[]
        for epoch in range(number_of_epoches) :
            np.random.shuffle(b_inds)
            for start in range(0,batch_size , minibatch_size):
                end = start +  minibatch_size
                mb_inds = b_inds[start:end]

                _ , newlogprob , entropy , newvalue =  agent.get_action_and_value(b_obs[mb_inds].to(device), b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clipping_coeff).float().mean().item()]


                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)


                #the policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages *torch.clamp(ratio ,1-clipping_coeff , 1+clipping_coeff )
                policy_loss = torch.max(pg_loss1 , pg_loss2).mean()


                #the_value_loss
                newvalue = newvalue.view(-1)#the value that you got from the new model
                if value_clipping :
                    v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
                    v_clipped = b_values[mb_inds] + torch.clip(
                            newvalue - b_returns[mb_inds]
                            , -clipping_coeff
                            ,clipping_coeff)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
                    v_loss_max = torch.max(v_loss_clipped , v_loss_unclipped)
                    v_loss =0.5 * v_loss_max.mean()
                else :
                    v_loss = 0.5 *((newvalue - b_returns[mb_inds])**2).mean()


                #entropy loss
                entropy_loss = entropy.mean()
                
                #the total loss   
                total_loss = policy_loss - (entro_coeff * entropy_loss) + (value_coeff * v_loss)



                optimizer.zero_grad()
                total_loss.backward()
                    
                torch.nn.utils.clip_grad_norm_(agent.parameters() , max_grad_norm)
                optimizer.step()


        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step_count)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step_count)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step_count)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step_count)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step_count)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step_count)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step_count)
        writer.add_scalar("losses/explained_variance", explained_var, global_step_count)
        writer.add_scalar("charts/SPS", int(global_step_count / (time.time() - start_time)), global_step_count)
        print("SPS:", int(global_step_count / (time.time() - start_time)))

    envs.close()


    ''' EVALUATION SECTION '''

    env = gym.make(gym_id)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    total_reward = 0
    episode_reward = [0 for _ in range(10)]

    for ep in range(10) :
        obs = env.reset()[0]
        obs = torch.tensor(obs).to(device)
        tru = False
        ter = False

        while not tru and not ter :
            with torch.no_grad() :
                action = agent.get_action(obs)
            obs , reward, tru , ter , _  = env.step(action.cpu().numpy())
            obs = torch.tensor(obs).to(device)
            episode_reward[ep] += reward

        total_reward += episode_reward[ep]

    avg_reward = total_reward / 10

    writer.add_scalar("avg_reward", avg_reward)
    writer.add_scalar("highest reward in a episode ", max(episode_reward))
    writer.close()
    env.close()

    return avg_reward , max(episode_reward) , episode_reward
