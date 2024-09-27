import random
import collections
from Train_And_Evaluate import Train_and_Evaluate_fn
from RandomArchitecture import Random_arch
from  Mutate import Mutate , check_DNA
from tqdm import tqdm
import time 
import torch.multiprocessing as mp
import copy 
from torch.utils.tensorboard import SummaryWriter
import csv 
import os 
    
class Model() :
    def __init__(self): 
        self.avg_reward = -500
        self.highest_reward = 0 
        self.arch = [] #The Arch is encoded as a list  ,called the DNA 
        self.episodes_rewards = []
        
def wrapper(shared_list) : 
    Random_model = Model()
    Random_model.arch  =Random_arch()
    check_DNA(Random_model.arch)    
    print(f"the model {Random_model.arch}")
    re , fe , se  = Train_and_Evaluate_fn(Random_model.arch) 
    Random_model.avg_reward  = re
    Random_model.highest_reward = fe
    Random_model.episodes_rewards = se
    shared_list.put(Random_model)
    
    
def worker_fn(shared_deque , population , sample_size):
    samples = []
    random.seed(os.urandom(1024))
    while len(samples) < sample_size : # taking smaples, then picking the best one (avg_reward wise) to mutate
        samples.append(random.choice(population))
        print(samples[0].avg_reward)
    Train = True
    parent = Model()
    for i in samples: 
        if i.avg_reward > parent.avg_reward : 
            parent = i
    
    child = Model()
    child.arch , mutation= Mutate(copy.deepcopy(parent.arch))
    print("The Parent \t" , parent.arch)
    print("The cild  \t" , child.arch)
    print("Last used mutation \t" , mutation)
    
    for tem in population : 
       if tem.arch == child.arch :
           child = tem 
           Train = False 
           break
       
    if Train :    
        child.avg_reward  , child.highest_reward , child.episodes_rewards = Train_and_Evaluate_fn(child.arch)
    
    shared_deque.put_nowait((child , mutation))

    
def Regularized_Evolution(cycles , population_size, sample_size , number_of_processes = 4):
    population = collections.deque()
    history = []
    writer = SummaryWriter('Regularized_Evolution_Lunar/')
    
    filename = 'History-RUN-4.CSV'
    progress_bar = tqdm(total=cycles + population_size + 3)
    with open(filename  , mode =  'a' , newline ='' ) as file : 
        csv_write = csv.writer(file)
        if file.tell() == 0:  # Check if the file is empty
            csv_write.writerow(['DNA Sequence', 'avg_reward' , 'highest reward in an episdoe' , 'the whole 10 episodes reward' , 'what kind of mutation '] )
            
        while len(history) < population_size : 
            progress_bar.update(number_of_processes)
            with mp.Manager() as manager:  # Use Manager to create shared objects
                shared_list = manager.Queue()  # Create a shared deque
                    
                processes = []
                for rank in range(number_of_processes):
                    p = mp.Process(target=wrapper, args=(shared_list, )  , name=f'Process--{rank}')
                    p.start()
                    processes.append(p)
            
                for p in processes:
                    p.join()
                    print(f'Finished {p.name}')
            
                # Now you can use the results stored in the shared deque
                for _ in range(number_of_processes): 
                    temp = shared_list.get()
                    population.append(temp)
                    history.append(temp)
                    csv_write.writerow([str(temp.arch) , temp.avg_reward , temp.highest_reward ,str(temp.episodes_rewards) ])        
        
        

        while len(history) < cycles :
            
            with mp.Manager() as manager:  # Use Manager to create shared objects
                shared_deque = manager.Queue()  # Create a shared deque
                    
                processes = []
                for rank in range(number_of_processes):
                    p = mp.Process(target=worker_fn, args=((shared_deque , population , sample_size)) , name=f'Process--{rank}')
                    p.start()
                    processes.append(p)
            
                for p in processes:
                    p.join()
                    print(f'Finished {p.name}')
            
                # Now you can use the results stored in the shared deque
                for _ in range(number_of_processes): 
                    child  , mutation= shared_deque.get()
                    population.append(child)
                    history.append(child)
                    population.popleft()
                    progress_bar.update(number_of_processes)
                    string_DNA = str(child.arch)
                    writer.add_scalar(f"the DNA of the child - {string_DNA}" , child.avg_reward )
                    csv_write.writerow([string_DNA , child.avg_reward , child.highest_reward ,str(child.episodes_rewards) , mutation])


                
      
     ############################################       
    the_highest_model = Model() 
    for i in history :    
        if i.avg_reward > the_highest_model.avg_reward : 
            the_highest_model = i 
    
    return the_highest_model.arch , the_highest_model.avg_reward , the_highest_model.highest_reward


################################################################################
start = time.time()
#1000 - 32 - 8 

ar , ac , hc= Regularized_Evolution( 1000 , 32 ,  8 )

print("how much time did the training take in minutes" , (time.time() - start)/60 )
print("the highest model arch" , ar)
print("the highest model avg reward" , ac)
print("the highest model highest reward" , hc)