import random
import collections
from Train_And_Evaluate import Train_and_Evaluate_fn
from RandomArchitecture import Random_arch
from  Mutate import Mutate , check_DNA
from tqdm import tqdm
import time 
import copy 
from torch.utils.tensorboard import SummaryWriter
import csv 
import os 
    
class Model() :
    def __init__(self): 
        self.avg_reward = 0 
        self.highest_reward = 0 
        self.arch = [] #The Arch is encoded as a list  ,called the DNA 
        self.episodes_rewards = []
        
    
def Regularized_Evolution(cycles , population_size, sample_size ):
    population = collections.deque()
    history = []
    writer = SummaryWriter('Regularized_Evolution_Lunar/')
    
    filename = 'History-RUN-4.CSV'
    progress_bar = tqdm(total=cycles + population_size + 3)
    with open(filename  , mode =  'a' , newline ='' ) as file : 
        csv_write = csv.writer(file)
        if file.tell() == 0:  # Check if the file is empty
            csv_write.writerow(['DNA Sequence', 'avg_reward' , 'highest reward in an episdoe' , 'the whole 10 episodes reward' , 'what kind of mutaion '] )
            
        while len(history) < population_size : 
            Random_model = Model()
            Random_model.arch  = Random_arch()
            check_DNA(Random_model.arch)        
            progress_bar.update(1)

            Random_model.avg_reward  , Random_model.highest_reward , Random_model.episodes_rewards= Train_and_Evaluate_fn(Random_model.arch)
            population.append(Random_model)
            history.append(Random_model)
            
            csv_write.writerow([str(Random_model.arch) , Random_model.avg_reward , Random_model.highest_reward ,str(Random_model.episodes_rewards) ])        
        
         
        while len(history) < cycles :
            samples = []
            random.seed(os.urandom(1024))
            while len(samples) < sample_size : # taking a smaple of size = sample_size , then picking the best one (avg_reward wise) to mutate
                samples.append(random.choice(population))
                
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
                       population.append(child)
                       Train = False 
                       break
                   
                if Train :    
                    child.avg_reward  , child.highest_reward , child.episodes_rewards = Train_and_Evaluate_fn(child.arch)
                
                    
                string_DNA = str(child.arch)
                writer.add_scalar(f"the DNA of the child - {string_DNA}" , child.avg_reward )
                population.append(child)
                history.append(child)
                progress_bar.update(1)
                
                csv_write.writerow([string_DNA , child.avg_reward , child.highest_reward ,str(child.episodes_rewards) , mutation])
            
    the_highest_model = Model()
    
    for i in history :    
        if i.avg_reward > the_highest_model.avg_reward : 
            the_highest_model = i 
    
    return the_highest_model.arch , the_highest_model.avg_reward , the_highest_model.highest_reward


start = time.time()
#1000 - 32 - 8 
ar , ac , hc= Regularized_Evolution( 15 , 10 ,  8 )

print("how much time did the training take in minutes" , (time.time() - start)/60 )
print("the highest model arch" , ar)
print("the highest model avg reward" , ac)
print("the highest model highest reward" , hc)