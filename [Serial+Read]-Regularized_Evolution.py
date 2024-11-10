import random
import collections
from Train_And_Evaluate import Train_and_Evaluate_fn
from  Mutate import Mutate 
from tqdm import tqdm
import time 
import copy 
from torch.utils.tensorboard import SummaryWriter
import csv 
import ast 
import os 
import wandb 
    


class Model() :
    def __init__(self): 
        self.avg_reward = 0 
        self.highest_reward = -500
        self.arch = [] #The Arch is encoded as a list  ,called the DNA 
        self.episodes_rewards = []
        
    
def Regularized_Evolution(cycles , population_size, sample_size ):
    wandb.init(project = "Serial QPPO , 10 Nov" ,  
               config={
        "learning_rate":2.5e-4,
        "updates": 100,
        "cycles" : cycles , 
        "population_size" : population_size , 
        "epochs": 4,
        })
    population = collections.deque()
    history = []
    writer = SummaryWriter('Regularized_Evolution_Lunar/')
    
    filename = 'History-RUN-4.CSV'
    progress_bar = tqdm(total=cycles + population_size + 3)
    with open(filename , mode  = 'r' ) as file : 
        reader = csv.reader(file)
        next(reader)
        
        # Read each row and print the first 4 columns
        for row in reader:
            temp_model = Model()
            temp_model.arch , temp_model.avg_reward , temp_model.highest_reward = ast.literal_eval(row[0]) , float(row[1]) , float(row[2])
            history.append(temp_model)
            
        for i in range(1,population_size+1): 
            population.append(history[-i])
            
    with open(filename  , mode =  'a' , newline ='' ) as file : 
        csv_write = csv.writer(file)

        while len(history) < cycles :
            samples = []
            random.seed(os.urandom(1024))
            while len(samples) < sample_size : # taking samples, then picking the best one (avg_reward wise) to mutate
                samples.append(random.choice(population))
                
                Train = True
                parent = Model()
                for i in samples: 
                    print("iam right here" , i.avg_reward , type(parent.avg_reward)) 
                    if i.avg_reward > parent.avg_reward : 
                        parent = i
                
            
                child = Model()
                child.arch , mutation= Mutate(copy.deepcopy(parent.arch))
                print("The Parent \t" , parent.arch)
                print("The child  \t" , child.arch)
                print("Last used mutation \t" , mutation)
                
                for tem in population : 
                   if tem.arch == child.arch :
                       child = tem 
                       Train = False 
                       break
                   
                if Train :    
                    child.avg_reward  , child.highest_reward , child.episodes_rewards = Train_and_Evaluate_fn(child.arch)
                
                    
                string_DNA = str(child.arch)
                writer.add_scalar(f"the DNA of the child - {string_DNA}" , child.avg_reward )
                population.append(child)
                population.pop()
                history.append(child)
                progress_bar.update(1)
                
                csv_write.writerow([string_DNA , child.avg_reward , child.highest_reward ,str(child.episodes_rewards) , mutation])
                wandb.log({"Model Arch" :string_DNA , "avg_reward": child.avg_reward , "max reward" : child.highest_reward })
    the_highest_model = Model()
    
    for i in history :    
        if i.avg_reward > the_highest_model.avg_reward : 
            the_highest_model = i 
    
    return the_highest_model.arch , the_highest_model.avg_reward , the_highest_model.highest_reward


start = time.time()
#1000 - 32 - 8 
ar , ac , hc= Regularized_Evolution( 1000 , 32 ,  8 )

print("how much time did the training take in minutes" , (time.time() - start)/60 )
print("the highest model arch" , ar)
print("the highest model avg reward" , ac)
print("the highest model highest reward" , hc)
