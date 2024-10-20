import random 
from random import randint , choice
from numpy import log2
import os 


def Random_arch(maximum_layers = 10 , min_qubtis = 2 , min_neurons = 2 , max_qubtis = 10 , max_neurons = 64 ) : 
    
    how_many_layers = randint(3, maximum_layers) #from 3 casue if your neural net has only 2 layers, then its just linear regression and the universla approximation theorem wont be satesfied
    Rand_DNA = []
    last_used_layer = ''
    last_used_output = 0
    
    random.seed(os.urandom(1024))
    Rand_DNA.append('C')#the first layer is not  random
    last_used_output = randint(16,max_neurons) 
    Rand_DNA.append(last_used_output)
    Rand_DNA.append(choice(['R','T']))
    last_used_layer = 'C'
    
    for layer in range(1 , how_many_layers) : 
            
        if last_used_layer == 'Q' : 
            bias_for_classical = randint(1, 100)
            if bias_for_classical > 80 :
                layer_type = choice(['Q' ,'C'])
            else :
                layer_type = 'C'
        else : 
            layer_type = choice(['Q' ,'C'])
            

        if layer_type == 'Q' : 
            Rand_DNA.append('Q')
            if last_used_layer == 'Q' : 
                if layer == how_many_layers-1 : 
                    qubit_count = min_qubtis
                else:
                    temp = [ k for k in range(min_qubtis , int(log2(last_used_output))+1)]
                    temp.append(min(max_qubtis,last_used_output))
                    qubit_count = choice(temp)
                                        
            else : # i wanna add a quantum layer , before it a classical layer 
                if last_used_output > max_qubtis : 
                    Rand_DNA[len(Rand_DNA)-3] = qubit_count = max_qubtis
                else : 
                    Rand_DNA[len(Rand_DNA)-3]  = qubit_count = randint(min_qubtis , max_qubtis) 
                    
            Rand_DNA.append(qubit_count)
            last_used_output = 2**qubit_count
            Rand_DNA.append(choice(['F' , 'O' , 'L']))
            Rand_DNA.append(choice([1,2,3]))
            last_used_layer = 'Q'
            
        else : 
            
            Rand_DNA.append('C')
            if layer == how_many_layers-1 : 
                last_used_output = 1 
            else :
                last_used_output = randint(min_neurons, max_neurons)
            Rand_DNA.append(last_used_output)
            Rand_DNA.append(choice(['R','T']))
            last_used_layer = 'C'
    
    if Rand_DNA[-1] in ['R' , 'T'] : 
        Rand_DNA.pop()
    return Rand_DNA


Random_arch()