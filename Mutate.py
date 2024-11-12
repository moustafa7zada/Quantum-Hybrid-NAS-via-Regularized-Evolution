from random import randint , choice
import random 
import os 

def check_DNA(DNA) : 
    for i in range(len(DNA)): 
        if DNA[i] == 'R' or DNA[i] == 'T' : 
            assert DNA[i+1] != 'T'   
            assert DNA[i+1] != 'R' 
        if DNA[i] == 'C':
            assert type(DNA[i+1]) == int , "your clasical layer's neuron count isnt correct"  
            if i+2 < len(DNA) : 
                assert DNA[i+2] == 'R' or DNA[i+2] == 'T' , f"your classical layer's activation isnt correct here {DNA[0:i+3]}" 
        elif DNA[i] == 'Q' : 
            assert DNA[i+1] != 1 , "yooooooooooooooo one qubit here"
            assert type(DNA[i+1]) == int , "your Quantum layer's qubit count isnt correct"
            assert DNA[i+2] in ['F','O' , 'L']  , "you Quantum layer's entanglmetn isnt correct"
            assert type(DNA[i+3]) == int , "your Quantum layer's rep count isnt correct"


possible_mutations = ['add a classical layer' , 'remove a classical layer' , 'alter a layer: add' ,
 'alter a layer: remove' , 'add a quantum layer' , 'remove a quantum layer', 
 'change the repetitions of the ansatz' , 'change the entanglement type of the ansatz' , 
 'change the activation fucntion of a layer' , 'identity' ]

def Mutate(DNA ,  max_layers = 10 ,max_neurons_per_layer = 64 , max_qubits_per_circuit = 10 ,  min_neurons_per_layer = 2 ,
           min_qubits_per_layer = 2 ,  the_output_of_the_lastLayer = 2) : 
    '''for the unification of output shapes , max neuron per layer is 64 and max qubit per layer is 6 , which gives an output of 2^6 = 64
    note that 2**min_qubits_per_layer should be >= the output of the neural net (the action space) and min_neurons_per_layer should also be == the output (the action space) if its the final layer '''
    random.seed(os.urandom(1024))
    def pop_this_classical(DNA, index_of_layer , which_layer): 
        for j in range(3):
            DNA.pop(index_of_layer[which_layer])
                    
    def reduce_qubits_beforeME(DNA , index_of_layer , which_layer) :        
        i = 1 
        while True : 
            try : 
                if DNA[index_of_layer[which_layer-i]] == 'Q' : 
                    tempy = DNA[index_of_layer[which_layer-i+1]+1]
                    tempy2 = DNA[index_of_layer[which_layer-i]+1]
                    
                    if tempy == 2**(tempy2)  or tempy in [e for e in range(tempy2+1)] :
                        break
                    else :
                        for now in range(tempy2 - min_qubits_per_layer + 1 ) : 
                            if tempy == 2**(tempy2-now) or tempy in [e for e in range(tempy2-now+1)] :
                                DNA[index_of_layer[which_layer-i]+1] -= now 
                                break #break the for inside of this while 
                            
                                
                else : # WE ARRIVED AT THE CLASSICAL LAYER!!!!!!!!!!!!!
                   DNA[index_of_layer[which_layer-i]+1] = DNA[index_of_layer[which_layer-i+1]+1] 
                   break
                    
                i +=1 #for the while loop
            except : 
                break # we cant access that part of the array , so we are at the end 
            
                
                
    def reduce_qubits_afterME(DNA , index_of_layer , which_layer) :     
        i = 1 
        while True : 
            try : 
                if DNA[index_of_layer[which_layer+i]] == 'Q' :
                    tempy = DNA[index_of_layer[which_layer+i-1]+1]
                    tempy2 = DNA[index_of_layer[which_layer+i]+1]                            
                    
                    if tempy2 == 2**(tempy) or tempy2 in [e for e in range(1 , tempy+1)] : 
                        break 
                    else : #we need to edit the quantum layer in front of me 
                        for now in range(tempy2 - min_qubits_per_layer + 1) :  
                            if tempy2-now == 2**tempy or (tempy2-now) in [e for e in range(tempy+1)] : 
                                DNA[index_of_layer[which_layer+i]+1] -= now 
                                break
                            
                else : # WE ARRIVED AT THE CLASSICAL LAYER!!!!!!!!!!!!!
                   DNA[index_of_layer[which_layer-i]+1] = DNA[index_of_layer[which_layer-i+1]+1] 
                   break 
               
                i += 1 
            except : # there is no layer after this quantum layer 
                break 
    def increas_qubits_beforeME(DNA ,index_of_layer ,  which_layer ) : 
        i = 1
        while True :
            try : 
                if DNA[index_of_layer[which_layer-i]] == 'Q' : 
                    tempy = DNA[index_of_layer[which_layer-i+1]+1]
                    tempy2 = DNA[index_of_layer[which_layer-i]+1]
                    if tempy == 2**(tempy2)  or tempy in [e for e in range(tempy2+1)] :
                        break
                    else :
                        for now in range(1 , max_qubits_per_circuit  - tempy2 +1 ) : 
                            if tempy == 2**(tempy2+now) or tempy in [e for e in range(tempy2+now+1)] : 
                                DNA[index_of_layer[which_layer-i]+1] += now 
                                break #break the for inside of this while 
                                    
                else : # WE ARRIVED AT THE CLASSICAL LAYER!!!!!!!!!!!!!
                   DNA[index_of_layer[which_layer-i]+1] = DNA[index_of_layer[which_layer-i+1]+1] 
                   break
                i +=1 #for the while loop
            except : 
                break # we cant access that part of the array , so we are at the end 
                
                
                
    def pop_this_quantum(DNA, index_of_layer, which_layer):
        for j in range(4):
            DNA.pop(index_of_layer[which_layer])
    
    
    
    def insert_this_quantum(DNA ,where_to_add , the_layer_qubits , the_layer_entaglement , the_layer_reps):
        try :
            DNA.insert(index_of_layer[where_to_add] , 'Q')
            DNA.insert(index_of_layer[where_to_add]+1 ,the_layer_qubits)
            DNA.insert(index_of_layer[where_to_add]+2 , the_layer_entaglement)
            DNA.insert(index_of_layer[where_to_add]+3 , the_layer_reps)
            index_of_layer.insert(where_to_add , index_of_layer[where_to_add])
            for k in range(where_to_add+1 , len(index_of_layer)) : 
                index_of_layer[k] += 4 
            
        except : 

            DNA.append('Q')
            DNA.append(the_layer_qubits)
            DNA.append(the_layer_entaglement)
            DNA.append(the_layer_reps)
            index_of_layer.append(where_to_add)


    number_of_layers = 0 
    num_of_quantum = 0 
    num_of_classical = 0 
    index_of_quantum = [777]
    index_of_classical = [222]
    index_of_layer = [999]  
    '''filled the first element with a flag number just to get rid of it and start counting from 1 
    since in the function onwords the layers are numbered from 1 , and the random function uses that numbering '''
                             
    for i in range(len(DNA)) : 
        if DNA[i] == 'C' or DNA[i] == "Q" : 
            number_of_layers += 1 
            index_of_layer.append(i)
            if DNA[i] == 'C' : 
                num_of_classical += 1
                index_of_classical.append(i)
            else : 
                num_of_quantum += 1 
                index_of_quantum.append(i)
            
            
    while True : 
        random_mutation = choice(possible_mutations)
            
        if random_mutation == 'add a classical layer' :
            if number_of_layers >= max_layers : 
                continue
            
            where_to_add = randint(1, number_of_layers+1)
            the_layer_output = randint(min_neurons_per_layer,max_neurons_per_layer)
            activation = choice(['R' ,'T'])
            if where_to_add == number_of_layers+1 :#if iam adding at the end  
                if DNA[index_of_layer[where_to_add-1]] == 'C' : 
                    DNA.append( activation)
                DNA.append('C')
                DNA.append(the_output_of_the_lastLayer )
                break
            
            
            if DNA[index_of_layer[where_to_add]] == 'Q' : #the only case worth discussing, a quantum layer after this layer 
                
                input_after = DNA[index_of_layer[where_to_add]+1]
                if where_to_add+1 < number_of_layers:
                    if DNA[index_of_layer[where_to_add+1]] == 'Q' : 
                        DNA.insert(index_of_layer[where_to_add] , 'C')
                        DNA.insert(index_of_layer[where_to_add]+1 ,input_after)
                        DNA.insert(index_of_layer[where_to_add]+2 , activation)
                        break

                    elif DNA[index_of_layer[where_to_add+1]] == 'C'  :

                        if input_after < max_qubits_per_circuit : 
                            DNA[index_of_layer[where_to_add]+1] = max_qubits_per_circuit
                        DNA.insert(index_of_layer[where_to_add], 'C')
                        DNA.insert(index_of_layer[where_to_add]+1 ,max_qubits_per_circuit )
                        DNA.insert(index_of_layer[where_to_add]+2 , activation)
                        break
                        
                        
                else :#a quantum laye after me , its the last layer 
                    DNA.insert(index_of_layer[where_to_add] , 'C')
                    DNA.insert(index_of_layer[where_to_add]+1 , input_after)
                    DNA.insert(index_of_layer[where_to_add]+2 , activation)
                    break
                    
                
                    
            #non of the conditions happened , no quantum layer after and its not the last layer , hence ,there is a classical layer after which is okay 
            DNA.insert(index_of_layer[where_to_add] , 'C')
            DNA.insert(index_of_layer[where_to_add]+1 ,the_layer_output )
            DNA.insert(index_of_layer[where_to_add] + 2 , activation)
            break 
                    
                        
       #################################################
               
        elif random_mutation == 'remove a classical layer' :
            if number_of_layers <= 3 or num_of_classical == 2 : 
                continue
            try : 
                the_layer_to_remove = randint(2, num_of_classical)
            except : 
                continue
            the_layer_to_remove = index_of_layer.index(index_of_classical[the_layer_to_remove])
            if the_layer_to_remove == number_of_layers : #the last layer 
                if DNA[index_of_layer[the_layer_to_remove-1]] == 'C' : 
                    DNA.pop()
                DNA.pop()
                DNA.pop()

                break
            
            if DNA[index_of_layer[the_layer_to_remove-1]] == 'Q' and DNA[index_of_layer[the_layer_to_remove+1]] == 'Q' :#if its sandwiched between two quantum layers , dont do it 
                continue

            if DNA[index_of_layer[the_layer_to_remove-1]] == 'C' and DNA[index_of_layer[the_layer_to_remove+1]] == 'C' : 
                DNA[index_of_layer[the_layer_to_remove-1]+1] += randint(0,(max_neurons_per_layer-DNA[index_of_layer[the_layer_to_remove-1]+1]))
                pop_this_classical(DNA, index_of_layer, the_layer_to_remove)
                break
                
            
            if DNA[index_of_layer[the_layer_to_remove-1]] == 'C' and DNA[index_of_layer[the_layer_to_remove+1]] == 'Q' : 
                output_before = DNA[index_of_layer[the_layer_to_remove-1]+1]
                input_after = DNA[index_of_layer[the_layer_to_remove+1]+1]
                
                mean = int((output_before + input_after) /2 )
                if mean < max_qubits_per_circuit : 
                    if mean >= min_qubits_per_layer : 
                        DNA[index_of_layer[the_layer_to_remove-1]+1] = mean 
                        DNA[index_of_layer[the_layer_to_remove+1]+1] = mean 
                        pop_this_classical(DNA, index_of_layer, the_layer_to_remove)
                        break
                    else : 
                        DNA[index_of_layer[the_layer_to_remove-1]+1] = min_qubits_per_layer
                        DNA[index_of_layer[the_layer_to_remove+1]+1] = min_qubits_per_layer
                        pop_this_classical(DNA, index_of_layer, the_layer_to_remove)
                        break
                else : 
                    DNA[index_of_layer[the_layer_to_remove-1]+1] = max_qubits_per_circuit
                    DNA[index_of_layer[the_layer_to_remove+1]+1] = max_qubits_per_circuit
                    pop_this_classical(DNA, index_of_layer, the_layer_to_remove)
                    break
                
            else : 

                pop_this_classical(DNA, index_of_layer, the_layer_to_remove)
                break
    #########################################################
            
            
        elif random_mutation == 'alter a layer: add' :
            try : 
                which_layer = randint(1, number_of_layers)#i cant change the output of the last layer ,so -1 
            except : 
                continue
            
            if DNA[index_of_layer[which_layer]] == 'C' : #only the output will be effected
                if DNA[index_of_layer[which_layer]+1] >= max_neurons_per_layer or which_layer == number_of_layers : 
                    continue
                
                
                if DNA[index_of_layer[which_layer+1]] == 'Q' :#quantum layer after me 
                    if DNA[index_of_layer[which_layer+1]+1] < max_qubits_per_circuit :
                        
                        how_much_to_add = randint(1, (max_qubits_per_circuit -  DNA[index_of_layer[which_layer+1]+1]))
                        DNA[index_of_layer[which_layer+1]+1] += how_much_to_add #add to the quantum layer 
                        DNA[index_of_layer[which_layer]+1] += how_much_to_add #add to the classical layer itself 
                        break
                       
                    else : 
                        continue #the quantum layer after me is at its maximum size
                    
                                    
                
                else : #a classical layer to alter , after it comes another classical layer 
                    how_much_to_add = randint(1,(max_neurons_per_layer - DNA[index_of_layer[which_layer]+1]))
                    DNA[index_of_layer[which_layer]+1] += how_much_to_add
                    break
                
                
        
                
            else : #its a quantum layer
                if DNA[index_of_layer[which_layer]+1] >= max_qubits_per_circuit:
                    continue
                
                how_much_to_add =  randint(1,(max_neurons_per_layer - DNA[index_of_layer[which_layer]+1]))
                DNA[index_of_layer[which_layer]+1] += how_much_to_add
                
                i = 1
                while True :
                    try : 
                        if DNA[index_of_layer[which_layer+i]] == 'Q' :
                            tempy = DNA[index_of_layer[which_layer+i-1]+1]
                            tempy2 = DNA[index_of_layer[which_layer+i]+1]                            
                            
                            if tempy2 == 2**(tempy) or tempy2 in [e for e in range(1 , tempy +  1)] : 
                                break 
                            else : #we need to edit the quantum layer in front of me 
                                for now in range(1 , max_qubits_per_circuit - tempy2 + 1) :  
                                    if tempy2 == 2**(tempy+now) or tempy2 in [e for e in range(1 , tempy+now+1)] : 
                                        DNA[index_of_layer[which_layer-i]+1] += now 
                                        break
                                    
                            
                        else : #the layer after me is a classical layer , so it takes any input 
                           break 
                    except : # there is no layer after this quantum layer 
                        break 
                    
                                     
                    i += 1 
                    
                    
                i = 1
                while True :
                    try : 
                        if DNA[index_of_layer[which_layer-i]] == 'Q' : 
                            tempy = DNA[index_of_layer[which_layer-i+1]+1]
                            tempy2 = DNA[index_of_layer[which_layer-i]+1]
                            
                            if tempy == 2**(tempy2)  or tempy in [e for e in range(tempy2+1)] :
                                break
                            else :
                                for now in range(1 , max_qubits_per_circuit  - tempy2 +1 ) : 
                                    if tempy == 2**(tempy2+now) or tempy in [e for e in range(tempy2+now+1)] :
                                        DNA[index_of_layer[which_layer-i]+1] += now 
                                        break #break the for inside of this while 
                                    
                                        
                        else : #a classical layer behind me
                            if DNA[index_of_layer[which_layer-i]+1] >= DNA[index_of_layer[which_layer-i+1]+1] :
                                DNA[index_of_layer[which_layer-i+1]+1]  = DNA[index_of_layer[which_layer-i]+1] =  max_qubits_per_circuit
                                return DNA ,random_mutation
                            else : 
                                DNA[index_of_layer[which_layer-i+1]+1]  = DNA[index_of_layer[which_layer-i]+1]
                                return DNA , random_mutation
                            
                            
                        i +=1 #for the while loop
                    except : 
                        break # we cant access that part of the array , so we are at the end 
    ########################################################
                
        elif random_mutation == 'alter a layer: remove'  : 
            try: 
                which_layer = randint(2, number_of_layers-1)
            except : 
                continue
            if DNA[index_of_layer[which_layer]] == 'C' : #only the output will be effected 
                if DNA[index_of_layer[which_layer]+1] <= min_neurons_per_layer  or which_layer == number_of_layers:
                    continue
                
                        
            if DNA[index_of_layer[which_layer]] == 'C' : 
                how_much_to_remove = randint(1,(DNA[index_of_layer[which_layer]+1]  - min_neurons_per_layer))
                if  DNA[index_of_layer[which_layer + 1 ]] == 'Q' :
                    DNA[index_of_layer[which_layer]+1] -= how_much_to_remove
                    reduce_qubits_afterME(DNA , index_of_layer , which_layer)
                    break 
                
                
            else : # a quantum layer to edit 
                if DNA[index_of_layer[which_layer]+1] == min_qubits_per_layer : 
                    continue
                try : 
                    how_much_to_remove = randint(1,(DNA[index_of_layer[which_layer]+1]  - min_qubits_per_layer))
                except : 
                    continue
                DNA[index_of_layer[which_layer]+1] -= how_much_to_remove
                reduce_qubits_afterME(DNA , index_of_layer , which_layer)
                reduce_qubits_beforeME(DNA , index_of_layer , which_layer)
                break 
               
   
    ##########################################################
   
        elif random_mutation == 'add a quantum layer' :
            if number_of_layers >= max_layers : 
                continue
            try : 
                where_to_add = randint(2, number_of_layers+1) #index for the index_of_layers(insertion)
            except: 
                continue
            the_layer_qubits = randint(min_qubits_per_layer , max_qubits_per_circuit) 
            the_layer_entaglement = choice(['L' , 'O' , 'F'])
            the_layer_reps = choice([1,2,3])
    
    
    
            if where_to_add == number_of_layers+1:
                if DNA[index_of_layer[where_to_add-1]] == 'C' :
                    DNA[index_of_layer[where_to_add-1]+1] = the_layer_qubits
                    DNA.append(choice(['R', 'T']) ) 
                    insert_this_quantum(DNA, where_to_add, the_layer_qubits , the_layer_entaglement, the_layer_reps)
                    break
                else :
                    continue
                            
            insert_this_quantum(DNA, where_to_add, the_layer_qubits, the_layer_entaglement, the_layer_reps)
            if DNA[index_of_layer[where_to_add+1]] == 'Q' : 
                increas_qubits_beforeME(DNA, index_of_layer, where_to_add +1 )
                break
                    
            
            if DNA[index_of_layer[where_to_add-1]] == 'C' :
                DNA[index_of_layer[where_to_add-1]+1] = the_layer_qubits
                break
            
            
            if DNA[index_of_layer[where_to_add-1]] == 'Q' :
                increas_qubits_beforeME(DNA, index_of_layer, where_to_add)
                break
            
        ##############################################        
                
        elif random_mutation == 'remove a quantum layer' :
            if number_of_layers <= 3 or num_of_quantum == 0:
                continue
            
            which_layer = randint(1, num_of_quantum)
            which_layer =index_of_layer.index(index_of_quantum[ which_layer ])
            
            
               #this if is for the case , A CLASSICAL layer behinde me         
            if DNA[index_of_layer[which_layer-1]] == 'C' :
                output_before = DNA[index_of_layer[which_layer-1]+1]
                if  which_layer == number_of_layers :#if this is the last layer 
                    DNA[index_of_layer[which_layer-1]+1] =  the_output_of_the_lastLayer
                    pop_this_quantum(DNA, index_of_layer, which_layer)
                    DNA.pop()
                    break
                
                #if its not the last layer , there is a classical layer before me and a quantum layer after me 
                if DNA[index_of_layer[which_layer+1]] == 'Q':
                    input_after = DNA[index_of_layer[which_layer+1]+1]
                    DNA[index_of_layer[which_layer-1]+1] = input_after
                    pop_this_quantum(DNA, index_of_layer, which_layer)
                    break    
                    
                else : # a classical layer before me and a classical after me , so just remove that quantum layer 
                    pop_this_quantum(DNA, index_of_layer, which_layer)
                    break
                    
                    
            else : #this else is for the case , A QUANTUM LAYER BEHINDE ME 
                output_before = [e for e in range(DNA[index_of_layer[which_layer-1]+1] + 1 )]
                output_before.append(2**DNA[index_of_layer[which_layer-1]+1])
                if which_layer == number_of_layers :#if this is the last layer 
                    pop_this_quantum(DNA, index_of_layer, which_layer)
                    break
                    
                #if its not the last layer , there is a quantum layer before and after and iam standing on  a quantum layer  ,what the f*ck is this architecture XD
                elif DNA[index_of_layer[which_layer+1]] == 'Q':
                    input_after = DNA[index_of_layer[which_layer+1]+1]
                    if input_after in output_before : #it works like this , dont fix what is not broken 
                        pop_this_quantum(DNA, index_of_layer, which_layer)
                        break
                    
                    else :
                        pop_this_quantum(DNA, index_of_layer, which_layer)
                        increas_qubits_beforeME(DNA, index_of_layer, which_layer)  
 
                        break                        
                        
                else : # a classical layer before me and a classical after me , so just remove that quantum layer 
                    pop_this_quantum(DNA, index_of_layer, which_layer)
                    break
                            
                
            
        elif random_mutation == 'change the repetitions of the ansatz' :
            if num_of_quantum == 0 :  
                continue
            which_layer = randint(1, num_of_quantum)
            current_rep = DNA[index_of_quantum[which_layer]+3]
           
            if current_rep == 1 : 
                DNA[index_of_quantum[which_layer]+3] = choice([ 2 ,3])
                break
            elif current_rep == 2 : 
                DNA[index_of_quantum[which_layer]+3] = choice([1 , 3])
                break
            else : 
                DNA[index_of_quantum[which_layer]+3] = choice([1 , 2])
                break
            

        elif random_mutation == 'change the entanglement type of the ansatz' : 
            if num_of_quantum == 0 : 
                continue
            
            which_layer = randint(1, num_of_quantum)
            current_entang = DNA[index_of_quantum[which_layer]+2]
            
            if current_entang == 'L' : 
                DNA[index_of_quantum[which_layer]+2] = choice(['O' , 'F'])
                break
            elif current_entang == 'O' : 
                DNA[index_of_quantum[which_layer]+2] = choice(['L' , 'F'])
                break
            else : 
                DNA[index_of_quantum[which_layer]+2] = choice(['L' , 'O'])
                break

        elif random_mutation == 'change the activation fucntion of a layer':
            if num_of_classical != 0 :
                try : 
                    temp = index_of_classical[randint(1, num_of_classical-1)]
                except : 
                    temp  = index_of_classical[1] 
            else : 
                continue
            if DNA[temp+2] == 'R' : 
                DNA[temp+2] = 'T'
                break
                
            elif DNA[temp+2] == 'T': 
                DNA[temp+2] = 'R'
                break
          
    
        elif random_mutation == 'identity' :
            return DNA , random_mutation

    return DNA , random_mutation