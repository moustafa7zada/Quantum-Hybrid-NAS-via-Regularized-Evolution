import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from pennylane import numpy as np 
import pennylane as qml 
import random 

Entanglmet_types = {'F':'full' , 'L' : 'linear' , 'O' :'circular'}


class MultLayer(nn.Module): 
    def __init__(self): 
        super().__init__()
    def forward(self , inputs) : 
        return torch.mul(inputs , 10 )

class My_quantum_layer(nn.Module): 
    def __init__(self, qnode, weights_shape , measurment_shape  , kind_of_measurment , torch_device = 'cpu'): 
        super().__init__()
        self.weight = nn.Parameter(torch.randn(weights_shape['weights'])).to(torch_device)
        print("These are your shapes:", weights_shape)
        self.qnode = qnode 
        self.measurment_shape = measurment_shape 
        self.kind_for_measurment = kind_of_measurment 
        self.torch_device = torch_device

    def forward(self, inputs): 
    
        q_out = torch.Tensor(0, self.measurment_shape ).to(self.torch_device)
        
        
        if self.kind_for_measurment == 'Expval' : 
            for elem in inputs:
                q_out_elem = torch.hstack(self.qnode(elem, self.weight)).float().unsqueeze(0)
                q_out = torch.cat((q_out.to(self.torch_device), q_out_elem.to(self.torch_device)))
                
            
            return q_out
        else : #Prob
            for elem in inputs:
                q_out_elem = self.qnode(elem, self.weight).float().unsqueeze(0)
                q_out = torch.cat((q_out.to(self.torch_device), q_out_elem.to(self.torch_device)))
                
            
            return q_out
    
            

def Layer_Ortho_inti(layer , std = np.sqrt(2) , bias = 0.0):
    torch.nn.init.orthogonal_(layer.weight , std )
    torch.nn.init.constant_(layer.bias , bias)
    return layer 

    
class Agent_From_DNA(nn.Module): 
    
    def __init__(self  ,DNA , the_input_of_theFIERSTlayer = 4  , simulator = 'lightning.qubit') : 
        super().__init__()
        self.Critic = nn.Sequential()
        self.number_of_measurments_critic = 0 
        self.number_of_measurments_actor = 0 
        self.kind_of_measurments_critic = str 
        self.kind_of_measurments_actor = str 
        self.counts = DNA.count("Q")
        self.internal_layer_index_actor = 0 
        self.internal_layer_index_critic = 0 
        self.index_of_quantum = [] 
        self.number_of_layers = DNA.count('C') + self.counts + DNA.count('R') + DNA.count('T')
        
        l = 0 
        for h in DNA: 
            if h == 'C' or h == 'R' or h == 'T' : 
              l += 1
            elif h =='Q' : 
                l += 1
                self.index_of_quantum.append(l)


        qnode_critic = {} 
        Circuits_critic = {}
        qlayer_critic = {}
        weight_shapes_critic = {}
        device_critic = {}
        num_qubits_critic  = {}
        Entanglment_critic = {}
        ansatz_critic = {}
        kind_of_measurments_critic = {}
        number_of_measurments_critic = {}
        
        
        layer_count = 0
        self.Critic = nn.Sequential()

        for i in range(len(DNA)) : 
            if DNA[i] == 'C' : 
                layer_count += 1
                if layer_count == 1 :
                    self.Critic.add_module(f"linear_{layer_count}", Layer_Ortho_inti(torch.nn.Linear(the_input_of_theFIERSTlayer, DNA[i+1]) ) )
                else :
                    if 'C' in DNA[i-3:i] :
                        self.Critic.add_module(f"linear_{layer_count}", Layer_Ortho_inti(torch.nn.Linear(DNA[i-2], DNA[i+1]) ) )
                    else : 
                        self.Critic.add_module(f"linear_{layer_count}", Layer_Ortho_inti(torch.nn.Linear(2**DNA[i-3], DNA[i+1]) ) )
                i += 1
            
            
            elif DNA[i] == 'R' : 
                layer_count += 1
                self.Critic.add_module(f"ReLu_{layer_count}", nn.ReLU())
            elif DNA[i] == 'T' :
                layer_count += 1
                self.Critic.add_module(f"Tanh_{layer_count}", nn.Tanh())
            elif DNA[i] == 'Q' : 
                layer_count += 1

                num_qubits_critic[f"num_for_{layer_count}"] = DNA[i+1]
                Entanglment_critic[f"ent_for_{layer_count}"] = DNA[i+2]
                if Entanglment_critic[f"ent_for_{layer_count}"] == 'O' : 
                    Entanglment_critic[f"ent_for_{layer_count}"] = random.choice(['F' , 'L']) 
                ansatz_critic[f"ans_for_{layer_count}"] = DNA[i+3]
                
                
                if len(DNA)-1 > i+4:
                    
                    if DNA[i+4] == "C" :
                        kind_of_measurments_critic[f"kind_for_critic{layer_count}"]  = 'Prob'
                        number_of_measurments_critic[f"meas_for_critic_{layer_count}"] = 2**num_qubits_critic[f"num_for_{layer_count}"]
                    
                    else :  #the only case when we actually need the modification of qiskit , when Q layer after it a Q layer 
                        if 2**num_qubits_critic[f"num_for_{layer_count}"] ==  DNA[i+5]  : 
                            kind_of_measurments_critic[f"kind_for_critic{layer_count}"]  = 'Prob'
                            number_of_measurments_critic[f"meas_for_critic_{layer_count}"] = DNA[i+5]
                        else : 
                            kind_of_measurments_critic[f"kind_for_critic{layer_count}"] = 'Expval'
                            number_of_measurments_critic[f"meas_for_critic_{layer_count}"]= DNA[i+5]
                    
                else : #the last layer , its a quantum layer 
                    kind_of_measurments_critic[f"kind_for_critic{layer_count}"]  = 'Expval'
                    number_of_measurments_critic[f"meas_for_critic_{layer_count}"]= 1
                
                
                def Circuit(inputs , weights):
                    self.internal_layer_index_critic  = (self.internal_layer_index_critic +  1) % self.counts  
                    layer_count = self.index_of_quantum[self.internal_layer_index_critic - 1 ]
                    
                    qml.templates.AngleEmbedding(inputs, wires = range(num_qubits_critic[f"num_for_{layer_count}"]) )
            
                    if Entanglment_critic[f"ent_for_{layer_count}"] == 'L' : 
                        qml.templates.BasicEntanglerLayers(weights , wires=range(num_qubits_critic[f"num_for_{layer_count}"]) )
                    else :  
                        qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits_critic[f"num_for_{layer_count}"]))
                        
                    
                    if kind_of_measurments_critic[f"kind_for_critic{layer_count}"] == 'Expval':
                        return [qml.expval(qml.Z(qubit)) for qubit in range( number_of_measurments_critic[f"meas_for_critic_{layer_count}"])] 
                    else : 
                        return qml.probs(wires=[wire for wire in range(num_qubits_critic[f"num_for_{layer_count}"])])

                
                
                if Entanglment_critic[f"ent_for_{layer_count}"] == 'L' : 
                    weight_shapes = {'weights' : (ansatz_critic[f"ans_for_{layer_count}"] , num_qubits_critic[f"num_for_{layer_count}"])}
                else : 
                    weight_shapes = {'weights' : qml.StronglyEntanglingLayers.shape(n_layers=ansatz_critic[f"ans_for_{layer_count}"], n_wires=num_qubits_critic[f"num_for_{layer_count}"]) }
                
                
                weight_shapes_critic[f"weights_for_{layer_count}"] = weight_shapes
                Circuits_critic[f"circuit_for_{layer_count}"] = Circuit
                
                device_critic[f"device_for_{layer_count}"]  = qml.device(simulator , num_qubits_critic[f"num_for_{layer_count}"]) 

                qnode_critic[f"qnode_for_{layer_count}"] = qml.QNode(Circuits_critic[f"circuit_for_{layer_count}"] , device =device_critic[f"device_for_{layer_count}"] , interface = 'torch')
                qlayer_critic[f"{layer_count}"] = My_quantum_layer(qnode_critic[f"qnode_for_{layer_count}"], weight_shapes_critic[f"weights_for_{layer_count}"]  , number_of_measurments_critic[f"meas_for_critic_{layer_count}"] ,  kind_of_measurments_critic[f"kind_for_critic{layer_count}"]) 

                self.Critic.add_module( f"quantum_layer_{layer_count}__meas_{number_of_measurments_critic}" , qlayer_critic[f"{layer_count}"]  )

                i += 3                
                
                
        layer_count = 0 
        qnode_actor = {} 
        Circuits_actor = {}
        qlayer_actor = {}
        weight_shapes_actor = {}
        device_actor = {}
        num_qubits_actor  = {}
        Entanglment_actro = {}
        ansatz_actor = {}
        kind_of_measurments_actor = {}
        number_of_measurments_actor = {}
        
        self.Actor = nn.Sequential()

        for i in range(len(DNA)) : 
            if DNA[i] == 'C' : 
                layer_count += 1
                if layer_count == 1 :
                    self.Actor.add_module(f"linear {layer_count} outputs:{DNA[i+1]}", Layer_Ortho_inti(torch.nn.Linear(the_input_of_theFIERSTlayer, DNA[i+1]) ) )
                else :
                    if 'C' in DNA[i-3:i] :
                        self.Actor.add_module(f"linear {layer_count} outputs:{DNA[i+1]} ", Layer_Ortho_inti(torch.nn.Linear(DNA[i-2], DNA[i+1]) ) )
                    else : 
                        self.Actor.add_module(f"linear {layer_count} outputs:{DNA[i+1]}", Layer_Ortho_inti(torch.nn.Linear(2**DNA[i-3], DNA[i+1]) ) )
                i += 1
            
            
            elif DNA[i] == 'R' : 
                layer_count += 1
                self.Actor.add_module(f"ReLu_{layer_count}", nn.ReLU())
            elif DNA[i] == 'T' :
                layer_count += 1
                self.Actor.add_module(f"Tanh_{layer_count}", nn.Tanh())
            elif DNA[i] == 'Q' : 
                layer_count += 1

                num_qubits_actor[f"num_for_{layer_count}"] = DNA[i+1]
                Entanglment_actro[f"ent_for_{layer_count}"] = DNA[i+2]
                if Entanglment_actro[f"ent_for_{layer_count}"] == 'O' : 
                    Entanglment_actro[f"ent_for_{layer_count}"] = random.choice(['F' , 'L']) 
                ansatz_actor[f"ans_for_{layer_count}"] = DNA[i+3]
                
                
                if len(DNA)-1 > i+4:
                    
                    if DNA[i+4] == "C" :
                        kind_of_measurments_actor[f"kind_for_actor{layer_count}"]  = 'Prob'
                        number_of_measurments_actor[f"meas_for_actor_{layer_count}"] = 2**num_qubits_actor[f"num_for_{layer_count}"]
                    
                    else :  #the only case when we actually need the modification of qiskit , when Q layer after it a Q layer 
                        if 2**num_qubits_actor[f"num_for_{layer_count}"] ==  DNA[i+5]  : 
                            kind_of_measurments_actor[f"kind_for_actor{layer_count}"]  = 'Prob'
                            number_of_measurments_actor[f"meas_for_actor_{layer_count}"] = DNA[i+5]
                        else : 
                            kind_of_measurments_actor[f"kind_for_actor{layer_count}"] = 'Expval'
                            number_of_measurments_actor[f"meas_for_actor_{layer_count}"]= DNA[i+5]
                    
                else : #the last layer , its a quantum layer 
                    kind_of_measurments_actor[f"kind_for_actor{layer_count}"]  = 'Expval'
                    number_of_measurments_actor[f"meas_for_actor_{layer_count}"]= 2
                                
                def Circuit(inputs , weights):
                    self.internal_layer_index_actor =(self.internal_layer_index_actor+ 1) % self.counts  
                    layer_count = self.index_of_quantum[self.internal_layer_index_actor - 1 ]
                    print("Input shape:", inputs.shape if hasattr(inputs, 'shape') else inputs)
                    qml.templates.AngleEmbedding(inputs, wires = range(num_qubits_actor[f"num_for_{layer_count}"]) )
            
                    if Entanglment_actro[f"ent_for_{layer_count}"] == 'L' : 
                        qml.templates.BasicEntanglerLayers(weights , wires=range(num_qubits_actor[f"num_for_{layer_count}"]) )
                    else :  
                        qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits_actor[f"num_for_{layer_count}"]))
                        
                    if kind_of_measurments_actor[f"kind_for_actor{layer_count}"] == 'Expval':
                        return [qml.expval(qml.Z(qubit)) for qubit in range( number_of_measurments_actor[f"meas_for_actor_{layer_count}"])] 
                    else : 
                       # print("iam heeeeeeere " , num_qubits_actor[f"num_for_{layer_count}"] , layer_count , num_qubits_actor)
                        return qml.probs(wires=[wire for wire in range(num_qubits_actor[f"num_for_{layer_count}"])])

                
                
                if Entanglment_actro[f"ent_for_{layer_count}"] == 'L' : 
                    weight_shapes = {'weights' : (ansatz_actor[f"ans_for_{layer_count}"] , num_qubits_actor[f"num_for_{layer_count}"])}
                else : 
                    weight_shapes = {'weights' : qml.StronglyEntanglingLayers.shape(n_layers=ansatz_actor[f"ans_for_{layer_count}"], n_wires=num_qubits_actor[f"num_for_{layer_count}"]) }
                
                
                weight_shapes_actor[f"weights_for_{layer_count}"] = weight_shapes
                Circuits_actor[f"circuit_for_{layer_count}"] = Circuit
                
                device_actor[f"device_for_{layer_count}"]  = qml.device(simulator , num_qubits_actor[f"num_for_{layer_count}"]) 

                qnode_actor[f"qnode_for_{layer_count}"] = qml.QNode(Circuits_actor[f"circuit_for_{layer_count}"] , device =device_actor[f"device_for_{layer_count}"] , interface = 'torch')
                qlayer_actor[f"{layer_count}"] = My_quantum_layer(qnode_actor[f"qnode_for_{layer_count}"], weight_shapes_actor[f"weights_for_{layer_count}"] ,number_of_measurments_actor[f"meas_for_actor_{layer_count}"] ,   kind_of_measurments_actor[f"kind_for_actor{layer_count}"]) 

                self.Actor.add_module( f"Quantum layer {layer_count} -- meas {number_of_measurments_actor}" , qlayer_actor[f"{layer_count}"])

                i += 3                
        if DNA[len(DNA) - 2 ] == 'C' : 
            self.Critic[-1]  = Layer_Ortho_inti(nn.Linear(self.Critic[-1].in_features, 1))
            self.Actor[ -1]  = Layer_Ortho_inti(nn.Linear(self.Actor[-1].in_features, 2))
        
        
        #self.Critic.add_module("leaky", MultLayer() ) 
        #self.to('cuda')
    def get_value(self,observation):    
        return self.Critic(observation)
    
    def get_action(self, observation) : 
        logits = self.Actor(observation) 
        probs = Categorical(logits= logits)
        return probs.sample()

    def get_action_and_value(self, observation , action = None ):
        logits = self.Actor(observation) 
        probs = Categorical(logits= logits)
        if action == None : 
            action = probs.sample()
        return action , probs.log_prob(action) , probs.entropy() , self.Critic(observation)