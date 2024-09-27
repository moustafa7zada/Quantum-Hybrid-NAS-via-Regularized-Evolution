Its known from the paper https://arxiv.org/abs/2007.04965 by abacus.ai
that the best enconding scheme for a model with aging/regularized evolution is the adjacent matrix,
but since the paper that we are working on (chen et al. 2023) is more layers oriented and the problem itself which is fairly simple and the 
nature of the hybrid quantum classical net that they used ,I chose the regular simple DNA encoding 

WHERE 
for example , DNA = ['C',4,'R','Q',4,'L',2]
Classical layer that has 4 neurons, then, the activation functions of the layer (ReLu), then, Q, a Quantum layer, which has 4 qubits
with linear entanglement scheme used in the ansatz , which have its repetitions set to 2 (what comes ater L)


WHERE
C = A Classical Layer , a fully connected linear torch layer 
Q = A Quantum Layer, a Parametrized Quantum Circuit 
R = ReLu Activation Function  
T = Tanh Activation Function
L = Linear entanglement in the ansatz
F = Full entanglement in the ansatz
O = Circular entanglement in the ansatz 



possible layers = Quantum and Classical(torch linear)

possible activations = ReLu and Tanh

possible entanglement = full , linear and circular 

possible repetitions of the ansatz = 1 , 2 and 3




perforcmence comaperison : 
lightning.qubit = 36 seconds 
default.qubit = 16 
