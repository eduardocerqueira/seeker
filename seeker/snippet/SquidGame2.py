#date: 2021-10-29T16:43:13Z
#url: https://api.github.com/gists/af57e85666f5b58b5027e974d8e4aa71
#owner: https://api.github.com/users/GDLMadushanka

from matplotlib import pyplot as plt
import numpy as np
from random import randrange
from qiskit import *
from qiskit.circuit.library.standard_gates import XGate

numPairs = 1000
winCount = 0

for pair in range(numPairs):
    # Creating the circuit with 4 classical bits and 4 qbits
    qc = QuantumCircuit(4,4)
    # Creating the entanglement between players qubits
    qc.h(1)
    qc.cx(1,0)

    # Applying X gate power raised to (-0.25)
    new_gate = XGate().power(-0.25)
    qc.append(new_gate, [0])
    qc.barrier()

    # Initialize input qubits randomly
    toss1 = randrange(10)
    if toss1 > 4:
        qc.x(2)
    toss2 = randrange(10)
    if toss2 > 4:
        qc.x(3)
    qc.barrier()

    # Players applying a controlled sqrt(x) gate to their qubits
    # taking input qubits as controllers
    qc.csx(2,0)
    qc.csx(3,1)
    qc.barrier()

    # Players measure their qubits and inform the result to guards
    qc.measure(0,0)
    qc.measure(1,1)

    # measuring the input qubits to calculate xy
    qc.measure(2,2)
    qc.measure(3,3)

    # un-comment the following line to display the circuit.
    #display(qc.draw(output='mpl'))

    # Executing the circuit in a simulator once
    result = execute(qc,Aer.get_backend('qasm_simulator'),shots=1).result()

    # Calculate the XOR of measurement results 
    for key, value in result.get_counts().items():
        # calculate xy
        xy = int(key[0]) * int(key[1])
        # calculate xor of player outputs
        xor = int(key[2], 2) ^ int(key[3], 2) 
        # count as a win of xy = xor of player outputs
        if xy == xor:
            winCount += value
print("Win probability : ", winCount/10)