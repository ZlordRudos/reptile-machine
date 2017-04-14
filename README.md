Introduction

This is implementation of neural turing machine (NTM) introduced in [Neural Turing Machines]https://arxiv.org/abs/1410.5401 paper. NTM is a recursive network with explicitly designed memory and read/write operations. It is differentiable, so it is learnable by gradient descent. Like usual turing machines it contains (in high-level view) head(s), (finite) memory tape and head(s) controller. Unlike classic turing machines heads can jump according to given key (emulating random-access memory), and it is "fuzzy": head reads superposition of tape cells, memory cell "symbols" are real numbers etc..

Since the mentioned paper doesn't give details about implementation it is open to many interpretations.

Implementation

Think of it as 3 layer architecture:

functions - In chain_functions directory. Functions described in NTM paper.

heads - Classes in ntm_heads. Head perform actions in this order WRITE, MOVE, READ (if it is just read head, exclude WRITE and vice versa). Parameters of these actions are given in control vector. WRITE and READ actions are performed on memory tape, while MOVE changes its weighting vector which defines (super)position on memory tape. Weighting vector is stored in head-object so it propagates through time.

wrapper - In ntm_wrapper. Wrapper contains controler, heads and memory. 
Controler can be basically any neural network. Its input is: output from previous layer and readouts from read-heads. Output: output vector to next layer and control vectors for heads. It is worth mentioning that controler's input and output sizes are not depended on memory size.
Wrapper routes control vectors from controler to heads and let those heads perform their actions on memory in defined order. Wrapper object stores the memory tape so it propagates through time. Wrapper also collects outputs from read-heads and sends them recurently to controler input, along with input from previous layer. Wrapper also relays controler output to next layer.


![dataflow](/images/ntm_flow.pdf)
