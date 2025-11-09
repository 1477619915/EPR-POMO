# EPR (Elite-Pattern Reinforcement)

This repository provides the implementation of **EPR (Elite-Pattern Reinforcement)** based on various POMO variants.

---
## 🚀 Descriptions of the roles and purposes of individual files in the project
1. **CVRPEnv.py or TSPEnv.py**
   This file defines the CVRPEnv or TSPEnv class, which simulates the environment for solving the CVRP or TSP using reinforcement learning. It handles loading problem instances, maintaining state transitions, computing rewards, and providing features for policy learning.
2. **CVRPModel.py or TSPModel.py**
   This file defines a neural network model for solving the CVRP or TSP using attention-based encoder-decoder architectures,which uses a learned decoder.
3. **generate_data.py**
   This file is used to generate, save, and load datasets with different instance distributions (uniform, clustered, or mixed). It supports both synthetic data generation and loading from preprocessed `.pkl` files, enabling flexible dataset creation for validation and testing. 
4. **models.py**
   This file defines the detailed architectures of the encoder and decoder, as well as how to use them to solve problem instances.
5. **ultis.py**
   This file mainly includes the following functions:
   rollout: Executes the model's inference process in the environment, generating complete solutions and their corresponding probabilities, used for training and evaluation. split_solutions_by_rewards: Splits solutions into superior and inferior routes based on their rewards. hgs_solution: Calls the HGS (Hybrid Genetic Search) solver to obtain approximate optimal solutions for the CVRP. compute_cost_difference Compute the differences between superior, inferior, and elite solutions, and incorporate them into the loss function.

## 🚀 Train EPR-POMO on CVRP or TSP

1. **Install dependencies**  
   Install `hgyese`, or `lkh`:
   ```bash
   pip install hgyese or lkh
2. **Generate the validation sets:**
   ```bash
   python generate_data.py
3. **Modify the load_checkpoint config in config.yml**
4. **run train**
   Since heuristic methods are used to generate elite solutions, a hgs_costs.pt file will be generated.
   ```bash
   python train.py

## 🚀 Test EPR-POMO on CVRP or TSP

1. **Under the CVRP/TSP folder, use the default settings in config.yml, run**
   Since heuristic methods are used to generate elite solutions, a hgs_costs.pt file will be generated.
   ```bash
   python test_cvrplib.py
   python test_vrplib.py
   
## 🚀 Test EPR on POMO、Omni-POMO、Sym-POMO、ELG-POMO
1. **We provide EPR implementations based on various POMO variants. Since the data files are large, you will need to download them yourself from their official repositories. The links are provided below:**
POMO：https://github.com/yd-kwon/POMO
Omni-POMO：https://github.com/yd-kwon/POMO
Sym-POMO：https://github.com/alstn12088/Sym-NCO
ELG-POMO：https://github.com/lamda-bbo/ELG
