#!/usr/bin/env python
# coding: utf-8

# # Genetic Algorithms

# ## Introduction
# 
# In this project, with the help of genetic algorithms, we want to find the gates needed to produce the desired logic circuit to work according to the given truth table. A genetic algorithm is an optimization or search algorithm that works essentially by mimicking the process of evolution. Genetic Algorithms have the ability to deliver a good-enough solution fast-enough. This makes genetic algorithms attractive for use in solving optimization problems.

# ## Genetic Representation
# The first step towards a genetic algorithm is to be able to represent candidate solutions genetically.
# 
# ### Chromosomes
# Each candidate solution could be termed an individual. Each individual is represented by a chromosome, that is a collection of genes. Each set of gates represents one chromosome. The total number of chromosomes constitutes the population.
# 
# ### Gene
# Genes are variables representing properties of the solution. They could be thought of as decision variables. Each gate inside a set (chromosome) represents one gene.

# Before anything we import libraries needed in the project.

# In[1]:


import random
import numpy as np
import pandas as pd

import schemdraw
import schemdraw.logic as logic
import schemdraw.parsing.logic_parser as logic_parser


# In the given file, the first nine columns are circuit inputs, and the last column is circuit output. So, in the end, a list containing nine gates should be the result as output. The way that gates are connected is like this, the first gate takes the first two inputs, and after that, each gate takes the output of the last gate and the following input of the circuit.

# In[2]:


df = pd.read_csv('../data/truth_table.csv')
df


# It should be noted that the set of gates that we deal with in this project is as follows: AND, OR, XOR, NAND, NOR, and XNOR.

# In[3]:


class TruthTable():
    def __init__(self, table):
        self.num_rows = np.size(table, 0)
        self.num_inputs = np.size(table, 1) - 1
        self.values = np.array(table.values.tolist())


# In[4]:


class LogicCircuits():
    def __init__(self, truth_table):
        self.logic_gates = {
            0: 'AND',
            1: 'OR',
            2: 'XOR',
            3: 'NAND',
            4: 'NOR',
            5: 'XNOR'
        }
        self.truth_table = truth_table
    
    def get_logic_gate_output(self, logic_gate, input1, input2):
        if logic_gate == 0:
            return np.bitwise_and(input1, input2)
        elif logic_gate == 1:
            return np.bitwise_or(input1, input2)
        elif logic_gate == 2:
            return np.bitwise_xor(input1, input2)
        elif logic_gate == 3:
            return np.bitwise_not(np.bitwise_and(input1, input2))
        elif logic_gate == 4:
            return np.bitwise_not(np.bitwise_or(input1, input2))
        elif logic_gate == 5:
            return np.bitwise_not(np.bitwise_xor(input1, input2))
        
    def cal_truth_table_satisfied_rows_number(self, logic_circuits):
        size = np.size(logic_circuits, 0)
        
        output = np.zeros([self.truth_table.num_rows, size], dtype=bool)
        out = truth_table.values[:,0]

        for j in range(size):
            for i in range(1, truth_table.num_inputs):
                out = self.get_logic_gate_output(logic_circuits[j][i-1], out, truth_table.values[:,i])

            output[:,j] = out
            output[:,j] = np.where(output[:,j] == df["Output"], True, False)

        return output.sum(axis=0)
    
    def draw_logic_circuits(self, logic_circuits, save=False):
        string = "(Input1 " + self.logic_gates[logic_circuits[0]].lower() + " Input2)"
        for i in range(len(logic_circuits) - 1):
            string = "(" + string
            string += " " + self.logic_gates[logic_circuits[i + 1]].lower() + " Input" + str(i + 3) + ")"
        print(string)
        if save:
            logic_parser.logicparse(string, gateH=2).save('logic_circuits.svg')
        return logic_parser.logicparse(string, gateH=2)


# In[5]:


class GeneticAlgorithm():
    def __init__(self, fitness_function, stopping_criteria, num_genes_in_chromosome, genes):
        self.population_size = self.get_next_power_of_two(np.power(num_genes_in_chromosome, 2)) # population_size > num_genes^2
        self.fitness_function = fitness_function
        self.stopping_criteria = stopping_criteria
        self.num_genes_in_chromosome = num_genes_in_chromosome
        self.p_crossover = 0.75 # between 0.65 and 0.85
        self.p_mutation = 0.1 # use either high crossover, low mutation (e.g. Xover = 80%, mutation = 5%), or moderate crossover, moderate mutation (e.g. Xover = 40%, mutation = 40%).
        self.genes = genes
        self.num_generations = 0
        self.best = None
    
    def get_next_power_of_two(self, n):
        if not (n & (n - 1)):
            return n
        return  int("1" + (len(bin(n)) - 2) * "0", 2)

    def initialize_population(self, genes_list, num_genes_in_chromosome):
        return np.random.choice(genes_list, size=(self.population_size, num_genes_in_chromosome))
    
    def cal_fitness(self, population):
        return self.fitness_function(population)

    def is_stopping_criteria_met(self, fitness, population):
        for index in range(self.population_size):
            if self.stopping_criteria == fitness[index]:
                print('='*64)
                print(f'\nOutput: {population[index]}')
                self.best = population[index]
                return True
        return False

    def select_parents(self, fitness, population):
        rank = np.zeros(self.population_size, dtype = int)
        prob_fitness = fitness/sum(fitness)
        index_weight = sorted(range(len(fitness)), key=lambda k: fitness[k], reverse=True)
        for i in range(self.population_size):
            rank[index_weight[i]] = self.population_size - i

        selected_parents = random.choices(population, weights=rank, k=self.population_size)
        return np.array(selected_parents)
    
    def apply_crossover(self, selected_parents):
        for i in range(0, self.population_size, 2):
            for j in range(self.num_genes_in_chromosome):
                if np.random.rand() < self.p_crossover:
                    temp = selected_parents[i][j]
                    selected_parents[i][j] = selected_parents[i + 1][j]
                    selected_parents[i + 1][j] = temp
        return selected_parents
    
    def apply_mutation(self, new_population):
        for i in range(self.population_size):
            for j in range(self.num_genes_in_chromosome):
                if np.random.rand() > self.p_mutation:
                    continue
                new_population[i][j] = np.random.randint(6)
        return new_population
    
    def perform_evolution_cycle(self, population, fitness_scores, log):
        selected_parents = self.select_parents(fitness_scores, population)
        new_population = self.apply_crossover(selected_parents)
        new_population = self.apply_mutation(new_population)

        population = new_population
        fitness_scores = self.cal_fitness(population)
        
        if log:
            print('='*64)
            print(f'Generation: {self.num_generations}\t Avg. Fitness Score: {np.average(fitness_scores)}')

        self.num_generations += 1
        self.p_mutation *= 0.99
        
        return population, fitness_scores
            
    def begin(self, log=False):
        self.num_generations = 0
        population = self.initialize_population(genes, self.num_genes_in_chromosome)
        
        fitness_scores = self.cal_fitness(population)
        
        while not self.is_stopping_criteria_met(fitness_scores, population):
            population, fitness_scores = self.perform_evolution_cycle(population, fitness_scores, log)
            

# The whole algorithm can be summarized as –  

# 1) Randomly initialize populations p
# 2) Determine fitness of population
# 3) Until convergence repeat:
#       a) Select parents from population
#       b) Crossover and generate new population
#       c) Perform mutation on new population
#       d) Calculate fitness for new population



# GA()
#    initialize population
#    find fitness of population
   
#    while (termination criteria is reached) do
#       parent selection
#       crossover with probability pc
#       mutation with probability pm
#       decode and fitness calculation
#       survivor selection
#       find best
#    return best



# In[6]:


truth_table = TruthTable(df)
logic_circuits = LogicCircuits(truth_table)

fitness_function = logic_circuits.cal_truth_table_satisfied_rows_number
stopping_criteria = logic_circuits.truth_table.num_rows
num_genes_in_chromosome = logic_circuits.truth_table.num_inputs - 1
genes = list(logic_circuits.logic_gates.keys())

ga = GeneticAlgorithm(fitness_function, stopping_criteria, num_genes_in_chromosome, genes)
ga.begin(True)


# In[7]:


logic_circuits.draw_logic_circuits(ga.best)

