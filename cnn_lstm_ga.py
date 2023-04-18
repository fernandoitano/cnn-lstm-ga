# -*- coding: utf-8 -*-
"""
Original article code:
'CNN-LSTM Optimized by Genetic Algorithm in Time Series Forecasting: An Automatic Method to use Deep Learning'
"""

# General Packages
import os
from datetime import datetime
from random import choice
from random import uniform
from math import sqrt

# Data Preparation Packages
import numpy as np
from numpy import genfromtxt
from numpy import array
from numpy.random import randint
from matplotlib import pyplot
import pandas as pd

# Neural Networks Packages
import tensorflow as tf
from keras import layers
from keras import models
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# Genetic Algorithm Developed Package
import cnn_lstm_ga_libs as ga


# =========================================================================== #
#                                CNN-LSTM-GA                                  #
# =========================================================================== #

def main():

    # General Defitions
    ga.log_message('Loading General Definitions')
    batch_size = 128
    n_features = 1
    M = 1
    
    exec_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir('cnn_models_' + exec_timestamp)


    # Data Loading and Preparation
    dataset = 'USmonthlysales.csv'
    file = f'data/{dataset}'
    ga.log_message(f'Loading DataSet: {file}')
    data = genfromtxt(file, delimiter=',')
    data = data[1:len(data)-1,4]

    train_factor = 0.7
    test_factor = 0.9

    data_train = data[:int(round(len(data)*train_factor))]
    data_valid = data[int(round(len(data)*train_factor)):int(round(len(data)*test_factor))]
    data_test = data[int(round(len(data)*test_factor)):]


    # Genetic Algorithm Definitions
    ga.log_message('Loading Genetic Algorithm Definitions')
    generations = 20
    num_pop = 50
    chromosome_set = ['f1','f3','f4',
                      'k',
                      'a1','a4',
                      'd1','d3','d4',
                      'op','ep','n',
                      'rmse', 'loss', 'type']


    # Initial Population Generation
    ga.log_message('Generating Initial Population')
    population = ga.generate_population(chromosome_set, num_pop)
    ga_evolution_pd = pd.DataFrame(columns=['generation'] + chromosome_set)

    # Evolutionary Process
    ga.log_message('Starting evolutionary process')
    for generation in range(generations):
        ga_evolution_lst = []
        population_rmse = []
        population_loss = []
        population_type = []
        for i, chromosome in enumerate(population):
            # Search chromosome in population history
            existing_chromosome = ga.find_cnn(ga_evolution_pd, chromosome) #use existing chromosome already trained
            if isinstance(existing_chromosome, tuple):
                rmse = existing_chromosome[0]
                loss = existing_chromosome[1]
                cnn_type = 'existing cnn'
                ga.log_message(('Generation ' + str(generation).zfill(2) + ' - Chromosome '+ str(i).zfill(2) + ': CNN already trained'))
            else:
                rmse, loss, model = ga.assess_chromosome(
                    chromosome,
                    M, n_features,
                    data_train, data_valid,
                    data_test, batch_size
                )
                cnn_type = 'new cnn'
                if model != None:
                    model_name = 'cnn_models_' + exec_timestamp + '/model_gen_' + str(generation).zfill(2) + '_cnn_' + str(i).zfill(2) + '.h5'
                    model.save(model_name)
                ga.log_message(('Generation ' + str(generation).zfill(2) + ' - Chromosome '+ str(i).zfill(2) + ': CNN new'))

            population_rmse.append(rmse)
            population_loss.append(loss)
            population_type.append(cnn_type)


            # Store evolution data by chromosome
            new_chrom_dict = {
                'dataset': dataset,
                'generation': int(generation),
                'f1': chromosome["f1"],
                'f3': chromosome["f3"],
                'f4': chromosome["f4"],
                'k' : chromosome["k"],
                'a1': chromosome["a1"],
                'a4': chromosome["a4"], 
                'd1': chromosome["d1"],
                'd3': chromosome["d3"],
                'd4': chromosome["d4"],
                'op': chromosome["op"],
                'ep': chromosome["ep"],
                'n' : chromosome["n"],
                'rmse': rmse,
                'loss': loss,
                'type': cnn_type
            }
            ga_evolution_pd = pd.concat(
                [
                    ga_evolution_pd,
                    pd.DataFrame(new_chrom_dict, index=[0])
                ]
            ) 

        # Generation Outcome
        ga.log_message('Generation ' + str(generation).zfill(2) + ' Outcome: ' + str(ga_evolution_pd['rmse'].min()))

        # Store evolution data by the end of generation
        ga.log_message('Generation ' + str(generation).zfill(2) + ' Data Stored' + '\n')
        ga_evolution_pd.to_csv('cnn_models_' + exec_timestamp + '/ga_evolution_' + exec_timestamp + '.csv', mode='w', index=False)

        # ELITISM: Best individuals selection
        ga.log_message('Genetic Operator: Elitism Selection Operator')
        pop_elit, fit_elit = ga.elitism_selection(population, population_rmse)

        # SELECTION: TOURNAMENT METHOD
        ga.log_message('Genetic Operator: Selection Operator')
        population, population_rmse = ga.selection_tournament(population, 
                                                              population_rmse, 
                                                              n=2)

        # CROSSOVER: UNIFORM METHOD
        ga.log_message('Genetic Operator: Crossover Operator')
        population, population_rmse = ga.population_crossover(population, 
                                                              population_rmse,
                                                              prob_cross=0.5)

        # MUTATION
        ga.log_message('Genetic Operator: Mutation Operator')
        population, population_rmse = ga.population_mutation(population, 
                                                             population_rmse, 
                                                             prob_mut=0.10)

        # ELITISM: Put back the best individuals
        ga.log_message('Genetic Operator: Elitism Back')
        population, population_rmse = ga.elitism_back(population, population_rmse, 
                                                      pop_elit, fit_elit)


    ga.log_message('Evolutionary process completed!')
    
if __name__ == "__main__":
    for i in range(1,2):
        try:
            main()
        except Exception as e:
            print(f'Erro no processo evolutivo: {e}')
			