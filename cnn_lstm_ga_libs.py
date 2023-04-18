from random import choice, sample
from random import uniform
from numpy.random import randint
from numpy import array
from sklearn.metrics import mean_squared_error
from math import sqrt

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from keras import layers
from keras import models
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
from datetime import datetime
import pandas as pd
from matplotlib import pyplot
import numpy as np


def log_message(message):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S - "), message)


# =========================================================================== #
#                             DATABASE PREPARATION                            #
# =========================================================================== #

def split_sequence(sequence, n_steps,pred_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-pred_steps:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+pred_steps]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def dataprep(data_train, data_valid, data_test, N, M):
    x_train, y_train = split_sequence(data_train, N, M)
    train_mean_x = np.mean(x_train)
    train_std_x = np.std(x_train)
    train_mean_y = np.mean(y_train)
    train_std_y = np.std(y_train)
    x_train = (x_train - train_mean_x)/train_std_x
    y_train = (y_train - train_mean_y)/train_std_y
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    
    # pyplot.plot(x_train[0])
    # pyplot.legend(['x_train_antes', 'x_train_depois'], loc='upper left')
    # pyplot.show()

    x_valid, y_valid = split_sequence(data_valid, N, M)
    x_valid = (x_valid - train_mean_x)/train_std_x
    y_valid = (y_valid - train_mean_y)/train_std_y
    x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1))

    x_test, y_test = split_sequence(data_test, N, M)
    x_test2 = (x_test - train_mean_x)/train_std_x
    y_test2 = (y_test - train_mean_y)/train_std_y
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    x_test2 = x_test2.reshape((x_test2.shape[0], x_test2.shape[1], 1))

    return x_train, y_train, x_valid, y_valid, x_test, y_test, x_test2, y_test2, train_std_x, train_mean_x, train_std_y, train_mean_y


# =========================================================================== #
#                 GENETIC ALGORITHMS - AUXILIARY FUNCTIONS                    #
# =========================================================================== #

def gene_filter(f=[2,4,8,16,32,64,128,256,384,512]):
    return choice(f)


def gene_kernel():
    return choice([1,2,3,4,5])


def gene_actfn():
    return choice(["relu", "selu", "elu"])


def gene_dropout():
    return choice([0, 0.001, 0.1])


def gene_optimizer():
    return choice(["Adamax", "Adadelta", "Adam", "Adagrad", "Ftrl", "Nadam", "RMSprop", "SGD"])


def gene_epochs():
    return randint(5, 700)


def gene_n():
    return randint(1, 200)


def gene(type):
    if type == 'f1':
        return gene_filter()
    elif type == 'f2':
        return gene_filter()
    elif type == 'f3':
        return gene_filter()
    elif type == 'f4':
        return gene_filter()
    elif type == 'k':
        return gene_kernel()
    elif type == 'a1':
        return gene_actfn()
    elif type == 'a2':
        return gene_actfn()
    elif type == 'a3':
        return gene_actfn()
    elif type == 'a4':
        return gene_actfn()
    elif type == 'd1':
        return gene_dropout()
    elif type == 'd2':
        return gene_dropout()
    elif type == 'd3':
        return gene_dropout()
    elif type == 'd4':
        return gene_dropout()
    elif type == 'op':
        return gene_optimizer()
    elif type == 'ep':
        return gene_epochs()
    elif type == 'n':
        return gene_n()
    elif type == 'rmse':
        return None
    elif type == 'loss':
        return None
    elif type == 'type':
        return None
    else:
        print('No valid type specified.')
        return None


def initialization(chromosome_set):
    parameters = {}
    for g in chromosome_set:
        parameters[g] = gene(g)

    return parameters


def generate_population(chromosome_set, n):
    population = []
    for i in range(n):
        individual = initialization(chromosome_set)
        population.append(individual)

    return population


def find_cnn(df, chromosome):
    for index, row in df.iterrows():
        if row['f1'] == chromosome['f1'] and\
           row['f3'] == chromosome['f3'] and\
           row['f4'] == chromosome['f4'] and\
           row['k']  == chromosome['k']  and\
           row['a1'] == chromosome['a1'] and\
           row['a4'] == chromosome['a4'] and\
           row['d1'] == chromosome['d1'] and\
           row['d3'] == chromosome['d3'] and\
           row['d4'] == chromosome['d4'] and\
           row['op'] == chromosome['op'] and\
           row['ep'] == chromosome['ep'] and\
           row['n']  == chromosome['n']:
            return row['rmse'], row['loss']
    return -1


def CNN_model(chromosome, M, n_features, x_train, y_train, x_valid, y_valid, batch_size):
    model = models.Sequential()

    # Layer Block 1 - CNN
    model.add(
        layers.Conv1D(
            filters = chromosome['f1'], 
            kernel_size = chromosome['k'],
            activation = chromosome['a1'],
            input_shape = (
                chromosome['n'],
                n_features
            )
        )
    )
    model.add(layers.Dropout(rate = chromosome['d1']))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Layer Block 3 - LSTM
    model.add(layers.LSTM(units = chromosome['f3']))
    model.add(layers.Dropout(rate = chromosome['d3']))

    # Layer Block 4 - DENSE
    model.add(layers.Dense(units = chromosome['f4'], 
                            activation = chromosome['a4']))
    model.add(layers.Dropout(rate = chromosome['d4']))
    model.add(layers.Dense(M))

    # Model Compilation
    model.compile(loss = "mse", optimizer = chromosome['op'])
    es = EarlyStopping(monitor="val_loss", patience = 40)

    history = model.fit(
        x_train, y_train, 
        batch_size=batch_size, 
        epochs=chromosome['ep'], verbose=0, 
        validation_data=(x_valid, y_valid), 
        callbacks=[es]
    )

    return model


# Fitness evaluation metric: Classification Accuracy 
def fitness_evaluation(model, x_test, y_test, x_test2, y_test2, train_std_x, train_mean_x, train_std_y, train_mean_y):
    metrics = model.evaluate(x_test2, y_test2, verbose=0)
    output = model.predict(x_test2)
    output = output*train_std_y + train_mean_y

    #pyplot.plot(y_test)
    #pyplot.plot(output)
    #pyplot.title('output and y_test - rmse=')
    #pyplot.suptitle(sqrt(mean_squared_error(y_test, output)))
    #pyplot.ylabel('value')
    #pyplot.xlabel('epoch')
    #pyplot.legend(['y_test', 'output'], loc='upper left')
    #pyplot.show()

    return metrics, output


def assess_chromosome(chromosome, M, n_features, 
                       data_train, data_valid, data_test, batch_size):
    try:                
        x_train, y_train, x_valid, y_valid, x_test, y_test, x_test2, y_test2, train_std_x, train_mean_x, train_std_y, train_mean_y = dataprep(
            data_train, 
            data_valid, 
            data_test,
            chromosome["n"], 
            M
        )

        model = CNN_model(
            chromosome,
            M,
            n_features, 
            x_train, y_train, 
            x_valid, y_valid,
            batch_size
        )

        # Model trained!
        loss, output = fitness_evaluation(model, x_test, y_test, x_test2, y_test2, train_std_x, train_mean_x, train_std_y, train_mean_y)
        rmse = sqrt(mean_squared_error(y_test, output))
    except:
        # Model is not possible
        loss=9999999
        rmse=9999999
        model=None
    
    return rmse, loss, model


# =========================================================================== #
#                            GENETIC OPERATORS                                #
# =========================================================================== #


def elitism_selection(population, fitness, prop=0.05):
    n_elit = int(max(1,round(prop * len(population),0)))
    pop_elit_index = sorted(range(len(fitness)), key=lambda k: fitness[k])[:n_elit]
    
    pop_elit = [population[i] for i in pop_elit_index]
    fit_elit = [fitness[i] for i in pop_elit_index]
    
    return pop_elit, fit_elit


def selection_tournament(population, avaliacao, n=2):
    npop = len(population)
    new_population = []
    new_fitness = []
    
    for p in range(npop):
        participantes = list(randint(0,len(population),n))
        
        tour_aval = []
        for part in participantes:
            tour_aval.append(avaliacao[part])

        # Selects the winner
        tour_winner = tour_aval.index(min(tour_aval))
        new_population.append(population[participantes[tour_winner]])
        new_fitness.append(avaliacao[participantes[tour_winner]])
        
    return new_population, new_fitness


def crossover_uniform(parent1, parent2, prob_cross):
    child1 = {}
    child2 = {}

    num_genes_changed = 0
    # Iterates over each gene
    for k in parent1:
        runi = uniform(0,1)
        if runi <= prob_cross :
            child1[k] = parent2[k]
            child2[k] = parent1[k]
            num_genes_changed+=1
        else:
            child1[k] = parent1[k]
            child2[k] = parent2[k]
    if num_genes_changed > 0:
        new_childs = True
    else:
        new_childs = False
    return child1, child2, new_childs


def population_crossover(population, fitness, prob_cross=0.5):

    new_population = []
    new_fitness = []
    
    # Select order to perform crossover
    pop_ind = sample(range(0,len(population)), 
                     k=len(population))
    
    if len(population) % 2 == 0:
        limit = len(population)//2
    else:
        limit = len(population)//2+1

    for i in range(0,limit):
        child1, child2, new_childs = crossover_uniform(
            population[pop_ind[i*2]],
            population[pop_ind[i*2+1]],
            prob_cross
        )
        
        # Adds the new childs to population
        new_population+=[child1, child2]

        # Adds the fitness
        if new_childs:
            new_fitness+=[None,None]
        else:
            new_fitness+=[fitness[pop_ind[i*2]],
                          fitness[pop_ind[i*2+1]]]

    return new_population, new_fitness


def mutation(chromosome, prob_mut=0.05):
    
    for selected_gene in chromosome:
        # Mutate if sampled value is lower than 'prob'
        gene_mutate = uniform(0,1)
        if gene_mutate <= prob_mut:
            curr_gene = chromosome[selected_gene]
            new_gene = curr_gene
            while new_gene == curr_gene:
                new_gene = gene(selected_gene)
            new_chromosome = chromosome.copy()
            new_chromosome[selected_gene] = new_gene
            return new_chromosome, True
        else:
            return chromosome, False


def population_mutation(population, fitness, prob_mut=0.05):
    new_population = []
    new_fitness = []
    
    for i, p in enumerate(population):
        p_mut, mut = mutation(p, prob_mut=prob_mut)
        new_population.append(p_mut)
        if mut:
            new_fitness.append(None)
        else:
            new_fitness.append(fitness[i])
    
    return new_population, new_fitness


def elitism_back(population, population_fitness, pop_elit, fit_elit):
    
    # Number of chromosomes to keep
    pop_nkeep = len(population) - len(pop_elit)
    
    # Select chromosomes with lower RMSE
    pop_pd = pd.DataFrame(population)
    pop_pd['rmse'] = population_fitness
    pop_keep = pop_pd.sort_values('rmse', 
                                  ascending=True,
                                  na_position='first').head(pop_nkeep)
    
    # Create DataFrame of Elite to insert
    pop_insert = pd.DataFrame(pop_elit)
    pop_insert['rmse'] = fit_elit

    # Unify chromosomes
    pop_new_pd = pd.concat([pop_keep, pop_insert])
    fit_new = pop_new_pd['rmse'].tolist()
    pop_new = pop_new_pd.to_dict(orient='records')
        
    return pop_new, fit_new