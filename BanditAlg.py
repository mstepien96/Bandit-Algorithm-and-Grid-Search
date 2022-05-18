# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:12:46 2020

@author: mstep
"""

import numpy as np
import pandas as pd



# LOADING THE DATA

BloodA = pd.read_csv('BloodDataA.csv')
#print(BloodA)

BloodB = pd.read_csv('BloodDataB.csv')
#print(BloodB)

samples = len(BloodA.iloc[1])    #calculating the number of samples
#print(samples)





def BanditAlg(exp, dist, decay, rwt, w0, infile, outfile):
    """Function running Bandit Algorithm on a given dataset. It takes as
    inputs the exploration rate, the uniform distribution parameter, 
    the decay rate, the reward rate and the initial weigth (a single value).
    The infile parameter takes value 'BloodDataA' or 'BloodDataB'. The name
    of the outfile needs to be specified ('name.txt'). The file logs every
    step and saves the step number, the sample chosen, the current reward,
    the total reward, the probabilities and the weights."""
    
    
    # ASSIGNING DATABASE ACCORGING TO INFILE PARAMETER
    
    if infile == 'BloodDataA':
        BloodData = BloodA
    elif infile == 'BloodDataB':
        BloodData = BloodB
    else:
        print('Wrong data file')
    #print(BloodData)
    
    
    
    # INITIALIZING WEIGHTS AND PROBABILITIES ARRAYS
    
    weights = np.array([])
    for i in range(samples):
        weights = np.append(weights, w0)
        
    p = np.array([])
    for i in range(samples):
        p = np.append(p, 1/samples)
        
        
        
    # INITIALIZING UTILITY ARRAYS AND VARIABLES
    
    #BloodDataSmall = BloodData.iloc[0:25]
    indexes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])   #for easier iteration
    #counter = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])    #for counting how many times a sample has been chosen (for debugging)
    sampleNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']    #for logging output
    totReward = 0
    step = 0
    
    file = open(outfile, 'w+')   #opening a file for logging
    
    
    # PICKING A SAMPLE
    
    for i in range(len(BloodData)):
        indexSelected = np.random.choice(indexes, p=p)    #randomly choosing the index in a row, based on probabilities
        
        step += 1     #incremental step
        #counter[indexSelected] += 1     #updating counter
        
        r = BloodData.iloc[i][indexSelected]         #fetching reward based on the index selected
        totReward += r              #updating total reward     
        # print('Row:', i)
        # print('Index selected:', indexSelected)
        # print('Reward:', r)
        # print('Total reward:', totReward)
        # print(weights)
        weights[indexSelected] = decay * weights[indexSelected] + rwt * r   #updating the weight of 'pulled lever'
        #print(weights)
       
        
       
        # NORMALIZATION OF WEIGHTS
        
        # Shifting all values to positive
        nonZeroWeights = np.array([])
        for i in indexes:
            nonZeroWeights = np.append(nonZeroWeights, weights[i])
        
        wMin = min(nonZeroWeights)
        for i in indexes:
            nonZeroWeights[i] = nonZeroWeights[i] - wMin + 0.01
        #print('nonzero weights:', nonZeroWeights)
    
    
        # Sum of the raw weight (for denominator for norm calculation)
        rawWeightsSum = 0
        for j in indexes:
            rawWeightsSum += nonZeroWeights[j]
        #print('Raw weights sum:', rawWeightsSum)
        
        #Calculating normalized weights
        wNorm = np.array([])
        for k in indexes:
            wNorm = np.append(wNorm, nonZeroWeights[k]/rawWeightsSum)
        #print('Normalized weights:', wNorm)
        
        
        
        # CALCULATING AND NORMALIZING PROBABILITIES
        
        for m in indexes:
            p[m] = wNorm[m] * (1 - exp) + exp * dist
        #print('Probabilities:', p)
        
        pMin = min(p)
        for i in indexes:
            p[i] = p[i] - pMin + 0.01
        #print('Te jakies p:', p)
        
        sumP = 0
        for n in indexes:
            sumP += p[n]
        #print('Sum of probabilities:', sumP)
        
        for m in indexes:
            p[m] = p[m]/sumP
        #print('Normalized probabilities:', p)
        #print("\t")
        
        
        
        # LOGGING THE OUTPUT
        
        file.write('Step: %s, Choice: Sample %s, CurrReward: %s, TotReward: %s,\nProbabilities (A-J): %s,\nWeights (A-J): %s\n\n' %(step, sampleNames[indexSelected], r, totReward, p, weights))

    #print('Counter:', counter)
    file.close()
    return totReward
    
    
    
    
    
BanditAlg(0.4, 0.4, 0.4, 0.65, 1, 'BloodDataB', 'LooG.txt')