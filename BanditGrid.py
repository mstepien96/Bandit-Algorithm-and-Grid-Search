# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:43:37 2020

@author: mstep
"""

import BanditAlg as ba
import numpy as np
import pandas as pd



# LOADING THE DATA

BloodA = pd.read_csv('BloodDataA.csv')
#print(BloodA)

BloodB = pd.read_csv('BloodDataB.csv')
#print(BloodB)

#samples = len(BloodA.iloc[1])    #calculating the number of samples
#print(samples)





def BanditGrid(maxiter, step, infile, outfile):
    """Function running Grid Search algorithm for the parameters of the
    Bandit Algorithm. It takes maximal number of iterations and the step size
    as inputs. The infile parameter takes either 'BloodDataA' or 'BloodDataB'.
    The outfile's name needs to be specified as 'name.txt'. It logs every
    step and saves to the file the round number and the parameters. At the
    end it shows the best score and corresponding parameters."""
    
    
    # INITIALIZING UTILITY VARIABLES
    bestReward = 0
    currIter = 0
    range1 = np.arange(0.01, 1, step)
    
    file = open(outfile, 'w+')   #opening a file for logging
    
    
    
    # NESTED LOOPS ITERATING OVER POSSIBLE VALUES FOR PARAMETERS
    for exp in range1:
        # print('EXP:', exp)
        for dist in range1:
            # print('DIST:', dist)
            for decay in range1:
                # print('DECAY:', decay)
                for rwt in range1:
                    for w0 in [1, 2, 3, 4, 5]:
                        # w0 = np.array([])
                        # for j in range(samples):
                        #     w0 = np.append(w0, i)
                            
                        if currIter == maxiter:      # checking the condition of max iterations
                            file.write('\n** BestScore: exp: %s, dist: %s, decay: %s, rwt: %s, w0: %s, TotReward: %s **\n' %(expBest, distBest, decayBest, rwtBest, w0Best, bestReward))
                            file.close()
                            return bestReward
                        else:
                            currIter +=1
                            totalReward = ba.BanditAlg(exp, dist, decay, rwt, w0, infile, 'BanditAlgLog.txt')  #calling bandit algorithm
                            # print('rwt:', rwt)
                            # print('Total Reward:', totalReward)
                            file.write('** Start round: %s, exp: %s, dist: %s, decay: %s, rwt: %s, w0: %s **\n' %(currIter, exp, dist, decay, rwt, w0))
                            if totalReward > bestReward:   #updating best score and saving the parameters
                                bestReward = totalReward
                                expBest = exp
                                distBest = dist
                                decayBest = decay
                                rwtBest = rwt
                                w0Best = w0
                                # print('____________________')
                                # print('Best Reward:', bestReward)
                                # print('exp:', exp)
                                # print('dist:', dist)
                                # print('decay:', decay)          
                                # print('rwt:', rwt)
                                # print('____________________')
                                # print('\t')

               
    

print(BanditGrid(10000, 0.15, 'BloodDataA', 'looooogB.txt'))