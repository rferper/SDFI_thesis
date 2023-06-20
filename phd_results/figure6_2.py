import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import SDFI.base.fuzzy_data as fuzzy_data
import SDFI.base.operators.implications as implications
import SDFI.base.operators.tnorms as tnorms
import SDFI.methods.STFI as STFI

if __name__ == '__main__':
    dataset = pd.read_csv('../assets/sd_startups.csv', sep=',')
    Implication = implications.ImplicationsExamples.get_fuzzy_implication(implications.ImplicationsExamples.GODEL)
    Tnorm = tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.MINIMUM)
    fuzzy_dataset = fuzzy_data.FuzzyDataUniformTriangular('wpbc', dataset, 3, ['L', 'M', 'H'])
    coverages = [x / 100 for x in range(0, 16, 1)]
    sharp = [0] * len(coverages)
    sharp_time = [0] * len(coverages)
    number_transitions = [0]*len(coverages)
    for i in range(len(coverages)):
        cov = coverages[i]
        print('Min Coverage: {}'.format(cov))
        st = time.time()
        jumps = STFI.STFI(dataset, fuzzy_dataset, Tnorm, Implication, max_features=5, min_coverage=cov)
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
        sharp[i] = jumps.sharpness.mean()
        sharp_time[i] = elapsed_time
        number_transitions[i] = len(jumps.sharpness)
        print(sharp)

    # plots

    # STFI
    fig1, ax1 = plt.subplots()
    ax1.plot(coverages, sharp, color="orange")
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('sharpness')
    fig1.savefig('mincov_sharp_fwracc.png')

    fig1, ax1 = plt.subplots()
    ax1.plot(coverages, sharp_time, color ="orange")
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('ctime')
    fig1.savefig('mincov_sharp_ctime.png')

    fig1, ax1 = plt.subplots()
    ax1.plot(coverages, number_transitions, color ="orange")
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('transitions')
    fig1.savefig('mincov_sharp_ntransitions.png')