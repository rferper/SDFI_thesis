import pandas as pd
import numpy as np
import time
import SDFI.base.operators.implications as implications
import SDFI.base.operators.tnorms as tnorms
import SDFI.base.fuzzy_data as fuzzy_data
import SDFI.methods.STFI as STFI
import random

if __name__ == '__main__':
    datasets = [
        'sd_wpbc.csv',
        'sd_usa2012.csv',
        'sd_startups.csv',
        'sd_electricity.csv',
        'sd_stockprices.csv',
        'sd_treasury.csv',
        'sd_calhousing.csv',
        'sd_metrotraffic.csv'
    ]

    I = implications.ImplicationsExamples.get_fuzzy_implication(implications.ImplicationsExamples.GOGUEN)
    T = tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.PRODUCT)
    results = [['dataset', 'computation_time']]
    for idx_dat in range(len(datasets)):
        dataset = pd.read_csv('../assets/' + datasets[idx_dat], sep=',')
        random.seed(2021)
        fuzzy_dataset = fuzzy_data.FuzzyDataFCM(datasets[idx_dat], dataset, 3)
        st = time.time()
        jumps = STFI.STFI(dataset, fuzzy_dataset, T, I, max_features=5, min_coverage=0.1)
        et = time.time()
        elapsed_time = et - st
        results = results + [[datasets[idx_dat],elapsed_time]]
    np.savetxt("ResultsComputationTimes2.csv",
               results,
               delimiter=", ",  # Set the delimiter as a comma followed by a space
               fmt='% s')  # Set the format of the data as string