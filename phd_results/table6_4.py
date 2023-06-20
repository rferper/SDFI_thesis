import pandas as pd
import numpy as np
import SDFI.base.operators.implications as implications
import SDFI.base.operators.tnorms as tnorms
import SDFI.base.fuzzy_data as fuzzy_data
import SDFI.methods.SDFIOE as SDFIOE
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

    partitions = ['uniform'] # Choose the partitions to test: uniform, cmeans or fcm
    n_fs = [3] # Choose the number of fuzzy seats to build for each partition
    impl_operators = [implications.ImplicationsExamples.get_fuzzy_implication(implications.ImplicationsExamples.GODEL),
                      implications.ImplicationsExamples.get_fuzzy_implication(implications.ImplicationsExamples.GOGUEN),
                      lambda x, y: implications.ImplicationsExamples.get_fuzzy_implication(
                          implications.ImplicationsExamples.FGM)(x, y, float(0.5)),
                      lambda x, y: implications.ImplicationsExamples.get_fuzzy_implication(
                          implications.ImplicationsExamples.KSS)(x, y, float(-1)),
                      lambda x, y: implications.ImplicationsExamples.get_fuzzy_implication(
                          implications.ImplicationsExamples.KH)(x, y, float(2)),
                      lambda x, y: implications.ImplicationsExamples.get_fuzzy_implication(
                          implications.ImplicationsExamples.KF)(x, y, float(2))]
    tnorms_operators = [tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.MINIMUM),
                        tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.PRODUCT),
                        tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.PRODUCT),
                        lambda x, y: tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.SCHWEIZER_SKLAR)(x, y,
                                                                                                            float(-1)),
                        lambda x, y: tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.HAMACHER)(x, y, float(2)),
                        lambda x, y: tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.FRANK)(x, y, float(2))]

    results = [['dataset', 'partition', 'n_fs', 'operator', 'target_index', 'mfwracc']]
    for idx_part in range(len(partitions)):
        partition = partitions[idx_part]
        for idx_fs in range(len(n_fs)):
            for idx_op in range(6):
                datasets_fwracc = [0] * len(datasets)
                for idx_dat in range(len(datasets)):
                    random.seed(2021)
                    print(datasets[idx_dat])
                    dataset = pd.read_csv('../assets/' + datasets[idx_dat], sep=',')
                    if partition == 'uniform':
                        fuzzy_dataset = fuzzy_data.FuzzyDataUniformTriangular(datasets[idx_dat], dataset, n_fs[idx_fs])
                    elif partition == 'cmeans':
                        fuzzy_dataset = fuzzy_data.FuzzyDataCMeans(datasets[idx_dat], dataset, n_fs[idx_fs])
                    else:
                        fuzzy_dataset = fuzzy_data.FuzzyDataFCM(datasets[idx_dat], dataset, n_fs[idx_fs])
                    I = impl_operators[idx_op]
                    T = tnorms_operators[idx_op]
                    for idx_target in range(n_fs[idx_fs]):
                        print(idx_target)
                        sr1, m1 = SDFIOE.SDFIOE(dataset, fuzzy_dataset, idx_target, T, I, K=10, max_features=5,
                                                min_coverage=0.1)
                        results = results + [[datasets[idx_dat], partition, n_fs[idx_fs], idx_op, idx_target, m1.fwracc.mean()]]
    np.savetxt("ResultsUniform3.csv",
               results,
               delimiter=", ",  # Set the delimiter as a comma followed by a space
               fmt='% s')  # Set the format of the data as string