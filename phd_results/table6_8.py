import pandas as pd
import numpy as np
import random
import SDFI.base.operators.implications as implications
import SDFI.base.operators.tnorms as tnorms
import SDFI.base.fuzzy_data as fuzzy_data
import SDFI.base.set_fuzzy_rules as set_fuzzy_rules
import SDFI.methods.SDFIOE as SDFIOE
import SDFI.methods.WCSDFI as WCSDFI

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
    lambdas = [0.25, 0.5, 0.75, 0.9]

    partitions = ['uniform']
    I = implications.ImplicationsExamples.get_fuzzy_implication(implications.ImplicationsExamples.GODEL)
    T = tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.MINIMUM)

    results = [
        ['dataset', 'lambda', 'target', 'fwracc_sin', 'fwracc_con', 'overall_coverage_sin', 'overall_coverage_con']]
    for idx_dat in range(len(datasets)):
        random.seed(2021)
        dataset = pd.read_csv('../assets/' + datasets[idx_dat], sep=',')
        fuzzy_dataset = fuzzy_data.FuzzyDataFCM(datasets[idx_dat], dataset, 3)
        for idx_target in range(0, 3):
            print(idx_target)
            sr3, m3 = SDFIOE.SDFIOE(dataset, fuzzy_dataset, idx_target, T, I, K=1000, max_features=5, min_coverage=0.1)
            if len(m3) < 10:
                for lam in lambdas:
                    oc1 = sr3.overall_coverage(T, lam)
                    results = results + [
                        [datasets[idx_dat], 'null', idx_target, m3.fwracc.mean(), m3.fwracc.mean(), oc1, oc1]]
            else:
                sr1 = set_fuzzy_rules.SetFuzzyRules(sr3.rule_list[0:10])
                m1 = m3[0:10]
                for lam in lambdas:
                    sr2, m2 = WCSDFI.WCSDFI(dataset, fuzzy_dataset, sr3, T, I, K=10, w=lam)
                    oc1 = sr1.overall_coverage(T, lam)
                    oc2 = sr2.overall_coverage(T, lam)
                    results = results + [
                        [datasets[idx_dat], lam, idx_target, m1.fwracc.mean(), m2.fwracc.mean(), oc1, oc2]]
                    print(results)
    np.savetxt("ResultsLambdasCase1FCM.csv",
               results,
               delimiter=", ",  # Set the delimiter as a comma followed by a space
               fmt='% s')  # Set the format of the data as string