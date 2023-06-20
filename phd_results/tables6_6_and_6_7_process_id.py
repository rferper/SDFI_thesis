import pandas as pd
import numpy as np

if __name__ == '__main__':

    datasets = [
        'sd_wpbc.csv',
        'sd_usa2012.csv',
        'sd_startups.csv',
        'sd_electricity.csv',
        'sd_stockprices.csv',
        'sd_treasury.csv',
        'sd_calhousing.csv',
    ]

    results = [['case 1', 'case 2', 'case 3', 'case 4', 'case 5', 'case 6']]
    # We have to generate the csv with the file tables6_6_and_6_7_generate_id
    rules = pd.read_csv('../assets/' + 'ResultsIDRulesUniformvsFCM.csv', sep=',')
    M = np.zeros((12, 12))
    for i in range(0, 12):
        for j in range(i + 1, 12):
            diss_perc = []
            for d in range(0, 7):
                for idx in range(0, 3):
                    rules_dataset_i = rules[(rules['dataset'] == datasets[d]) & (rules[' operator'] == i) & (rules[' target'] == idx)][' rule_id']
                    rules_dataset_j = rules[(rules['dataset'] == datasets[d]) & (rules[' operator'] == j) & (rules[' target'] == idx)][' rule_id']
                    diss_perc = diss_perc + [len(set(rules_dataset_i) & set(rules_dataset_j)) / float(len(set(rules_dataset_i) | set(rules_dataset_j))) * 100]
            M[i][j] = round(np.mean(diss_perc),3)
