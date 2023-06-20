import pandas as pd
from pathlib import Path
import random
import SDFI.base.fuzzy_data as fuzzy_data
import SDFI.base.fuzzy_set as fuzzy_set
import SDFI.base.membership_function as membership_function
import SDFI.base.fuzzy_linguistic_variable as fuzzy_linguistic_variable


if __name__ == '__main__':
    # Read daset
    name_dataset = 'sd_usa2012.csv'
    dataset = pd.read_csv('../assets/' + name_dataset, sep=',')
    dataset_features = dataset.drop(['Obama'], axis=1)
    dataset_target = dataset['Obama']
    random.seed(2021)
    # automatic fuzzy partition of features
    fuzzy_dataset_features = fuzzy_data.FuzzyDataFCM('usa_features', dataset_features, 3, ['L', 'M', 'H'])
    # custom fuzzy partition of target
    fv_list = []
    TFS1 = fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=0, b=0, c=45, d=50), 'Republican')
    TFS2 = fuzzy_set.FuzzySet(membership_function.TriangularMF(a=45, b=50, c=55), 'Competitive')
    TFS3 = fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=50, b=55, c=100, d=100), 'Democratic')

    fuzzy_target = fuzzy_linguistic_variable.LinguisticVariable(name="State",
                                                                fs_list=[TFS1, TFS2, TFS3],
                                                                universe_of_discourse=[0, 100])
    fuzzy_dataset = fuzzy_data.FuzzyData(name="FuzzyUsa", fv_list=fuzzy_dataset_features.fv_list + [fuzzy_target])
    data = dataset.copy()
    for i in range(len(fuzzy_dataset.fv_list)):
        data[dataset.columns[i]] = dataset[dataset.columns[i]].map(
            lambda x: fuzzy_dataset.fv_list[i].eval_max_fuzzy_set(x))
    filepath = Path('../assets/usa2012_discretized.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath, index=False)
