import numpy as np
import pandas as pd
import SDFI.base.set_fuzzy_rules as set_fuzzy_rules
def N(x, w):
    try:
        return (1 - x) / (1 - w * x)
    except:
        return 1


def WCSDFI(dataset, fuzzy_dataset, set_rules, T, I, K=10, w=0.85):
    """
    :param dataset: Dataset with input data, the target at the end
    :param fuzzy_dataset: Dataframe with all the corresponding fuzzy linguistic variables, the target at the end
    :param set_rules: Set of rules to post-process
    :param T: T-norm
    :param I: Fuzzy implication function
    :param K: Number of output rules
    :param w: Parameter of the Sugeno negation
    :return: A subset of set_rules of size K with the highest value of MFWRAcc in each iteration
    """
    n_examples = len(dataset)
    n_features = len(fuzzy_dataset.fv_list) - 1
    nlabels_features = [0] * n_features
    for i in range(n_features):
        nlabels_features[i] = len(fuzzy_dataset.fv_list[i].get_labels)
    weights = np.array([1] * n_examples, dtype=np.float64)
    explored_rules = pd.DataFrame(columns=['binary_rule', 'fwracc'])
    explored_rules['binary_rule'] = set_rules.rule_list
    explored_rules['fwracc'] = explored_rules['binary_rule'].apply(lambda x: x.fwracc_weights(T, weights))
    selected_rules = []
    while len(selected_rules) < K:
        i_new = explored_rules['fwracc'].argmax()
        new_rule = explored_rules['binary_rule'].iloc[i_new]
        selected_rules.append(new_rule)
        # drop selected rule
        explored_rules = explored_rules.drop(index=explored_rules.iloc[i_new].name)
        # modify weights
        weights = np.array(list(map(lambda x, y: T(x, N(y, w)), weights, new_rule.antecedents)))
        explored_rules['fwracc'] = explored_rules['binary_rule'].apply(lambda x: x.fwracc_weights(T, weights))

    selected_rules = set_fuzzy_rules.SetFuzzyRules(selected_rules)
    out = selected_rules.measures(I, fuzzy_dataset)
    out = out.sort_values(by=['fwracc'], ascending=False)

    print('| ------ SUMMARY ------ |')
    print('Mean Num_features: {}'.format(round(np.mean(out.num_features), 3)))
    print('Mean FCoverage: {}'.format(round(np.mean(out.fcoverage), 3)))
    print('Mean FSupport: {}'.format(round(np.mean(out.fsupport), 3)))
    print('Mean FConfidence: {}'.format(round(np.mean(out.fconfidence), 3)))
    print('Mean FWRAcc: {}'.format(round(np.mean(out.fwracc), 3)))

    return selected_rules, out