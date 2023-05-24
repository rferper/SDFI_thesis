import numpy as np
import pandas as pd
import copy
import progressbar
import SDFI.base.fuzzy_rule as fuzzy_rule
import SDFI.base.set_fuzzy_rules as set_fuzzy_rules

def N(x, w):
    try:
        return (1 - x) / (1 - w * x)
    except:
        return 1

def GSDFIW(dataset, fuzzy_dataset, ind_target_class, T, I, K=10, max_features=5, min_coverage=0.1, w=0.85):
    """
    :param dataset: Dataset with input data, the target at the end
    :param fuzzy_dataset: Dataframe with all the corresponding fuzzy linguistic variables, the target at the end
    :param ind_target_class: Index of the Target Class
    :param T: T-norm
    :param I: Fuzzy implication function
    :param K: Number of output rules
    :param max_features: Maximum number of features in the antecedent
    :param min_coverage: Minimum fuzzy coverage for the considered rules
    :param w: Parameter of the Sugeno negation
    :return: A set,list of the K fuzzy rules with the highest value of MWRAcc in each iteration among those satisfying the
    restrictions of max_features and min_coverage. The list is decreasingly ordered by FWRAcc and also provides the
    other measures
    """
    n_examples = len(dataset)
    n_features = len(fuzzy_dataset.fv_list) - 1
    max_features = min(n_features, max_features)
    nlabels_features = [0] * n_features
    for i in range(n_features):
        nlabels_features[i] = len(fuzzy_dataset.fv_list[i].get_labels)
    nlabels_target = len(fuzzy_dataset.fv_list[-1].get_labels)
    number_rules = np.prod(np.array(nlabels_features, dtype='int64') + 1) - 1

    def refine_rule(parent, m, explored_rules, rules_explored):
        if parent.get_num_features() >= max_features:
            rules_explored = rules_explored + np.prod(np.array(nlabels_features[m + 1:], dtype='int64') + 1) - 1
            bar.update(rules_explored)
            return explored_rules, rules_explored
        else:
            for k in range(m + 1, n_features):
                nLL = len(fuzzy_dataset.fv_list[k].get_labels)
                for j in range(nLL):
                    rules_explored = rules_explored + 1
                    children_ant = copy.deepcopy(parent.brule[:-1])
                    children_ant[k][j] = 1
                    children_con = [[0] * nlabels_target]
                    children_con[0][ind_target_class] = 1
                    children_brule = children_ant + children_con
                    children = fuzzy_rule.FuzzyRule(children_brule)
                    children = children.evaluate_rule_database(dataset, fuzzy_dataset, T, I)
                    measures = children.test_rule()
                    if float(measures.fcoverage) >= min_coverage:
                        children_df = pd.DataFrame([[children, float(measures.fwracc)]],
                                                   columns=['binary_rule', 'fwracc'])
                        explored_rules = pd.concat([explored_rules, children_df], ignore_index=True)
                        explored_rules, rules_explored = refine_rule(children, k, explored_rules, rules_explored)
                    else:
                        if k < n_features - 1:
                            rules_explored = rules_explored + np.prod(np.array(nlabels_features[k + 1:], dtype='int64') + 1) - 1
                            bar.update(rules_explored)
        return explored_rules, rules_explored

    root_ant = [[0] * nlabels_features[n] for n in range(n_features)]
    v = [0] * nlabels_target
    root = fuzzy_rule.FuzzyRule(root_ant + [v])
    explored_rules = pd.DataFrame(columns=['binary_rule', 'fwracc'])
    rules_explored = 0
    widgets = [
        progressbar.Percentage(), ' ',
        progressbar.Counter(format='(%(value)02d/%(max_value)d rules considered)'),
        progressbar.Bar(),
        progressbar.Timer(), '~',
        progressbar.ETA(),
    ]
    with progressbar.ProgressBar(max_value=number_rules, widgets=widgets) as bar:
        explored_rules, rules_explored = refine_rule(root, -1, explored_rules, rules_explored)
    explored_rules['fwracc'] = pd.to_numeric(explored_rules['fwracc'])
    explored_rules = explored_rules.sort_values(by=['fwracc'], ascending=False)
    selected_rules = []
    weights = np.array([1] * n_examples, dtype=np.float64)
    while len(selected_rules) < K:
        i_new = explored_rules['fwracc'].argmax()
        new_rule = explored_rules['binary_rule'].iloc[i_new]
        selected_rules.append(new_rule)
        # drop selected rule
        explored_rules=explored_rules.drop(index=explored_rules.iloc[i_new].name)
        # modify weights
        weights = T(weights, N(new_rule.antecedents, w))
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
