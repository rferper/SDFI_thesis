import pandas as pd
import copy
import SDFI.base.fuzzy_rule_antecedent as fuzzy_rule_antecedent
import SDFI.base.rule_transition as rule_transition

def STFI(dataset, fuzzy_dataset, T, I, max_features=5, min_coverage=0.1):
    """
    :param dataset: Dataset with input data, the target at the end
    :param fuzzy_dataset: Dataframe with all the corresponding fuzzy linguistic variables, the target at the end
    :param T: T-norm
    :param I: Fuzzy implication function
    :param max_features: Maximum number of features in the antecedent
    :param min_coverage: Minimum fuzzy coverage for the considered rules
    :return: A list of all the transitions, satisfying the restrictions of max_features and min_coverage, decreasingly
    ordered by Sharpness
    """
    n_features = len(fuzzy_dataset.fv_list) - 1
    max_features = min(n_features, max_features)
    nlabels_features = [0] * n_features
    for i in range(n_features):
        nlabels_features[i] = len(fuzzy_dataset.fv_list[i].get_labels)

    def recursive_new_rule(origin, sharp_transitions, evaluated):
        if len(origin.get_used_features()) >= max_features or origin.ID in evaluated:
            return sharp_transitions, evaluated
        unused_features = set(range(n_features)) - origin.get_used_features()
        for k in unused_features:
            nLL = len(fuzzy_dataset.fv_list[k].get_labels)
            for j in range(nLL):
                destination_ant = copy.deepcopy(origin.bant)
                destination_ant[k][j] = 1
                destination_ant = fuzzy_rule_antecedent.FuzzyRuleAntecedent(destination_ant)
                destination_ant = destination_ant.evaluate_antecedent_database(dataset, fuzzy_dataset, T, I)
                if destination_ant.fcoverage() >= min_coverage:
                    destination = destination_ant.best_consequent(dataset, fuzzy_dataset, T, I)
                    jump_order = abs(destination.best_ind_target - origin.best_ind_target)
                    if jump_order > 0 and len(origin.get_used_features()) > 0:
                        sharpness = jump_order * min(origin.fconfidence, destination.fconfidence)
                        new_transition = rule_transition.RuleTransition(origin, destination, jump_order, sharpness)
                        sharp_transitions = pd.concat([sharp_transitions, new_transition.description(fuzzy_dataset)], ignore_index=True)
                    sharp_transitions, evaluated = recursive_new_rule(destination, sharp_transitions, evaluated)
        evaluated.add(origin.ID)
        return sharp_transitions, evaluated

    root_ant = [[0] * nlabels_features[n] for n in range(n_features)]
    root = fuzzy_rule_antecedent.FuzzyRuleAntecedent(root_ant)
    sharp_transitions = pd.DataFrame(columns=['origin_rule',
                                'condition_added',
                                'new_consequent',
                                'destination_fcoverage',
                                'sharpness'])
    evaluated = set()
    sharp_transitions, evaluated = recursive_new_rule(root, sharp_transitions, evaluated)

    sharp_transitions = sharp_transitions.sort_values(by=['sharpness'], ascending=False, ignore_index=True)
    return sharp_transitions