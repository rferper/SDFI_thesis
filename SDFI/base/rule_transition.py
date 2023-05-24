import pandas as pd
import numpy as np


class RuleTransition(object):
    def __init__(self, origin, destination, jump_order, sharpness):
        self.origin = origin
        self.destination = destination
        self.jump_order = jump_order
        self.sharpness = sharpness

    def sentence_condition_added(self, fuzzy_data):
        new_feature = list(self.destination.get_used_features() - self.origin.get_used_features())[0]
        feature = fuzzy_data.fv_list[new_feature]
        labels = feature.get_labels
        out = []
        r = int(np.nonzero(self.destination.bant[new_feature])[0])
        out.append(feature.name)
        out.append('IS')
        out.append(labels[r])
        return ' '.join(out)

    def sentence_new_consequent(self, fuzzy_data):
        target = fuzzy_data.fv_list[-1]
        labels = target.get_labels
        nlabels_target = len(labels)
        v = [0] * nlabels_target
        new_ind_target = self.destination.best_ind_target
        v[new_ind_target] = 1
        v = np.array(v)
        out = ['THEN', target.name, 'IS', labels[int(np.nonzero(v)[0])]]
        return ' '.join(out)

    def description(self, fuzzy_data):
        out = pd.DataFrame([[self.origin.to_sentence_rule(fuzzy_data),
                             self.sentence_condition_added(fuzzy_data),
                             self.sentence_new_consequent(fuzzy_data),
                             self.destination.fcoverage(),
                             self.sharpness]],
                           columns=['origin_rule',
                                    'condition_added',
                                    'new_consequent',
                                    'destination_fcoverage',
                                    'sharpness'])
        return out