import pandas as pd
import numpy as np
import SDFI.base.membership_function as membership_function
import SDFI.base.fuzzy_set as fuzzy_set
import SDFI.base.fuzzy_linguistic_variable as fuzzy_linguistic_variable
from pyFTS.partitioners import CMeans, FCM, Entropy, Huarng
from pyFTS.common import Membership as mf


class FuzzyData(object):
    """
        Creates a new fuzzy data.
        Args:
            fv_list: a list of LinguisticVariables instances.
            name: a name of for the fuzzy data
    """

    def __init__(self, name='', fv_list=[]):

        if not fv_list:
            raise Exception("ERROR: please specify at least one fuzzy variable")
        self.fv_list = fv_list
        self.name = name

    @property
    def get_names(self):
        labels = []
        for v in self.fv_list:
            labels.append(v.name)
        return labels

    # Plot

    def __repr__(self):
        if self.name is None:
            text = "N/A"
        else:
            text = self.name
        return "<Fuzzy Data'" + text + "', contains linguistic variables %s" % (str(self.fv_list))


class FuzzyDataUniformTriangular(FuzzyData):
    def __init__(self, name: str, dataset: pd.core.frame.DataFrame, n_labels, labels=None):
        if labels is None:
            labels = ['Label ' + str(i) for i in list(range(0, n_labels))]
        names_variables = dataset.columns.values
        fv_list = []
        for name_v in names_variables:
            variable = dataset[name_v]
            if variable.dtype == float:
                fv_list.append(fuzzy_linguistic_variable.UniformTriangle(name=name_v, n_sets=n_labels,
                                                                         labels=labels,
                                                                         universe_of_discourse=[
                                                                             np.min(variable),
                                                                             np.max(variable)]))
            if variable.dtype == object:
                fv_list.append(fuzzy_linguistic_variable.CrispLinguisticVariable(name_v, variable.unique()))
        super().__init__(name, fv_list)


class FuzzyDataCMeans(FuzzyData):
    def __init__(self, name: str, dataset: pd.core.frame.DataFrame, n_labels=3, labels=None):

        names_variables = dataset.columns.values
        fv_list = []
        for name_v in names_variables:
            variable = dataset[name_v]
            if variable.dtype == float:
                part = CMeans.CMeansPartitioner(data=variable, npart=n_labels, func=mf.trimf)
                keys = list(part.sets.keys())
                fs_list = []
                if labels is None:
                    labels = keys
                for i in range(len(keys)):
                    parameters = part.sets[keys[i]].parameters
                    fs_list.append(fuzzy_set.FuzzySet(membership_function.TriangularMF(a=parameters[0],
                                                                                       b=parameters[1],
                                                                                       c=parameters[2]), labels[i]))

                fv_list.append(fuzzy_linguistic_variable.LinguisticVariable(name=name_v,
                                                                            fs_list=fs_list,
                                                                            universe_of_discourse=[
                                                                                np.min(variable),
                                                                                np.max(variable)]))
            if variable.dtype == object:
                fv_list.append(fuzzy_linguistic_variable.CrispLinguisticVariable(name_v, variable.unique()))
        super().__init__(name, fv_list)


class FuzzyDataFCM(FuzzyData):
    def __init__(self, name: str, dataset: pd.core.frame.DataFrame, n_labels=3, labels=None):

        names_variables = dataset.columns.values
        fv_list = []
        for name_v in names_variables:
            variable = dataset[name_v]
            if variable.dtype == float:
                part = FCM.FCMPartitioner(data=variable, npart=n_labels, func=mf.trimf)
                keys = list(part.sets.keys())
                fs_list = []
                if labels is None:
                    labels = keys
                for i in range(len(keys)):
                    parameters = part.sets[keys[i]].parameters
                    fs_list.append(fuzzy_set.FuzzySet(membership_function.TriangularMF(a=parameters[0],
                                                                                       b=parameters[1],
                                                                                       c=parameters[2]), labels[i]))

                fv_list.append(fuzzy_linguistic_variable.LinguisticVariable(name=name_v,
                                                                            fs_list=fs_list,
                                                                            universe_of_discourse=[
                                                                                np.min(variable),
                                                                                np.max(variable)]))
            if variable.dtype == object:
                fv_list.append(fuzzy_linguistic_variable.CrispLinguisticVariable(name_v, variable.unique()))
        super().__init__(name, fv_list)


