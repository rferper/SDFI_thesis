from enum import Enum
import numpy as np

class ImplicationsExamples(Enum):
    GODEL = "godel_implication"
    GOGUEN = "goguen_implication"
    FGM = "FGM"
    KSS = "KSS"
    KH = "KH"
    KF = "KF"

    @staticmethod
    def get_fuzzy_implication(implication):
        if implication == ImplicationsExamples.GODEL:
            return ImplicationsExamples.godel_implication
        if implication == ImplicationsExamples.GOGUEN:
            return ImplicationsExamples.goguen_implication
        if implication == ImplicationsExamples.FGM:
            return ImplicationsExamples.fgm_implication
        if implication == ImplicationsExamples.KSS:
            return ImplicationsExamples.kss_implication
        if implication == ImplicationsExamples.KH:
            return ImplicationsExamples.kh_implication
        if implication == ImplicationsExamples.KF:
            return ImplicationsExamples.kf_implication

    @staticmethod
    def godel_implication(x: float, y: float) -> float:
        if x <= y:
            return 1
        else:
            return y

    @staticmethod
    def goguen_implication(x: float, y: float) -> float:
        if x <= y:
            return 1
        else:
            return y / x

    @staticmethod
    def fgm_implication(x: float, y: float, theta: float) -> float:
        if x == 0:
            return 1
        else:
            return y + theta * y * (1 - x) * (1 - y)

    @staticmethod
    def kss_implication(x: float, y: float, theta: float) -> float:
        if x == 0:
            return 1
        else:
            if y == 0:
                return 0
            else:
                if x <= np.exp((y ** theta - 1) / theta):
                    return 1
                else:
                    return (y ** theta - theta * np.log(x)) ** (1 / theta)

    @staticmethod
    def kh_implication(x: float, y: float, theta: float) -> float:
        if x <= y / (theta + (1 - theta) * y):
            return 1
        else:
            return theta * y / (theta * x - (1 - theta) * (1 - x) * y)

    @staticmethod
    def kf_implication(x: float, y: float, theta: float) -> float:
        if x == 0:
            return 1
        else:
            if y == 0:
                return 0
            else:
                if x <= (theta ** y - 1) / (theta - 1):
                    return 1
                else:
                    return np.log((x + theta ** y - 1) / x) / np.log(theta)