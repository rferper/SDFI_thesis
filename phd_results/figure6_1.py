import pandas as pd
import time
import matplotlib.pyplot as plt
import SDFI.base.fuzzy_data as fuzzy_data
import SDFI.base.operators.implications as implications
import SDFI.base.operators.tnorms as tnorms
import SDFI.methods.SDFIOE as SDFIOE
import SDFI.methods.GSDFIW as GSDFIW

if __name__ == '__main__':
    dataset = pd.read_csv('../assets/sd_wpbc.csv', sep=',')
    Implication = implications.ImplicationsExamples.get_fuzzy_implication(implications.ImplicationsExamples.GODEL)
    Tnorm = tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.MINIMUM)
    fuzzy_dataset = fuzzy_data.FuzzyDataUniformTriangular('wpbc', dataset, 3, ['L', 'M', 'H'])
    coverages = [x / 100 for x in range(0, 51, 5)]
    oe_fwracc = [[0] * len(coverages) for i in range(3)]
    oe_time = [[0] * len(coverages) for i in range(3)]
    w_fwracc = [[0] * len(coverages) for i in range(3)]
    w_time = [[0] * len(coverages) for i in range(3)]
    for i in range(len(coverages)):
        cov = coverages[i]
        print('Min Coverage: {}'.format(cov))
        for j in range(3):
            print('Target: {}'.format(j))
            st = time.time()
            sr1, m1 = SDFIOE.SDFIOE(dataset, fuzzy_dataset, j, Tnorm, Implication, K=10, max_features=5,
                                    min_coverage=cov)
            et = time.time()
            elapsed_time = et - st
            oe_fwracc[j][i] = m1.fwracc.mean()
            oe_time[j][i] = elapsed_time

    # plots

    # SDFIOE
    fig1, ax1 = plt.subplots()
    ax1.plot(coverages, oe_fwracc[0], color='red', label='Target=L')
    ax1.plot(coverages, oe_fwracc[1], color='blue', label='Target=M')
    ax1.plot(coverages, oe_fwracc[2], color='green', label='Target=H')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('AvFWRAcc')
    fig1.legend()
    fig1.savefig('mincov_oe_fwracc.png')

    fig2, ax2 = plt.subplots()
    ax2.plot(coverages, oe_time[0], color='red', label='Target=L')
    ax2.plot(coverages, oe_time[1], color='blue', label='Target=M')
    ax2.plot(coverages, oe_time[2], color='green', label='Target=H')
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('ctime')
    fig2.legend()
    fig2.savefig('mincov_oe_ctime.png')


    # GSDFIW
    w_fwracc = [[0] * len(coverages) for i in range(3)]
    w_time = [[0] * len(coverages) for i in range(3)]
    for i in range(len(coverages)):
        cov = coverages[i]
        print('Min Coverage: {}'.format(cov))
        for j in range(3):
            print('Target: {}'.format(j))
            st = time.time()
            sr2, m2 = GSDFIW.GSDFIW(dataset, fuzzy_dataset, j, Tnorm, Implication, K=10, max_features=5,
                                   min_coverage=cov,w=0.7)
            et = time.time()
            elapsed_time = et - st
            w_fwracc[j][i] = m2.fwracc.mean()
            w_time[j][i] = elapsed_time

    fig3, ax3 = plt.subplots()
    ax3.plot(coverages, w_fwracc[0], color='red', label='Target=L')
    ax3.plot(coverages, w_fwracc[1], color='blue', label='Target=M')
    ax3.plot(coverages, w_fwracc[2], color='green', label='Target=H')
    ax3.set_xlabel('alpha')
    ax3.set_ylabel('fwrac')
    fig3.legend()
    fig3.savefig('mincov_w_fwracc.png')

    fig4, ax4 = plt.subplots()
    ax4.plot(coverages, w_time[0], color='red', label='Target=L')
    ax4.plot(coverages, w_time[1], color='blue', label='Target=M')
    ax4.plot(coverages, w_time[2], color='green', label='Target=H')
    ax4.set_xlabel('alpha')
    ax4.set_ylabel('ctime')
    fig4.legend()
    fig4.savefig('mincov_w_ctime.png')