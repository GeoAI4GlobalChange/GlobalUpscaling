import os
from tigramite import data_processing as pp
import numpy as np
import time
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import pandas as pd

def Build_input_data(input_data_dir,var_names,len_limit=10,window_size=2):
    files = os.listdir(input_data_dir)
    for file_idx in range(len(files)):
        file = files[file_idx]
        data_path = input_data_dir + file
        df = pd.read_csv(data_path)
        data = df[var_names].values
        not_nan_len = len(data) - np.sum(np.any(np.isnan(data), axis=1))
        if not_nan_len > len_limit:
            trend = np.full([data.shape[0], data.shape[1]], np.nan)
            for j in range(data.shape[1]):
                for i in range(data.shape[0]):
                    trend[i, j] = np.nanmean(
                        data[np.max([i - window_size, 0]):np.min([i + window_size + 1, data.shape[0]]), j])  #
                    data[i, j] = data[i, j] - trend[i, j]
                data[:, j] = (data[:, j] - np.nanmin(data[:, j])) / (np.nanmax(data[:, j]) - np.nanmin(data[:, j]))

            flag_value = -9999
            data[np.isnan(data)] = flag_value
            dataframe = pp.DataFrame(data, var_names=var_names, missing_flag=flag_value)
        else:
            print('time series is too short for casuality inference')
            dataframe=None
    return dataframe
def Build_initial_links(tau_min=0,tau_max=1,var_names=['FCH4','GPP']):
    ###########################################
    ##This function provides the initial knowledge that governs wetland CH4 dynamics
    #References:
    # Ruddell, B. L., & Kumar, P. (2009). Ecohydrologic process networks: 1. Identification. Water Resources Research, 45(3).
    # Yuan et al. (2022). Causality guided machine learning model on wetland CH4 emissions across global wetlands. Agricultural and Forest Meteorology, 324, 109115.
    ##########################################
    _vars = list(range(len(var_names)))
    _int_link_assumptions = {}
    for j in _vars[:]:
        _int_link_assumptions[j] = {}
        if j == 0:  ##The target variable (i.e., FCH4) is the first variable
            for i in _vars:
                if i == j:
                    for lag in range(1, 2):##consider FCH4(t-1)->FCH4(t): Ruddell and Kumar (2009);
                        _int_link_assumptions[j][(i, -lag)] = '-?>'
                else:
                    if var_names[i] in ['GPP', 'TS', 'TA', 'P', 'SWC']:##variabels with lagged controls: Yuan et al. (2022)
                        for lag in range(tau_min, tau_max + 1):
                            if not (i == j and lag == 0):
                                if lag == 0:
                                    _int_link_assumptions[j][(i, 0)] = 'o?o'
                                else:
                                    _int_link_assumptions[j][(i, -lag)] = '-?>'
                    else:##variabels without lagged controls e.g., windspeed
                        _int_link_assumptions[j][(i, 0)] = 'o?o'
        else:
            for i in _vars:
                for lag in range(tau_min, tau_max + 1):
                    if not (i == j and lag == 0):
                        if lag == 0:
                            _int_link_assumptions[j][(i, 0)] = 'o?o'
                        else:
                            _int_link_assumptions[j][(i, -lag)] = '-?>'
    return _int_link_assumptions
def Causality_detection_PCMCI(max_lag=1,pc_alpha=[0.05],var_names=['y','x1','x2'],input_data_dir='./',output_dir='./'):
    dataframe=Build_input_data(input_data_dir, var_names)
    if dataframe is not None:
        _int_link_assumptions = Build_initial_links(tau_min, tau_max=max_lag, var_names=var_names)
        np.random.seed(42)  # Ensures reproducibility
        start_time = time.time()  # Start timing
        parcorr = ParCorr(significance='analytic')
        pcmci_parcorr = PCMCI(
            dataframe=dataframe,
            cond_ind_test=parcorr,
            verbosity=0)
        results = pcmci_parcorr.run_pcmciplus(link_assumptions=_int_link_assumptions,max_conds_dim=3,tau_min=tau_min, tau_max=tau_max,max_combinations=len(_int_link_assumptions[0]), max_conds_px=1,max_conds_py=3,pc_alpha=pc_alpha)
        end_time = time.time()  # End timing
        print(end_time - start_time)   # Compute elapsed time

        val_matrix = results['val_matrix'][:,0]
        p_matrix = results['p_matrix'][:,0]
        indexes = []
        for lag in range(0, max_lag + 1):
            indexes.append(f't_{lag}')
        df_result=pd.DataFrame(val_matrix,columns=indexes,index=var_names)
        df_result_p = pd.DataFrame(p_matrix, columns=indexes, index=var_names)
        save_dir=output_dir
        save_path=save_dir+f'PCMCI_{parcorr_method}.csv'
        save_path_p = save_dir + f'PCMCI_{parcorr_method}_p.csv'
        df_result.to_csv(save_path)
        df_result_p.to_csv(save_path_p)
        print('Causality inference completed,results saved to:',save_dir)
    else:
        print('Causality inference not completed because of limited length of data available')
if __name__=='__main__':
    max_lag=12# 12 weeks (~3 months for intra-seasonal controls)
    pc_alpha = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    input_data_dir=r'./PCMCI_input_data/'
    output_dir=r'./PCMCI_results/'
    var_names = ['FCH4_weekly', 'GPP', 'PA', 'TS', 'TA', 'P', 'WS', 'SWC'] # variables used to infer causal relationships
    Causality_detection_PCMCI(max_lag,pc_alpha,var_names,input_data_dir,output_dir)
