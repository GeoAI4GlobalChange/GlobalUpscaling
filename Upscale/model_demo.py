import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from methane_module import Causal_ML,ML_train,ML_validate,ML_test
from methane_data import fluxnetData4ML,Data4ML_tensor,get_causal_strength,chamberData4ML
import os
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

def model_training(depths, batch_size, prediction_horizon, test_precent, seed_num, data_source, site_types, target,
                   target_cols, dir, path_site_info, dir_causal_stren, chamber_dir, para_dir, results_dir):
    for depth in depths:
        # Initialize dataframe 'df_result' to save the results, including MAE, Pearson R, R^2,and feature importance of 'GPP','PA','TS','TA','P','WS','SC','SWC'.
        df_result = pd.DataFrame(
            columns=['wetlandtype', 'experiment_id', 'mae', 'r', 'r2'])
        df_idx=0
        for seed in range(seed_num):
            random.seed(seed)
            dir_site_data = dir
            input_cols=target_cols
            target_variable=target
            #obtain the training, validation and test datasets
            train_datasets, val_datasets, test_datasets=fluxnetData4ML(dir_site_data, target_variable, input_cols, path_site_info, seed,site_types,depth,prediction_horizon,test_precent)
            print(train_datasets.keys())
            causal_punish_para = 1 * pow(10, -2)
            #obtain causal strength
            causal_dict_1=get_causal_strength(data_source,depth,dir_causal_stren,site_types)
            #load chamber datasets
            chamber_input,chamber_output,chamber_ID=chamberData4ML(chamber_dir,data_source,depth)

            for type in site_types:
                causal_stren = causal_dict_1[type]
                causal_stren_raw = causal_stren[np.newaxis, :]
                train_loader, val_loader, test_loader,chamber_train_idxs, chamber_validate_idxs, chamber_test_idxs,x_means, x_std, y_means, y_std=\
                    Data4ML_tensor(type, train_datasets, val_datasets, test_datasets, chamber_input, chamber_output,
                               batch_size, seed,test_precent)
                device='cpu'
                # Model building
                him_dim=4#hiden state vector dimention 4
                model = Causal_ML(train_datasets[type]['X'].shape[2], 1, him_dim,device=device,dropout=0.1).to(device=device)
                opt = torch.optim.Adam(model.parameters(), lr=0.01,)
                epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
                opt_chamber = torch.optim.Adam(model.parameters(), lr=0.01, )
                epoch_scheduler_chamber = torch.optim.lr_scheduler.StepLR(opt_chamber, 1, gamma=0.9)
                epochs = 80
                loss = nn.MSELoss()
                patience = 50
                min_val_loss = 9999
                #Initialize or load model parameters
                para_path = f"{para_dir}{data_source}_CausalConstrained_ML_{type}_{him_dim}_seed{seed}_lag{depth}.pt"
                if not os.path.exists(directory_path):
                    os.makedirs(para_dir)
                if os.path.exists(para_path):
                    model.load_state_dict(torch.load(para_path))

                for i in range(epochs):
                    mse_train = 0
                    #####################################
                    #train the model
                    chamber_tune_epochs = 3
                    x_chamber = chamber_input[type]
                    y_chamber = chamber_output[type]
                    ML_train(model, opt, opt_chamber, loss, causal_stren_raw, train_loader,
                             x_chamber, y_chamber, chamber_train_idxs,
                             x_means, x_std, y_means, y_std, chamber_tune_epochs,causal_punish_para,mse_train, device)
                    epoch_scheduler.step()
                    epoch_scheduler_chamber.step()
                    #######################################
                    # validate the model
                    counter=ML_validate(model, loss, val_loader, min_val_loss, patience, para_path,
                                x_chamber, y_chamber, chamber_validate_idxs,
                                x_means, x_std, y_means, y_std, device)
                    if counter == patience:
                        break
                #############################################
                # test the model
                mse_val = 0
                preds = []
                true = []
                alphas = []
                betas = []
                preds, true, alphas, betas=ML_test(model, loss, test_loader, preds, true, alphas, betas, mse_val,
                        chamber_input, chamber_output, chamber_ID, chamber_test_idxs,
                        x_means, x_std, y_means, y_std,type, device)
                preds = np.concatenate(preds)
                true = np.concatenate(true)
                preds = preds*(y_std+pow(10,-6)) + y_means
                true = true*(y_std+pow(10,-6)) + y_means
                mse = mean_squared_error(true, preds)
                mae = mean_absolute_error(true, preds)
                r=stats.pearsonr(true, preds)[0]
                r2=r2_score(true, preds)
                temp_result = [type, seed, mae, r,r2]
                df_result.loc[df_idx] = temp_result
                df_idx += 1
                print(type,mse, mae,stats.pearsonr(true, preds)[0])
        ###############################################################
        df_final=pd.DataFrame(columns=['wetlandtype','mae_mean','r_mean','r2_mean','mae_std','r_std','r2_std'])
        df_idx_stats=0
        for type in site_types:
            mae=df_result[(df_result['wetlandtype']==type)]['mae'].values
            r=df_result[(df_result['wetlandtype']==type)]['r'].values
            r2=df_result[(df_result['wetlandtype']==type)]['r2'].values
            df_final.loc[df_idx_stats] = [type, np.mean(mae), np.mean(r),np.mean(r2), np.std(mae), np.std(r), np.std(r2)]
            df_idx_stats += 1
        # obtain and save the results
        df_final.to_csv(f'{results_dir}{data_source}_CausalConstrained_upscale_ml_mean_std_lag{depth}_20_test.csv')
        df_result.to_csv(f'{results_dir}{data_source}_CausalConstrained_upscale_ml_performance_lag{depth}_20_test.csv')

if __name__=='__main__':
    depths = [12]  # maximum time lags
    batch_size = 32  # number of sub-samples that used to calculate the loss
    prediction_horizon = 0  # leading time
    test_precent = 0.1  # percentage of test dataset selection
    seed_num = 20  # number of seeds. Here we repeat the experiment for 20 times to estimate model parameter uncertainty due to random data selection.
    data_source = 'era5'  # name of the input forcing data source
    site_types = ['All']  # all wetland types
    target = 'FCH4_weekly'  # name of methane emission variable in the observation datasets
    target_cols = ['GPP','PA','TS','TA','P','WS','SC','SWC','T_CLAY', 'T_GRAVEL',  'T_SAND', 'T_SILT',  'T_PH_H2O']  # name of the drivers
    dir = r'./data/'  # path of observation datasets (including FCH4 and its drivers)
    path_site_info = r'./site_location.csv'  # path of site informations (including site name and the corresponding wetland type)
    dir_causal_stren = f'./PCMCI_causality/'  # path of causality strength between FCH4 and its drivers
    chamber_dir = './chamber_data/'  # path of chamber datasets (including FCH4 and its drivers)
    para_dir = './para/'  # path to save the model parameters
    results_dir = './output/'  # path to save the results
    #train the causal ML model and save the results
    model_training(depths, batch_size, prediction_horizon, test_precent, seed_num, data_source, site_types, target,
                   target_cols, dir, path_site_info, dir_causal_stren, chamber_dir, para_dir, results_dir)

