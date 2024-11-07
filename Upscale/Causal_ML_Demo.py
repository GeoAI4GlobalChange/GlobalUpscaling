import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from networks_cpu import IMVFullLSTM
import os
import copy
import random
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import time
import pickle
from torch.autograd import Variable
import torch.nn.functional as F


depths=[12]#time lag (weekly),
data_source='era5'
for depth in depths:# depths=sequence length or maximum time lag
    batch_size = 32 # number of sub-samples that used to calculate the loss
    prediction_horizon = 0 # leading time
    test_precent=0.1# datasets include train, validate, and test
    site_types=['All']
    df_result=pd.DataFrame(columns=['model','wetlandtype','experiment_id','mae','r','r2','GPP','PA','TS','TA','P','WS','SC','SWC','T_CLAY', 'T_GRAVEL',  'T_SAND', 'T_SILT',  'T_PH_H2O'])
    df_idx=0
    for seed in range(20):
        random.seed(seed)
        dir='./data/global_all_driver_gosif_era5_allsites_soil/' #path of EC data

        files=os.listdir(dir)
        files=files[1:]
        target = 'FCH4_weekly'
        target_cols=[
            'GPP','PA','TS','TA','P','WS','SC','SWC','T_CLAY', 'T_GRAVEL',  'T_SAND', 'T_SILT',  'T_PH_H2O']
        train_datasets={} #train dataset, wetland type: ndarray
        val_datasets={} #validate dataset
        test_datasets={} #test dataset
        site_type_dict={} # save site name: wetland type
        df=pd.read_csv(r'./data/site_global/site_location_global.csv') # path of site information
        df = df[((df['LAT'] >= 23)|(df['LAT'] <=- 23))]
        df=df[['ID','Type_all']].values

        for idx in range(df.shape[0]):
            site_type_dict[df[idx,0]]=df[idx,1]
        site_start={}
        for item in site_types:
            site_start[item]=True
        for file_idx in range(len(df)):
            site_name=df[file_idx,0]
            print(site_name)
            temp_type=site_type_dict[site_name] # get wetland type for each site according to its name
            if temp_type in site_types:
                file_path=dir+site_name+'_gosif_era5_soil.csv'
                data = pd.read_csv(file_path)
                site_final_cols=target_cols

                length=data.shape[0]
                data1=data
                X_train1 = np.zeros((len(data1), depth, len(site_final_cols)))
                for i, name in enumerate(site_final_cols):
                    for j in range(depth):
                        X_train1[:, j, i] = data1[name].shift(depth - j - 1)
                if prediction_horizon>0:
                    y_train1 = np.array(data1[target].shift(-prediction_horizon))
                    X_train1 = X_train1[(depth-1):-prediction_horizon]
                    y_train1 = y_train1[(depth-1):-prediction_horizon]
                else:
                    y_train1 = np.array(data1[target])
                    X_train1 = X_train1[(depth-1):]
                    y_train1 = y_train1[(depth-1):]

                #remove samples with nan or extreme values
                y_nan_mask=np.isnan(y_train1) | np.isinf(y_train1)
                y_extreme_mask=np.abs(y_train1)>pow(10,10)
                y_nan_mask=(y_nan_mask | y_extreme_mask)
                x_temp=X_train1.reshape((X_train1.shape[0],-1)).copy()
                x_nan_mask=np.any(np.isnan(x_temp),axis=1) | np.any(np.isinf(x_temp),axis=1)
                nan_mask=(y_nan_mask | x_nan_mask)

                if np.sum(nan_mask==False)>0:
                    X_train1=X_train1[nan_mask==False]
                    y_train1 = y_train1[nan_mask == False]
                    all_idxs = [item_idx for item_idx in range(y_train1.shape[0])]
                    random.seed(seed)
                    test_idxs = random.sample(all_idxs, int(test_precent * len(all_idxs)))
                    train_validate_idxs = list(set(all_idxs).difference(set(test_idxs)))
                    random.seed(seed)
                    validate_idxs = random.sample(train_validate_idxs, int(1 / 9 * len(train_validate_idxs)))
                    train_idxs = np.array(list(set(train_validate_idxs).difference(set(validate_idxs))))
                    test_idxs = np.array(test_idxs)
                    validate_idxs = np.array(validate_idxs)
                    if site_start[temp_type]:
                        train_datasets[temp_type]={}
                        train_datasets[temp_type]['X']=X_train1[train_idxs]
                        train_datasets[temp_type]['Y'] = y_train1[train_idxs]
                        test_datasets[temp_type] = {}
                        test_datasets[temp_type]['X'] = X_train1[test_idxs]
                        test_datasets[temp_type]['Y'] = y_train1[test_idxs]
                        val_datasets[temp_type] = {}
                        val_datasets[temp_type]['X'] = X_train1[validate_idxs]
                        val_datasets[temp_type]['Y'] = y_train1[validate_idxs]
                        site_start[temp_type]=False
                    else:
                        train_datasets[temp_type]['X'] = np.concatenate([train_datasets[temp_type]['X'],X_train1[train_idxs]], axis=0)
                        train_datasets[temp_type]['Y'] = np.concatenate([train_datasets[temp_type]['Y'],y_train1[train_idxs]], axis=0)
                        test_datasets[temp_type]['X'] = np.concatenate([test_datasets[temp_type]['X'], X_train1[test_idxs]], axis=0)
                        test_datasets[temp_type]['Y'] = np.concatenate([test_datasets[temp_type]['Y'], y_train1[test_idxs]],axis=0)
                        val_datasets[temp_type]['X'] = np.concatenate([val_datasets[temp_type]['X'], X_train1[validate_idxs]], axis=0)
                        val_datasets[temp_type]['Y'] = np.concatenate( [val_datasets[temp_type]['Y'], y_train1[validate_idxs]], axis=0)
        print(train_datasets.keys())

        causal_punish_para = 1 * pow(10, -2)
        device = 'cpu'
        causal_dict_1 = {}
        dir_causal_stren = f'./data/site_global/data/pcmci_results/'# load causality
        for temp_type in site_types:
            causal_dict_1[temp_type]=np.load(dir_causal_stren + f'{data_source}_{temp_type}_{depth}.npy',allow_pickle=True).astype(np.float32)


        with open(f"./data/chamber_model_input/{data_source}_input_chamber_{depth}.pkl", "rb") as tf:
            chamber_input=pickle.load(tf)
        with open(f"./data/site_global/data/chamber_model_input/{data_source}_output_chamber_{depth}.pkl", "rb") as tf:
            chamber_output=pickle.load(tf)
        with open(f"./data/chamber_model_input/{data_source}_ID_cleanchamber_{depth}.pkl", "rb") as tf:
            chamber_ID=pickle.load(tf)
        df_chamber=pd.read_csv(f'./data/site_global/All_chamber.csv')

        for type in site_types:
            causal_stren = causal_dict_1[type]
            causal_stren_raw = causal_stren[np.newaxis, :]
            causal_stren = np.repeat(causal_stren_raw, batch_size, axis=0)
            causal_stren = causal_stren / np.sum(causal_stren, axis=1, keepdims=True)
            causal_stren = torch.Tensor(causal_stren)
            weights = Variable(causal_stren, requires_grad=False).to(device=device)


            x_chamber=chamber_input[type]
            y_chamber=chamber_output[type]
            x_chamber=np.vstack(x_chamber)
            y_chamber=np.vstack(y_chamber)[:,0]
            x_train=train_datasets[type]['X']
            y_train=train_datasets[type]['Y']
            print('number of trained samples:',y_train.shape[0])
            x_test = test_datasets[type]['X']
            y_test = test_datasets[type]['Y']
            x_validate = val_datasets[type]['X']
            y_validate = val_datasets[type]['Y']
            x_all = np.concatenate((x_train, x_validate, x_test), axis=0)
            y_all = np.concatenate((y_train, y_validate, y_test), axis=0)

            ###########combine chamber
            x_all = np.concatenate((x_all, x_chamber), axis=0)
            y_all = np.concatenate((y_all, y_chamber), axis=0)

            print(type,x_all.shape[0])
            x_means=np.nanmean(x_all.reshape((-1,x_all.shape[2])),axis=0)
            x_std=np.nanstd(x_all.reshape((-1,x_all.shape[2])),axis=0)
            y_means=np.nanmean(y_all,axis=0)
            y_std = np.nanstd(y_all, axis=0)

            para_mean_std_dir = f'./data/site_global/data/ML_results/'
            np.save(para_mean_std_dir+f'{data_source}_x_mean_{depth}_{type}_{seed}.npy',x_means)
            np.save(para_mean_std_dir + f'{data_source}_x_std_{depth}_{type}_{seed}.npy', x_std)
            np.save(para_mean_std_dir + f'{data_source}_y_mean_{depth}_{type}_{seed}.npy', y_means)
            np.save(para_mean_std_dir + f'{data_source}_y_std_{depth}_{type}_{seed}.npy', y_std)

            x_train=(x_train-x_means)/(x_std+pow(10,-6))
            y_train=(y_train-y_means)/(y_std+pow(10,-6))
            x_test=(x_test-x_means)/(x_std+pow(10,-6))
            y_test=(y_test-y_means)/(y_std+pow(10,-6))
            x_validate = (x_validate - x_means) / (x_std + pow(10, -6))
            y_validate = (y_validate - y_means) / (y_std + pow(10, -6))

            x_train = torch.Tensor(x_train)
            x_test = torch.Tensor(x_test)
            y_train = torch.Tensor(y_train)
            y_test = torch.Tensor(y_test)
            x_validate = torch.Tensor(x_validate)
            y_validate = torch.Tensor(y_validate)

            x_chamber = chamber_input[type]
            y_chamber = chamber_output[type]
            chamber_all_idxs = [chamber_idx for chamber_idx in range(len(x_chamber))]
            random.seed(seed)
            chamber_test_idxs = random.sample(chamber_all_idxs, int(test_precent * len(chamber_all_idxs)))
            chamber_train_validate_idxs = list(set(chamber_all_idxs).difference(set(chamber_test_idxs)))
            random.seed(seed)
            chamber_validate_idxs = random.sample(chamber_train_validate_idxs,
                                                  int(1 / 9 * len(chamber_train_validate_idxs)))
            chamber_train_idxs = np.array(list(set(chamber_train_validate_idxs).difference(set(chamber_validate_idxs))))
            chamber_test_idxs = np.array(chamber_test_idxs)
            chamber_validate_idxs = np.array(chamber_validate_idxs)

            train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=batch_size)
            val_loader = DataLoader(TensorDataset(x_validate, y_validate), shuffle=False, batch_size=batch_size)
            test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=False, batch_size=batch_size)

            device='cpu'
            him_dim=4 #hiden state vector dimention 4
            model = IMVFullLSTM(x_train.shape[2], 1, him_dim,device=device,dropout=0.1).to(device=device)
            opt = torch.optim.Adam(model.parameters(), lr=0.01,)
            epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)

            opt_chamber = torch.optim.Adam(model.parameters(), lr=0.01, )
            epoch_scheduler_chamber = torch.optim.lr_scheduler.StepLR(opt_chamber, 1, gamma=0.9)


            epochs = 80
            loss = nn.MSELoss()
            patience = 50
            min_val_loss = 9999
            counter = 0
            para_path = f"./data/site_global/data/ML_results/para/{data_source}_CausalML_{type}_{him_dim}_seed{seed}_lag{depth}.pt"
            if os.path.exists(para_path):
                model.load_state_dict(torch.load(para_path))


            for i in range(epochs):
                mse_train = 0
                iteration_start = time.monotonic()
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device=device)
                    batch_y = batch_y.to(device=device)
                    opt.zero_grad()
                    y_pred, alphas, betas = model(batch_x)
                    y_pred = y_pred.squeeze(1)
                    l = loss(y_pred, batch_y)
                    causal_stren = np.repeat(causal_stren_raw, batch_x.size(0), axis=0)
                    causal_stren = causal_stren / np.sum(causal_stren, axis=1, keepdims=True)
                    causal_stren = torch.Tensor(causal_stren)
                    weights_temp = Variable(causal_stren, requires_grad=False).to(device=device)
                    betas = betas.squeeze(2)
                    betas=betas[:,:weights_temp.size(dim=1)]
                    causal_loss = F.kl_div(betas.log(), weights_temp, None, None, 'sum')
                    l = l + causal_punish_para * causal_loss
                    l.backward()
                    mse_train += l.item() * batch_x.shape[0]
                    opt.step()
                #############################################
                ####finetune chamber
                finetune_epochs = 3

                for finetune_epoch_idx in range(finetune_epochs):
                    for chamber_idx in chamber_train_idxs:
                        x_chamber_data = x_chamber[chamber_idx]
                        x_nan_num = np.sum(np.isnan(x_chamber_data.reshape(-1)))
                        if x_nan_num == 0:
                            y_chamber_data = y_chamber[chamber_idx]
                            x_chamber_data = (x_chamber_data - x_means) / (x_std + pow(10, -6))
                            y_chamber_data = (y_chamber_data - y_means) / (y_std + pow(10, -6))
                            x_chamber_data = torch.Tensor(x_chamber_data)
                            y_chamber_data = torch.Tensor([y_chamber_data])
                            opt_chamber.zero_grad()
                            y_pred, alphas, betas = model(x_chamber_data)
                            y_pred = y_pred.squeeze(1)
                            y_mean = torch.mean(y_pred, dim=0, keepdim=True)
                            l = loss(y_mean, y_chamber_data)
                            l.backward()
                            opt_chamber.step()
                            print('loss', l.item())
                    ####finetune chamber
                    #############################################
                epoch_scheduler.step()
                epoch_scheduler_chamber.step()
                # validate
                with torch.no_grad():
                    model.eval()
                    mse_val = 0
                    preds = []
                    true = []
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(device=device)
                        batch_y = batch_y.to(device=device)
                        output, alphas, betas = model(batch_x)
                        output = output.squeeze(1)
                        preds.append(output.cpu().numpy())
                        true.append(batch_y.cpu().numpy())
                        mse_val += loss(output, batch_y).item() * batch_x.shape[0]
                    for chamber_idx in chamber_validate_idxs:
                        x_chamber_data = x_chamber[chamber_idx]
                        x_nan_num = np.sum(np.isnan(x_chamber_data.reshape(-1)))
                        if x_nan_num == 0:
                            y_chamber_data = y_chamber[chamber_idx]
                            x_chamber_data = (x_chamber_data - x_means) / (x_std + pow(10, -6))
                            y_chamber_data = (y_chamber_data - y_means) / (y_std + pow(10, -6))
                            x_chamber_data = torch.Tensor(x_chamber_data)
                            y_chamber_data = torch.Tensor([y_chamber_data])
                            opt.zero_grad()
                            y_pred, alphas, betas = model(x_chamber_data)
                            y_pred = y_pred.squeeze(1)
                            y_mean = torch.mean(y_pred, dim=0, keepdim=True)
                            l = loss(y_mean, y_chamber_data)
                            mse_val +=l.item()
                            preds.append(y_mean.cpu().numpy())
                            true.append(y_chamber_data.cpu().numpy())
                preds = np.concatenate(preds)
                true = np.concatenate(true)

                if min_val_loss > mse_val ** 0.5:
                    min_val_loss = mse_val ** 0.5
                    print("Saving...")
                    torch.save(model.state_dict(), para_path)
                    counter = 0
                else:
                    counter += 1

                if counter == patience:
                    break
                print("Iter: ", i, "train: ", (mse_train / len(x_train)) ** 0.5, "val: ", (mse_val / len(x_train)) ** 0.5)
                iteration_end = time.monotonic()
                print("Iter time: ", iteration_end - iteration_start)
                if (i % 10 == 0):
                    preds = preds * (y_std+pow(10,-6)) + y_means
                    true = true * (y_std+pow(10,-6)) + y_means
                    mse = mean_squared_error(true, preds)
                    mae = mean_absolute_error(true, preds)
                    r=stats.pearsonr(true, preds)[0]
                    print("mse: ", mse, "mae: ", mae,'r:',r)

            # test
            mse_val = 0
            preds = []
            true = []
            alphas = []
            betas = []
            with torch.no_grad():
                model.eval()
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device=device)
                    batch_y = batch_y.to(device=device)
                    output, a, b = model(batch_x)
                    output = output.squeeze(1)
                    preds.append(output.cpu().numpy())
                    true.append(batch_y.cpu().numpy())
                    alphas.append(a.cpu().numpy())
                    betas.append(b.cpu().numpy())
                    mse_val += loss(output, batch_y).item()*batch_x.shape[0]
            ############################################################
            #######test chamber dataset

            with torch.no_grad():
                model.eval()
                x_chamber = chamber_input[type]
                y_chamber = chamber_output[type]
                IDs=chamber_ID[type]
                for chamber_idx in range(len(x_chamber)):
                    x_chamber_data = x_chamber[chamber_idx]
                    ID=IDs[chamber_idx]
                    x_nan_num = np.sum(np.isnan(x_chamber_data.reshape(-1)))
                    if x_nan_num == 0:
                        y_chamber_data = y_chamber[chamber_idx]
                        x_chamber_data = (x_chamber_data - x_means) / (x_std + pow(10, -6))
                        y_chamber_data = (y_chamber_data - y_means) / (y_std + pow(10, -6))
                        x_chamber_data = torch.Tensor(x_chamber_data)
                        y_chamber_data = torch.Tensor([y_chamber_data])
                        y_pred, a, b= model(x_chamber_data)
                        y_pred = y_pred.squeeze(1)
                        y_mean = torch.mean(y_pred, dim=0, keepdim=True)
                        y_mean=y_mean.detach().cpu().numpy()
                        y_chamber_data=y_chamber_data.cpu().numpy()
                        y_predict_item=y_mean[0]*(y_std + pow(10, -6))+y_means
                        y_true_item=y_chamber_data[0]*(y_std + pow(10, -6))+y_means
                        df_chamber.loc[df_chamber['ID']==ID,'predicted']=y_predict_item
                        df_chamber.loc[df_chamber['ID'] == ID, 'true'] = y_true_item
                        df_chamber.loc[df_chamber['ID'] == ID, 'Abs_Error'] =np.abs(y_predict_item-y_true_item)
                        df_chamber.loc[df_chamber['ID'] == ID, 'Relative_Error'] = np.abs(y_predict_item - y_true_item)/y_true_item
                        preds.append(y_mean)
                        true.append(y_chamber_data)
                        alphas.append(a.cpu().numpy())
                        betas.append(b.cpu().numpy())
            preds = np.concatenate(preds)
            true = np.concatenate(true)
            alphas = np.concatenate(alphas)
            betas = np.concatenate(betas)
            alphas = alphas.mean(axis=0)
            betas = betas.mean(axis=0)

            alphas = alphas[..., 0]
            betas = betas[..., 0]

            preds = preds*(y_std+pow(10,-6)) + y_means
            true = true*(y_std+pow(10,-6)) + y_means

            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            r=stats.pearsonr(true, preds)[0]
            r2=r2_score(true, preds)
            temp_result = ['imv', type, seed, mae, r,r2]
            temp_result.extend(betas.tolist())
            df_result.loc[df_idx] = temp_result
            df_idx += 1
            print(type,mse, mae,stats.pearsonr(true, preds)[0])

    df_chamber.to_csv(
        f'./data/site_global/data/ML_results/{data_source}_chamber_CausalML_prediction_lag{depth}.csv')

    df_final=pd.DataFrame(columns=['model','wetlandtype','mae_mean','r_mean','r2_mean','mae_std','r_std','r2_std'])
    df_idx_stats=0
    for type in site_types:
        mae=df_result[(df_result['wetlandtype']==type)]['mae'].values
        r=df_result[(df_result['wetlandtype']==type)]['r'].values
        r2=df_result[(df_result['wetlandtype']==type)]['r2'].values
        df_final.loc[df_idx_stats] = ['imv', type, np.mean(mae), np.mean(r),np.mean(r2), np.std(mae), np.std(r), np.std(r2)]
        df_idx_stats += 1

    df_final.to_csv(f'./data/site_global/data/ML_results/{data_source}_CausalML_result_mean_std_lag{depth}.csv')
    df_result.to_csv(f'./data/site_global/data/ML_results/{data_source}_CausalML_var_attn_weight_lag{depth}.csv')

