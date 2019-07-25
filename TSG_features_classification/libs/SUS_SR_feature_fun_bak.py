import pandas as pd
import numpy as np
import operator
import csv

def GetAPIList(API_features_with_label):
    cols = list(API_features_with_label.columns.values)
    std_apis = []

    print('num of cols', len(cols))
    for col in cols:
        if col != 'label' and col != 'mw_name':
            std_apis.append(col)
    print('num of std_apis', len(std_apis))
    return std_apis

def IndicatorSum(feature):
    return np.nonzero(feature)[0].shape[0]

def CallSum(feature):
    return np.sum(feature)

def CallSumOverAllSamples(samples, std_apis):
    s = 0
    for pkg in std_apis:
        s+=CallSum(samples[pkg])
    return s

def IsPkgSparse(pkg, thres):
    return IndicatorSum(df_good[pkg])/float(df_good.shape[0]) < thres and IndicatorSum(df_banker[pkg])/float(df_banker.shape[0]) < thres    

def GetPowerMatrix(aja_mat, sus_list):
    num_pkg = aja_mat.shape[0]
    res = np.zeros((num_pkg, num_pkg))
    aja_trans = aja_mat.transpose()
#   handle sink nodes
#   https://stackoverflow.com/questions/24829829/the-sum-of-my-page-ranks-converge-at-0-9
    for i in range(num_pkg):
        if(sum(aja_trans[:,i]) == 0):
            aja_trans[:,i] = np.ones((num_pkg,1)).reshape(num_pkg)
            aja_trans[i,i] = 0
    
    normalize_vector = np.ones((1,num_pkg)).dot(aja_trans)
    
    for i in range(num_pkg):
        res[:,i] = aja_trans[:,i] * sus_list[0,i] / float(normalize_vector[0,i])
    return res

# remove nodes of -1 sus value, singular nodes, self-linked edges
def RemoveInvalidNodes(aja_matrix, std_api_prevalence, std_apis):
#     self-linked edges
    for i in range(aja_matrix.shape[0]):
        aja_matrix[i,i] = 0

    for idx in range(len(std_apis)):
        api = std_apis[idx]
        if std_api_prevalence[api] < 0:
            aja_matrix[idx,:] = 0
            aja_matrix[:,idx] = 0

    remove_id = []
    valid_apis = []
    pre_apis = std_apis

#   remove singular nodes recursively
    has_singular = True
    test_iter = 0

    while has_singular:
        test_iter +=1
        has_singular = False
        remove_id = []
        valid_apis = []
        for i in range(aja_matrix.shape[0]):
            if not (len(aja_matrix[:,i].nonzero()[0]) or len(aja_matrix[i,:].nonzero()[0])):
                remove_id.append(i)
                has_singular = True
            else:
                valid_apis.append(pre_apis[i])
        aja_matrix = np.delete(aja_matrix,remove_id,0)
        aja_matrix = np.delete(aja_matrix,remove_id,1)
        pre_apis = valid_apis

    return valid_apis, aja_matrix


def PrintSortedSUS(sus_dict):
    sorted_x = sorted(sus_dict.items(), key=operator.itemgetter(1), reverse = True)
    for key, value in sorted_x:
        print(key,value)

def PrintSortedSR(sus_rk,valid_apis):
    sorted_idx = np.argsort(sus_rk)[::-1]
    sorted_name = np.array(valid_apis)[sorted_idx]

    print('sum of sr',np.sum(sus_rk))
    for i in range(num_pkg):
        print(sorted_name[i], sus_rk[sorted_idx[i]], feature_importance[sorted_name[i]])


# input: list of hashes, same order with the input feature
#.       list of standard apis

# output: dictionary: api-sus
def GetSUS(std_apis, banker_label, good_label, API_features_with_label, method_flag):
    USE_BINARY = False
    USE_ALL_SUM = False

    if method_flag =='1':
        USE_BINARY = True
    elif method_flag =='2b':
        USE_ALL_SUM = True

    sus_dict = dict()
    df_good = API_features_with_label.loc[API_features_with_label['label'].isin([good_label])]
    df_banker = API_features_with_label.loc[API_features_with_label['label'].isin([banker_label])]

    if USE_ALL_SUM:
        num_good = CallSumOverAllSamples(df_good, std_apis)
        num_banker = CallSumOverAllSamples(df_banker, std_apis)
    else:
        num_good = df_good.shape[0]
        num_banker = df_banker.shape[0]
    for pkg in std_apis:
        if pkg not in df_banker:
            print('error!')
            continue
            
        banker_pkg_feature = df_banker[pkg]
        good_pkg_feature = df_good[pkg]

        if USE_BINARY:
            num_banker_calls = IndicatorSum(banker_pkg_feature)
            num_good_calls = IndicatorSum(good_pkg_feature)
        else:
            num_banker_calls = CallSum(banker_pkg_feature)
            num_good_calls = CallSum(good_pkg_feature)

        if num_banker_calls ==0 and num_good_calls ==0:
            # define prevalence as -1????
            sus_dict[pkg] = -1
        else:
            norm_good_calls = float(num_good_calls) / num_good
            norm_banker_calls = float(num_banker_calls) / num_banker
            sus_dict[pkg] = norm_banker_calls / (norm_banker_calls + norm_good_calls)
        
    return sus_dict

# compute sr
# input: method
#       method_flag,1 for '1',2 for '2a, 3 for '2b'
# output: dictionary: api-sr
def GetSR(sus_dict, aja_matrix, std_apis):
    valid_apis, aja_matrix = RemoveInvalidNodes(aja_matrix, sus_dict, std_apis)

    delta = 0.85

    num_pkg = len(valid_apis)
    sus_list = np.zeros((1,num_pkg))
    for i in range(num_pkg):
        sus_list[0,i] = sus_dict[valid_apis[i]]
        if sus_list[0,i] < 0:
            print('###error in remove invalid nodes!')
            sus_list[0,i] = 0.
    # initialize sus rank
    sus_rk = sus_list.transpose() / num_pkg
    power_mat = GetPowerMatrix(aja_matrix, sus_list)
    cst_mat = ((1-delta)/float(num_pkg)) * np.ones((num_pkg, 1))

    # algebra method
    tmp = np.identity(num_pkg) - delta * power_mat
    sus_rk = np.linalg.solve(tmp, cst_mat)
    sus_rk = sus_rk.reshape(num_pkg)
    
    return dict(zip(valid_apis, sus_rk))


#       method_flag,1 for '1',2 for '2a, 3 for '2b'
#       return the feature df
def GetSRSUSFeature(sus_dict, sr_dict, group_num, group_width, std_apis, use_freq, API_features_with_label,df_sample_names):
    
    sorted_sus = sorted(sus_dict.items(), key=operator.itemgetter(1), reverse = True)
    sorted_sr = sorted(sr_dict.items(), key=operator.itemgetter(1), reverse = True)

    df = pd.DataFrame()
    n_samples = API_features_with_label.shape[0]
    sr_length = len(sr_dict)
    start_idx = 0
    freq_features = []

    for i in range(group_num):
        sus_feature_name = "sus_{}".format(str(i+1))
        sus_bin_feature = "sus_bin_{}".format(str(i+1))
        sr_feature_name = "sr_{}".format(str(i+1))
        sr_bin_feature = "sr_bin_{}".format(str(i+1))
        if use_freq:
            freq_features.append(sus_feature_name)
            freq_features.append(sr_feature_name)
        df[sus_bin_feature]= np.zeros(n_samples)
        df[sr_bin_feature]= np.zeros(n_samples)
        df[sus_feature_name]= np.zeros(n_samples)
        df[sr_feature_name] = np.zeros(n_samples)

        for j in range(start_idx, start_idx + group_width):
            if j >= sr_length:
                break
            sr = sorted_sr[j][0]
            sus = sorted_sus[j][0]

#             frequency features
            call_sus_samples = API_features_with_label[sus].nonzero()[0]
            call_sr_samples = API_features_with_label[sr].nonzero()[0]
            if use_freq:
                df.loc[call_sus_samples,sus_feature_name] += API_features_with_label.loc[call_sus_samples,sus]
                df.loc[call_sr_samples,sr_feature_name] += API_features_with_label.loc[call_sr_samples,sr]
            else:
                df.loc[call_sus_samples,sus_feature_name] += 1
                df.loc[call_sr_samples,sr_feature_name] += 1
#           binary features
            df.loc[call_sus_samples,sus_bin_feature] = 1
            df.loc[call_sr_samples,sr_bin_feature] = 1
        start_idx += group_width

    if use_freq:
#         normalization
        samples_call_sum = API_features_with_label[std_apis].sum(axis = 1)
        df[freq_features] = df[freq_features].divide(samples_call_sum, axis = 'index')

#     add mw_name and label
    df = df.astype(np.float32)
    df['mw_name'] = API_features_with_label['mw_name']
    df_final = pd.merge(df_sample_names.to_frame(),df,how = 'left', on = 'mw_name')

    return df_final
