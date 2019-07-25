import pandas as pd
import numpy as np
import operator
import csv
from scipy import stats

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

# output: dictionary: api-sus
def GetSUS(std_apis, banker_label, good_label, API_features_with_label, method_flag, var_flag):
#   var_flag: 0:f, 1:f^2, 2:f^3, 3:f^0.5, 4:e^f, 5: ln(f)
    USE_BINARY = False
    USE_ALL_SUM = False
    if method_flag =='1':
        USE_BINARY = True
    elif method_flag =='2b':
        USE_ALL_SUM = True

    sus_dict = dict()
    API_features_with_label.astype(np.float32, inplace = True)
    df_good = API_features_with_label.loc[API_features_with_label['label'].isin([good_label])]
    df_banker = API_features_with_label.loc[API_features_with_label['label'].isin([banker_label])]
    for col in ['label', 'mw_name']:
        if col in df_good:
            df_good.drop(col, axis = 1, inplace =  True)
            df_banker.drop(col, axis = 1, inplace = True)
    if USE_ALL_SUM:
        num_good = CallSumOverAllSamples(df_good, std_apis)
        num_banker = CallSumOverAllSamples(df_banker, std_apis)
    else:
        num_good = df_good.shape[0]
        num_banker = df_banker.shape[0]

    cols = list(df_banker.columns.values)
    if var_flag == 1:
        df_banker.loc[:,cols] = np.power(df_banker.as_matrix(), 2)
        df_good.loc[:,cols] = np.power(df_good.as_matrix(), 2)

    elif var_flag == 2:
        df_banker.loc[:,:] = np.power(df_banker.as_matrix(), 3)
        df_good.loc[:,:] = np.power(df_good.as_matrix(), 3)
    elif var_flag == 3:
        df_banker.loc[:,:] = np.power(df_banker.as_matrix(), 0.5)
        df_good.loc[:,:] = np.power(df_good.as_matrix(), 0.5)
    elif var_flag == 4:
        max_freq = max(np.amax(df_banker.as_matrix()), np.amax(df_good.as_matrix()))
        min_freq = min(np.amin(df_banker.as_matrix()), np.amin(df_good.as_matrix()))
        shift = 0.5 * (max_freq + min_freq)
        df_banker.loc[:,:] = np.exp(df_banker.as_matrix()- shift)
        df_good.loc[:,:] = np.exp(df_good.as_matrix()- shift)
    elif var_flag == 5:
        shift = 1 # ensure that log result is non-negative
        df_banker.loc[:,:] = np.log(df_banker.as_matrix() + shift)
        df_good.loc[:,:] = np.log(df_good.as_matrix() + shift)
    
    for pkg in std_apis:
        if pkg not in df_banker:
            print('error!',pkg,df_banker.shape)
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
def GetSRSUSFeature(sus_dict, sr_dict, group_num, group_width, std_apis, use_freq, API_features_with_label, method_flag, var_flag):
    # median mode max weithed_sum....
    more = True

    sorted_sus = sorted(sus_dict.items(), key=operator.itemgetter(1), reverse = True)

    sorted_sr = sorted(sr_dict.items(), key=operator.itemgetter(1), reverse = True)
    API_features_with_label.reset_index(drop = True, inplace = True)
    df = pd.DataFrame()
    n_samples = API_features_with_label.shape[0]
    sr_length = len(sr_dict)
    start_idx = 0
    for i in range(group_num):
        suffix = "{}_var{}_{}".format(method_flag, str(var_flag), str(i+1))
        sus_feature_name = "sus_{}".format(suffix)
        sus_bin_feature = "sus_bin_{}".format(suffix)
        sr_feature_name = "sr_{}".format(suffix)
        sr_bin_feature = "sr_bin_{}".format(suffix)

        sus_sum = "sus_sum_{}".format(suffix)
        sus_med = "sus_median_{}".format(suffix)
        sus_max = "sus_max_{}".format(suffix)
        sus_wsm = "sus_wsum_{}".format(suffix)
        sr_sum = "sr_sum_{}".format(suffix)
        sr_med = "sr_median_{}".format(suffix)
        sr_max = "sr_max_{}".format(suffix)
        sr_wsm = "sr_wsum_{}".format(suffix)

        for key in [sus_feature_name,sus_bin_feature,sr_feature_name,sr_bin_feature,sus_sum,sus_med,sus_max,sus_wsm,sr_sum,sr_med,sr_max,sr_wsm]:
            df[key] = np.zeros(n_samples)

        sus_group, sr_group = [],[]
        sus_vl_group, sr_vl_group = [],[]
        for j in range(start_idx, start_idx + group_width):
            if j >= sr_length:
                break
                        
            # all stat
            sus_group.append(sorted_sus[j][0])
            sr_group.append(sorted_sr[j][0])
            sus_vl_group.append(sorted_sus[i][1])
            sr_vl_group.append(sorted_sr[i][1])
        
        sus_mat = API_features_with_label[sus_group].as_matrix()
        df[sus_feature_name] = np.count_nonzero(sus_mat, axis = 1)
        df[sus_bin_feature] = (sus_mat > 0).astype('int')
        df[sus_med] = np.median(sus_mat, axis = 1)
        df[sus_sum] = np.sum(sus_mat, axis = 1)
        df[sus_max] = np.max(sus_mat, axis = 1)
        df[sus_wsm] = np.dot(sus_mat, np.array(sus_vl_group))
            
        sr_mat = API_features_with_label[sr_group].as_matrix()
        df[sr_feature_name] = np.count_nonzero(sr_mat, axis = 1)
        df[sr_bin_feature] = (sr_mat > 0).astype('int')
        df[sr_med] = np.median(sr_mat, axis = 1)
        df[sr_sum] = np.sum(sr_mat, axis = 1)
        df[sr_max] = np.max(sr_mat, axis = 1)
        df[sr_wsm] = np.dot(sr_mat, np.array(sr_vl_group))

        start_idx += group_width

    df = df.astype(np.float32)

    return df
