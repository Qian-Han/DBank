from libs.SUS_SR_feature_fun import *

def GenAllSR(API_ftrs_for_train, applied_API_ftrs, aja_matrix, std_apis, group_width):
    banker_label = 1
    nonbanker_label = 0

    use_freq = True

    method_flags = ['1','2a','2b']
    df = None
    first_flag = True
    
    for method in method_flags:
        var = range(0,6)
        if method == '1':
            var = [0]
        for var_flag in var:
            if var_flag == 4:
                continue
            # API_ftrs_for_train: API_feature + Label
            # First step: Training samples
            # Second step: add test data and predicted label
            # Third step: update predicted label
            sus_dict = GetSUS(std_apis, banker_label, \
                nonbanker_label, API_ftrs_for_train, \
                method, var_flag)
            # return SUS for each API package
            sr_dict = GetSR(sus_dict, aja_matrix, std_apis)

            # compute group num according to valid SR list
            group_num = int(len(sr_dict) / group_width)

            current_df = GetSRSUSFeature(sus_dict, sr_dict, \
                group_num, group_width, \
                std_apis, use_freq, \
                applied_API_ftrs, \
                method, var_flag)

            if first_flag:
                first_flag = False
                df = current_df
            else:
                df = pd.concat([df, current_df], axis = 1, \
                    join_axes = [df.index])

    print('SR features shape', df.shape)

    return df

