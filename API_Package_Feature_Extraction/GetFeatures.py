import sys
import os
import numpy as np
import pandas as pd
import csv

PATH_INSTALL = "./"
sys.path.append(PATH_INSTALL)
from androguard.core.bytecodes import dvm, apk
from androguard.core.analysis import analysis
from androguard.decompiler.dad import decompile
from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.analysis.analysis import ExternalClass
from datetime import datetime

timestr = str(datetime.now())

STD_API_LIST_DIR = 'android_level_23_package_name.csv'
SAMPLE_DIR = "Your Directory for Android Malware Samples"
LABEL = 1

SAMPLES_LIST = "rooting_malware_hash.txt"
RESULT_DIR = "API_feature_results/"
EXPIRED_HASH_LIST = "rooting_hashes_expired_{}".format(timestr)
BATCH_SIZE = 10

csvname =  "{}{}_{}_pkgfreq.csv".format(RESULT_DIR, timestr, str(LABEL))

def ReadStandardAPIList(filename):
    reader = csv.reader(open(filename, 'rU'), dialect=csv.excel_tab)
    std_apis = []

    for name in reader:
        std_apis.append(name[0])

    return std_apis

def ProcessClassName(c_name):
    c_name = c_name.encode('utf8')
    c_name = c_name.replace(";","")
    c_name = c_name.replace("/",".")
    c_name = c_name.replace("[","")

    if c_name[0] == 'L':
       c_name = c_name[1:]
    # $ means the class is a internal class of the api
    # currently we ignore that
    if c_name.find("$") >= 0:
        c_name = ""
    return c_name

def ProcessMethodName(m_name):

    return m_name.encode('utf8')


def GetPackageNameFromClassName(c_name):
    # remove the class name
    idx = c_name.rfind('.')
    if idx < 1:
        return c_name
    return c_name[:idx]


def GetPkgFreq(dx):
    # getting external methods and call frequences
    ecs = dx.get_external_classes()
    pkg_calls = dict()
    pkg_freqs = []

    for class_analysis in ecs:
        external_c = class_analysis.get_vm_class();
        external_methods = external_c.get_methods();
    
        ec_name = external_methods[0].get_class_name()
        ec_name = ProcessClassName(ec_name)
        if (not ec_name):
            continue

        p_name = GetPackageNameFromClassName(ec_name)
        freq = 0  

        for em in external_methods:
            # method name text processing
            em_name = ProcessMethodName(em.get_name())
            if not em_name:
                continue

            e_c_a = dx.get_method_analysis(em)
            xr_f = e_c_a.get_xref_from()
            freq += len(xr_f)
    
        if p_name in pkg_calls:
            pkg_calls[p_name] += freq
        else:
            pkg_calls[p_name] = freq

    return pkg_calls


def WriteAPICall(calls, filename):
    with open(filename,'w') as f:
        for line in calls:
            f.write(str(line) + '\n')


def WriteCallFreq(call_freq, filename):
    with open(filename,'w') as f:
        for num in call_freq:
            f.write(str(num) + '\n')


def GetFeature(pkg_calls, std_apis):
    feature = [0] * len(std_apis)  
    for i in range(len(std_apis)):
        if std_apis[i] in pkg_calls:
            feature[i] = pkg_calls[std_apis[i]]
    # print(feature)
    return feature

def WriteExpiredHashes(hash):
    expired_hash_file = open(EXPIRED_HASH_LIST, 'a')
    expired_hash_file.write(hash + '\n')
    expired_hash_file.close()

def GetAnalyzedDex(sample_name):
    a = APK(sample_name)
    d = dvm.DalvikVMFormat(a.get_dex())
    dx = analysis.Analysis(d)

    if type(dx) is list:
        dx = dx[0]

    dx.create_xref()

    return dx

# @profile
def GetFeatures():

    std_apis = ReadStandardAPIList(STD_API_LIST_DIR)
    len_std_apis = len(std_apis)
    # create a data frame to store features

    col = list(std_apis)
    col.append('label')
    col.append('mw_name')
    df = pd.DataFrame(columns = col)
    with open(csvname, 'a') as f: 
        df.to_csv(f, header  = True)

    # delete the file index in the goodware.txt and change the following idx
    idx = 1
    files = open(SAMPLES_LIST,'r')
    for filen in files:
        filen = filen[:-1]
        sample_name = SAMPLE_DIR + filen
        print("processing {}: {}".format(idx, filen))

        try:
            dx = GetAnalyzedDex(sample_name)

        except:
            print("failed to analyse " + filen)
            idx += 1
            continue

        if not dx:
            print("failed to analyse " + filen)
            idx += 1
            continue

        # get pkg calls and freqs
        pkg_calls = GetPkgFreq(dx)
        del dx

        # get feature, append to df
        feature = GetFeature(pkg_calls, std_apis)
        feature.append(LABEL)
        feature.append(filen)
        df.loc[idx] = feature

        if idx % BATCH_SIZE == 0:
            with open(csvname, 'a') as f: 
                df.to_csv(f, header = False)
                df = pd.DataFrame(columns = col)
        idx += 1

# write to the final feature file
    with open(csvname, 'a') as f: 
        df.to_csv(f, header = False)
    files.close()

    print("##################### API Package Feature Extraction Completed ###################### \n")
    return

if __name__ == "__main__":
    GetFeatures()
