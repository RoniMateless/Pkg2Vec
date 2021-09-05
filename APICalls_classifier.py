
import csv
import ast
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score ,f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from constants import Id2Apk, Family2Id, combined_API_list


def load_api_calls():
    API_CALLS_PATH = "D:\\APK\\Authorship\\authorship_api_calls_1215.csv"
    reader = csv.DictReader(open(API_CALLS_PATH))
    results = []
    for row in reader:
        results.append(row)

    apk2APIcalls = {}
    for row in results:
        label = int(Family2Id[row['variety']]) # multiclass

        api_functions_dict = {}
        for functions_api in ast.literal_eval(row['functions api calls1']):
            api_functions_dict[functions_api] = 1
        for functions_api in ast.literal_eval(row['functions api calls2']):
            api_functions_dict[functions_api] = 1

        if apk2APIcalls.has_key((row['apk_name'], row['family'])):
            print("Error - apk already exists")
            print(row['apk_name'])  # apk already exists
        else:
            #apk2APIcalls[(row['apk_name'], row['variety'])] = (api_functions_dict, label)
            apk2APIcalls[row['apk_name']] = (api_functions_dict, label)

    print ("Total apk 2 api calls loaded count: ", len(apk2APIcalls.keys()))

    return apk2APIcalls


def get_features(ids, apk2data):
    error_count = 0
    count = 0
    data_list = []
    new_labels = []
    functions_api_calls = []
    output_ids = []
    for id in ids:
        output_ids.append(id)
        apk = Id2Apk[id][0]
        print (apk)
        if apk2data.has_key(apk):
            data_list.append(apk2data[apk][0])
            new_labels.append(apk2data[apk][1])
            count += 1
        else:
            #            print apk
            data_list.append({})
            new_labels.append(0)
            error_count += 1

    print ("Number of missing APKS: ", error_count)
    print ("Number of APKS: ", count)

    vec = DictVectorizer()

    data_vec = vec.fit_transform(data_list).toarray()
    print ("Data Features count: ", len(vec.get_feature_names()))

    print ("Data Features Names: ", vec.get_feature_names())

    return data_vec, new_labels, vec.get_feature_names(), output_ids
