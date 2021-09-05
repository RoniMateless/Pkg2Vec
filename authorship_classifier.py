import ntpath
import os
import numpy as np
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

#from constants import Id2Apk, Id2APICalls, Id2File, MAX_FEATURES, MAX_SEQUENCE_LENGTH, Family2Id, ID_TF_MATRIX_PATH,  combined_API_list, reserved_words, family_tokens,largest_families, largest_ids, VOC_PATH, comment_remover
from constants import Id2File, MAX_FEATURES, MAX_SEQUENCE_LENGTH, Family2Id, largest_families, largest_ids, VOC_PATH, comment_remover, ID_TF_MATRIX_PATH
import matplotlib.pyplot as plt
#from APICalls_classifier import load_api_calls, get_features

import itertools as it
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import operator
import json
import re
import random



clf_names = [
#            "MLP",
            "Random Forest"
            #"Naive Bayes"
]

classifiers = [
#    MLPClassifier(alpha=1),
    RandomForestClassifier(n_estimators=50)#, max_depth=5)
#    MultinomialNB(alpha=0.01)
]



methods_names = [
            "Source Code"
 #           "API Calls"
#            "Ensemble"
]
def replace_tokens(s):

    s_API = set(combined_API_list)
    s_RV = set(reserved_words)
    s_FT = set(family_tokens)

    pattern = re.compile(r'import\s([^\s]+)') # packages
    def replacer(m):
        s = m.group(1)
        #print s
        return ''

    def sub(m):
        if m.group() in s_RV:
            return 'ReservedWord'
        elif m.group() in s_FT:
            return 'FamilyToken'
        else:
            return m.group()
    def subAPI(m):
        return 'APICall' if m.group() in s_API else m.group()  # APICall

    s = re.sub('\d+', '', s)  # numerics
#    s = re.sub(r'\w+', subAPI, s) #API
#    s = re.sub(pattern, replacer, s) # Packages
#    s = re.sub(r'\w+', sub, s) # reserved words and family token

    return s
import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')

def write_file(str_list, file_path, extra):
    file_str = '----------------------------------------'.join(str_list)
    new_path = file_path.replace('Sources_1215', 'SubSources_1215_mul3')
    new_new_path = new_path.replace('mergedWhole.java_useable.java', 'mergedWhole.java_useable' + extra + '.java')
    with safe_open_w(new_new_path) as f:
        f.write(file_str)
    return new_new_path, file_str

def read_concrete_files(max_files_per_family, first_elem, last_elem):
    X = []
    y = []
    ids = []
    counter=0
    counter1=0
    family2counter = defaultdict(int)
    Id2File2 = {}
    for k,v in Id2File.items():
        if int(k)<last_elem and int(k)>=first_elem:
            file_path = v[0]
            apk_name = v[1]
            family_name = v[2]
            if family2counter[family_name] >= max_files_per_family:#500
                continue
            with open(file_path,errors='ignore',mode='r') as fp:
                file_str = fp.read()

            family2counter[family_name] += 1

            counter += 1
            if counter % 1000 == 0:
                print ("Counter:", counter)

            file_str = comment_remover(file_str)
            #file_str = replace_tokens(file_str)
            '''
            file_str1 = file_str.split('----------------------------------------')
            indexes_l = random.sample(range(0, len(file_str1)), min(3000, len(file_str1)))
            file_str_l = [file_str1[i] for i in indexes_l]
            file_1 = file_str_l[:int(len(file_str_l)/3)]
            file_2 = file_str_l[int(len(file_str_l) / 3):int(len(file_str_l) / 1.5)]
            file_3 = file_str_l[int(len(file_str_l) / 1.5):]
            #print(file_path, len(file_str1), len(file_str_l))
            path, file_str = write_file(file_1, file_path, '_1')
            Id2File2[counter1] = (path, apk_name, family_name)
            ids.append(counter1)
            X.append(file_str)
            y.append(Family2Id[v[2]])
            counter1 += 1
            path, file_str = write_file(file_2, file_path, '_2')
            Id2File2[counter1] = (path, apk_name, family_name)
            ids.append(counter1)
            X.append(file_str)
            y.append(Family2Id[v[2]])
            counter1+=1
            path, file_str = write_file(file_3, file_path, '_3')
            Id2File2[counter1] = (path, apk_name, family_name)
            ids.append(counter1)
            X.append(file_str)
            y.append(Family2Id[v[2]])
            counter1+=1
            '''


            ids.append(k)
            X.append(file_str)
            y.append(Family2Id[v[2]])
    print(family2counter)
    #json.dump(Id2File2, open(".\\constants\\idFile_1215_mul3.txt", 'w'))
    return X,y,ids

def read_source_code(work_dir):
    X = []
    y = []
    #X = ['' for i in range(len(Family2Id.keys()))]
    #y = [999 for i in range(len(Family2Id.keys()))]
    count_files = 0
    #family2count = defaultdict(list)#{}

    for src_dir, dirs, files in os.walk(work_dir):
        for file_ in files:
            if file_ == "mergedWholeuseable.java":#"mergedSpecific.java":
                src_file = os.path.join(src_dir, file_)
                family_name = src_dir.strip().split("\\")[-2]
                #family2count[family_name] = family2count.get(family_name, 0) + 1
                #if family2count[family_name] > 5:
                #    continue
                if count_files % 100 ==0:
                    print (count_files)
                count_files += 1
                #family2count[family_name].append((os.stat(src_file).st_size, src_dir))
                with open(src_file, 'r') as fp:
                    file_str = fp.read()
                file_str = comment_remover(file_str)
                #file_str = replace_tokens(file_str)
                X.append(file_str)
                #id = int(Family2Id[family_name])
                #X[id] = " ".join([X[id], file_str])
                #y[id] = id
                y.append(int(Family2Id[family_name]))
    #print family2count
    '''
    with open('dict1.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in family2count.iteritems():
            list_size = [x[0] for x in value]
            list_size.insert(0, key)
            list_path = [x[1] for x in value]
            list_path.insert(0, key)

            writer.writerow(list_size)
            writer.writerow(list_path)
    '''
    print ('files number: ', count_files)
    return X,y


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    np.set_printoptions(precision=2)

    plt.figure(figsize=(16, 16))
    #plt.figure()
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(title)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.set_xticklabels(classes, rotation=(45), fontsize=10, va='bottom', ha='left')
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    cm = np.around(cm, decimals=2)  # rounding to display in figure
    thresh = cm.max() / 2.
    print (thresh)
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j]:  # print values different than zero
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     verticalalignment='center',
                     fontsize=10,
                     color="black" if cm[i, j] > thresh else "black")

    # fig = plt.gcf()
    # fig.set_size_inches(25, 18, forward=True)
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()



# run classifiers with 10 folds CV
families_list = [i[2] for i in Id2File.values()]
labels_list = [Family2Id[i] for i in families_list]

acc_scores = []
loss_scores = []


n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
import yaml

ROC_list = []
for method_name in methods_names:
    dim = 0
    if method_name == 'Source Code':
        max_features = 1000
        #dir = "D:\\APK\\Authorship\\Sources\\Decompiled_src"
        #X, y = read_source_code(dir)

        #X = vec.fit_transform(X)
        #writer.writerow(['IDs'] + vec.get_feature_names())
        #for idx, id in enumerate(ids):
        #    list1 = [id] + X[idx].toarray()[0].tolist()
        #    writer.writerow(list1)
        #writer.writerows(X.toarray())
        #feature_names = vec.get_feature_names()
        #json.dump(vec.vocabulary_, open(VOC_PATH, 'w'))
        #print vec.vocabulary_
        #print(feature_names)
        #print vec.get_feature_names()
        #print X.toarray()
        #y = np.array(y)
        dim = max_features
    elif method_name == 'API Calls':
        '''
        apk2APIcalls = load_api_calls()
        X, y, feature_names, ids = get_features(Id2File.keys(), apk2APIcalls)
        y = np.array(y)
        dim = 1000
        fs = SelectKBest(score_func=chi2, k=dim).fit(X, y)
        X = fs.transform(X)
        list1 = []
        for idx, id in enumerate(ids):
            list1 = [id] + X[idx].tolist()
            writer.writerow(list1)
        '''

    else: #ensemble
        file = "D:\\AMData\\ComplexLab\\run\\binary.csv"
        X_Source_Code, y, sc_feature_names = read_source_code(file)
        #apk2APIcalls = load_api_calls()
        #X_API_Calls, labels1, api_feature_names = get_features(ids, apk2APIcalls)
        X_Permissions, labels1, per_feature_names  = get_features(ids, apk2perm)
        X = np.concatenate((X_Source_Code, X_Permissions), axis=1)
        feature_names = sc_feature_names+per_feature_names
        dim = 1000

    #print "X Shape: ", X.shape
    #print "y Shape: ", y.shape

X1, y1, ids = read_concrete_files(500, 0, 1000000)

vec = TfidfVectorizer(decode_error='ignore', use_idf=False, max_features=max_features,ngram_range=(2,2),
                      analyzer='word', token_pattern=u'(?u)\\b\\w+\\b')
#X1 = vec.fit_transform(X1)
vec.fit(X1)
step=1000

f = open(ID_TF_MATRIX_PATH, 'a')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(['IDs'] + vec.get_feature_names())

for i in range(0, len(Id2File.keys()), step):
    X, y, ids = read_concrete_files(10000, i, i+step)
    X = vec.transform(X)
    y = np.array(y)

    #sum_words = X.sum(axis=0)
    #words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    #words_tup =sorted(words_freq, key = lambda x: x[1], reverse=True)
    #words_freq = [x[0] for x in words_tup]
    #print ("Sorted:", words_freq)
    print ("Original:", vec.get_feature_names())

    ids2dim = np.array(ids).reshape(-1,1)
    #fields = np.array(['IDs'] + words_freq).reshape(1,-1)

    #sorted_indexes = [vec.get_feature_names().index(w) for w in words_freq]

    #print ("Sorted_index:", sorted_indexes)
    #sort_l = [0 for i in range(len(sorted_indexes))]
    #for i, o in enumerate(sorted_indexes):
    #    sort_l[o] = i

    #sortedX = []
    #for x in X.toarray().tolist():
    #    sortedX.append([i  for _,i in sorted(zip(sort_l,x))])

    #ids_X = np.append(ids2dim, sortedX, axis=1)
    ids_X = np.append(ids2dim, X.toarray().tolist(), axis=1)
    #fields_X = np.append(fields, ids_X, axis=0)

    #f = open(ID_TF_MATRIX_PATH, 'a')
    #writer = csv.writer(f, lineterminator='\n')
    #writer.writerows(fields_X)
    writer.writerows(ids_X)
#Id2Voc = {}
#counter=1
#Id2Voc = {i+1:f for i, f in enumerate(words_freq)}
#print (Id2Voc)
#json.dump(Id2Voc, open(VOC_PATH, 'w'))
#

'''
f1 = open('sorted.csv', 'a')
writer = csv.writer(f1, lineterminator='\n')
writer.writerow(['IDs'] + words_freq)

for idx, id in enumerate(ids):
    list1 = [id] + X[idx].toarray()[0].tolist()
    writer.writerow(list1)
'''
# writer.writerows(X.toarray())
# feature_names = vec.get_feature_names()
# json.dump(vec.vocabulary_, open(VOC_PATH, 'w'))
'''
for clf_name, clf in zip(clf_names, classifiers):
    total_cm = 0.0
    fold_i = 0
    for train_index, test_index in skf.split(X, y):
        fold_i += 1
        print "Running Fold", fold_i, "/", n_folds
        #print train_index

#        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #X_train = vec.fit_transform(X_train)
        #X_test = vec.transform(X_test)


        #X_train = np.zeros((len(train_index), MAX_FEATURES), dtype='float32')
        #y_train = np.zeros((len(train_index)), dtype='float32')


        # Feature Selection
        
        fs = SelectKBest(score_func=chi2, k=dim).fit(X_train, y_train)
        X_train = fs.transform(X_train)
        # print "X After Selection: ", X_train.shape
        X_test = fs.transform(X_test)
        mask = fs.get_support()  # list of booleans
        scores = fs.scores_  # list of ...
        
        #Classification
        _ = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        total_cm += confusion_matrix(y_test, predicted, labels=range(len(Family2Id.keys())))
    
    new_features_2_score = {}
    for idx, bool in enumerate(mask):
        if bool:
            new_features_2_score[feature_names[idx]] = scores[idx]
    #print new_features_2_score # check the source
    sorted_row = sorted(new_features_2_score.items(), key=operator.itemgetter(1), reverse=True)
    selected_tokens = [i[0] for i in sorted_row[:20]]
    print selected_tokens
    
    avg_acc = np.trace(total_cm) / float(np.sum(total_cm))
    row_measures = [clf_name, avg_acc]#, selected_tokens]
    print(row_measures)
    #writer.writerow(row_measures)

    cm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]

    reduced_non_normalized_cm = total_cm[largest_ids,:][:, largest_ids]
    reduced_normalized_cm = cm[largest_ids, :][:, largest_ids]
    print reduced_non_normalized_cm

    plot_confusion_matrix(reduced_non_normalized_cm, classes=largest_families, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(reduced_normalized_cm, classes=largest_families, normalize=True,title='Normalized Confusion Matrix (%s)' % clf_name)
    
f.close()
'''