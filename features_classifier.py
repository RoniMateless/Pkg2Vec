import numpy as np
from constants import Id2File, MAX_FEATURES, DOC2VEC_FEATURES, Id2BoW, Family2Id, features_names, largest_families, largest_ids
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, log_loss, f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import operator

import matplotlib.pyplot as plt
import itertools as it
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB



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
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False)  # ticks along the top edge are off


    cm = np.around(cm, decimals=2)  # rounding to display in figure
    thresh = cm.max() / 2.
    #print (thresh)
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

clf_names = [
#            "MLP"
#            'DecisionTreeClassifier',
#            'LogisticRegression',
            "Random Forest",
#            "LinearSVC"
#            "Naive Bayes"
]


classifiers = [
#    MLPClassifier(hidden_layer_sizes=(125,), max_iter=1000 ,batch_size=32 )
#    DecisionTreeClassifier(random_state=0),
#    LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),
    RandomForestClassifier(n_estimators=50),# max_depth=5)
    #LinearSVC(random_state=0, tol=1e-5)
#    MultinomialNB(alpha=0.01)
]

#X = np.zeros((len(Id2File.keys()), DOC2VEC_FEATURES), dtype='float32')
X = np.zeros((len(Id2File.keys()), MAX_FEATURES), dtype='float32')
y = np.zeros((len(Id2File.keys())), dtype='float32')#int

for i, (k, v) in enumerate(Id2File.items()):
    X[int(i),] = np.array(Id2BoW[k])
    y[int(i)] = Family2Id[v[2]]

print ("X Shape: ", X.shape)

n_folds = 10
train_test = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=42)

for clf_name, clf in zip(clf_names, classifiers):
    total_cm = 0.0
    fold_i = 0
    acc_scores = []
    loss_scores = []
    f1_scores = []
    pre_scores = []
    recall_scores = []
    accuracy_scores = []

    for train_index, test_index in train_test.split(X ,y):
        fold_i += 1
        #print ("Running Fold", fold_i, "/", n_folds)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Feature Selection

        fs = SelectKBest(score_func=chi2, k=1000).fit(X_train, y_train)
        X_train = fs.transform(X_train)
        # print "X After Selection: ", X_train.shape
        X_test = fs.transform(X_test)
        #mask = fs.get_support()  # list of booleans
        #scores = fs.scores_  # list of ...

        # Classification
        _ = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        #p_predicted = clf.predict_proba(X_test)

        #loss_scores.append(log_loss(y_test, p_predicted, labels=range(len(Family2Id.keys()))))
        accuracy_scores.append(accuracy_score(y_test, predicted))
        f1_scores.append(f1_score(y_test, predicted, average='micro'))
        total_cm += confusion_matrix(y_test, predicted, labels=range(len(Family2Id.keys())))

    '''
    new_features_2_score = {}
    for idx, bool in enumerate(mask):
        if bool:
            new_features_2_score[features_names[idx]] = scores[idx]
    print (new_features_2_score) # check the source
    sorted_row = sorted(new_features_2_score.items(), key=operator.itemgetter(1), reverse=True)
    selected_tokens = [i[0] for i in sorted_row[:50]]
    print (selected_tokens)
    '''
    avg_acc = np.trace(total_cm) / float(np.sum(total_cm))
    row_measures = [clf_name, avg_acc, np.mean(accuracy_scores), np.std(accuracy_scores), np.mean(loss_scores), np.std(loss_scores),  np.mean(f1_scores),np.std(f1_scores)]

    print(row_measures)
    # writer.writerow(row_measures)

    cm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
    
    #reduced_non_normalized_cm = total_cm[largest_ids,:][:, largest_ids]
    reduced_normalized_cm = cm[largest_ids, :][:, largest_ids]
    #print (reduced_non_normalized_cm)
    
    #plot_confusion_matrix(reduced_non_normalized_cm, classes=largest_families, title='Confusion matrix, without normalization')
    
    # Plot normalized confusion matrix
    print(largest_families)
    np.save('confusionAMD_M', reduced_normalized_cm)
    plot_confusion_matrix(reduced_normalized_cm, classes=largest_families, normalize=True,title='Normalized Confusion Matrix (%s)' % clf_name)
