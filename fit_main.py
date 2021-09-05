import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

import numpy as np
import csv
from sklearn.model_selection import StratifiedKFold
#from constants import Id2File, Family2Id, Id2Family, MAX_FEATURES, MAX_SEQUENCE_LENGTH, MAX_FUNCTIONS, Id2BoW ,largest_ids, largest_families, MAX_FUNCTIONS_PER_FILE, SPLITS
from constants import Id2File, Family2Id, Id2Family, MAX_FEATURES, MAX_SEQUENCE_LENGTH, MAX_FUNCTIONS,largest_ids, largest_families, MAX_FUNCTIONS_PER_FILE, MAX_FILES
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger
from keras.utils import to_categorical
from file_model import build_bow, build_long_sequence, build_half_hierarchical_model, build_1_sequence,build_cnn_above_tf, build_hierarchical_model, build_yang_model, build_tang_cnn_model, build_tang_lstm_model
from data_generator import load_cutted_pkg, load_smart_pkg, load_bow_pkg, reshape_sequences, reshape_file_sequences, X_to_one_hot
import datetime
from sklearn import metrics
from scipy import stats
from sklearn.metrics import confusion_matrix


batch_size = 32#MAX_FUNCTIONS#int(MAX_FUNCTIONS / SPLITS)#8

import matplotlib.pyplot as plt
import itertools as it
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
    plt.title(title, fontsize='xx-large')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    classes1 = [c[:10] for c in classes]
    print (classes)
    # plt.set_xticklabels(classes, rotation=(45), fontsize=10, va='bottom', ha='left')
    plt.xticks(tick_marks, classes1, rotation=45, fontsize='x-large')
    plt.yticks(tick_marks, classes1, fontsize='x-large')

    cm = np.around(cm, decimals=2)  # rounding to display in figure
    thresh = cm.max() / 2.
    #print (thresh)
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j]:  # print values different than zero
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     verticalalignment='center',
                     fontsize=12,
                     color="black" if cm[i, j] > thresh else "black")

    # fig = plt.gcf()
    # fig.set_size_inches(25, 18, forward=True)
    plt.tight_layout()
    plt.ylabel('True Label', fontsize='x-large')
    plt.xlabel('Predicted Label', fontsize='x-large')
    #plt.show()
    fig = "fig51.png"
    plt.savefig(fig, bbox_inches='tight', dpi=100)

def predict_subsampling(model, X_test, y_test):
    seq_loss_scores = []
    seq_acc_scores = []

    for X_row, y_row in zip(X_test, y_test):
        preds = model.predict(X_row, batch_size=batch_size)
        preds_mean = np.mean(preds, axis=0)

        y = to_categorical(y_row, num_classes=len(Family2Id))
        seq_loss_scores.append(metrics.log_loss(y, preds_mean))

        majority_idx = stats.mode(np.argmax(preds, axis=-1))[0][0]
        seq_acc_scores.append(float(majority_idx == y_row))

    return [np.mean(seq_loss_scores), np.mean(seq_acc_scores)]

def main():
    print (datetime.datetime.now())

    #callbacks
    #early_stopCB = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    early_stopCB = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    # tbCB = TensorBoard(log_dir='Graph', histogram_freq=0,write_graph=True, write_images=True)
    #for i in [1000,3000,5000,8000,10000]:
    #    MAX_FUNCTIONS = i
    #load data
    #X = np.zeros((len(Id2File.keys()), MAX_FUNCTIONS, MAX_SEQUENCE_LENGTH), dtype='float32')#int32
    X = np.zeros((len(Id2File.keys()), MAX_FILES, MAX_FUNCTIONS_PER_FILE, MAX_SEQUENCE_LENGTH), dtype='float32')  # int32
    y = np.zeros((len(Id2File.keys())), dtype='float32')#int
    X_BoW = np.zeros((len(Id2File.keys()), MAX_FEATURES), dtype='float32')#int
    #X_API_Calls = np.zeros((len(Id2File.keys()), MAX_FEATURES), dtype='float32')  # int

    for i, (k, v) in enumerate(Id2File.items()):
        file_path = v[0]
        #X[int(i),] = load_cutted_pkg(file_path)
        X[int(i),] = load_smart_pkg(file_path)
        #X_BoW[int(i),] = np.array(Id2BoW[k])
        #X_API_Calls[int(i),] = np.array(Id2APICalls[k])
        y[int(i)] = Family2Id[v[2]]
    #X = X.reshape(X.shape[0], -1)# for 1 sequence
    #X = X_to_one_hot(X) # try one hot encoder
    #print ("X Shape: ", X.shape)
    print ("X_BoW Shape: ", X_BoW.shape)

    # Write results to a file
    f = open('amd_test11.csv', 'a')
    #fields = ('Method', 'Configuration', 'Family', 'Classifier', 'Accuracy', 'best features')
    writer = csv.writer(f, lineterminator='\n')
    # writer.writerow(fields)

    n_folds = 3
    train_test = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_i = 0
    acc_scores = []
    loss_scores = []
    f1_scores = []
    total_cm = 0.0
        #training_index, testing_index = next(train_test.split(X,y))
    for training_index, testing_index in train_test.split(X,y):
        fold_i += 1
        print(datetime.datetime.now())
        print ("Running Fold", fold_i, "/", n_folds)

        print ("train_index length: ", len(training_index))
        print ("test_index length: ", len(testing_index))
        X_training, X_test = X[training_index], X[testing_index]
        y_training, y_test = y[training_index], y[testing_index]
        #X_training_BoW, X_test_BoW = X_BoW[training_index], X_BoW[testing_index]
        #X_training_API, X_test_API = X_API_Calls[training_index], X_API_Calls[testing_index]

        # Split Train to Train/Validation
        train_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        train_index, val_index = next(train_val.split(X_training, y_training))
        X_train, X_val = X_training[train_index], X_training[val_index]
        y_train, y_val = y_training[train_index], y_training[val_index]
        #X_train_BoW, X_val_BoW = X_training_BoW[train_index], X_training_BoW[val_index]
        #X_train_API, X_val_API = X_training_API[train_index], X_training_API[val_index]


        '''
        # for long LSTM sequences
        X_train, y_train = reshape_sequences(X_train, y_train)
        X_val, y_val = reshape_sequences(X_val, y_val)
        print ("X_train Shape: ", X_train.shape)
        print ("y_train Shape: ", y_train.shape)
        '''
        y_tr_cat = to_categorical(y_train, num_classes=len(Family2Id))
        y_v_cat = to_categorical(y_val, num_classes=len(Family2Id))
        y_te_cat = to_categorical(y_test, num_classes=len(Family2Id))
        '''
        # for TF-BPTT sequences
        X_train, y_train = reshape_file_sequences(X_train, y_train)
        X_val, y_val = reshape_file_sequences(X_val, y_val)
        print ("X_train Shape: ", X_train.shape)
        print ("y_train Shape: ", y_train.shape)
        '''

        # Train model on dataset
        print ("Build model")
        #model = build_long_sequence()
        #model = build_1_sequence()
        #model = build_half_hierarchical_model()
        model = build_hierarchical_model()
        #model = build_tang_cnn_model()
        #model = build_tang_lstm_model()
        #model = build_bow()
        #model = build_cnn_above_tf()
        #model = build_kim_model()
        #model = build_long_sequence_with_3inputs()

        mcp_save1 = ModelCheckpoint('.m1.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        print ("fit...")
        hist = model.fit(x=X_train,
        #hist = model.fit(x=[X_train, X_train_BoW, X_train_API],
        #hist = model.fit(x=[X_train,X_train_API],
        #hist = model.fit(x=[X_train,X_train_BoW],
        #hist = model.fit(x=X_train_BoW,
        #                 y=[y_tr_cat, y_tr_cat, y_tr_cat],
                         #y=to_categorical(y_train, num_classes=len(Family2Id)),
                         y=to_categorical(y_train, num_classes=len(Family2Id)),
                         batch_size=batch_size,
                         epochs=1000,
                         verbose=0,
                         callbacks=[early_stopCB, mcp_save1],
                         #validation_data=[[X_val, X_val_BoW, X_val_API], to_categorical(y_val, num_classes=len(Family2Id))],
                         #validation_data=[[X_val, X_val_API], to_categorical(y_val, num_classes=len(Family2Id))],
                         #validation_data=[[X_val, X_val_BoW], [y_v_cat, y_v_cat, y_v_cat]],
                         validation_data=[X_val, to_categorical(y_val, num_classes=len(Family2Id))],
                         #validation_data=[X_val_BoW, to_categorical(y_val, num_classes=len(Family2Id))],
                         shuffle=True)


        # load best model
        model.load_weights(filepath='.m1.hdf5')
        #scores = model.evaluate(x=X_test_BoW, y=to_categorical(y_test, num_classes=len(Family2Id)), batch_size=batch_size, verbose=1)
        scores = model.evaluate(x=X_test, y=to_categorical(y_test, num_classes=len(Family2Id)), batch_size=batch_size, verbose=1)
        #scores = model.evaluate(x=[X_test, X_test_BoW], y=[y_te_cat, y_te_cat, y_te_cat],batch_size=batch_size, verbose=1)
        #scores = model.evaluate(x=[X_test, X_test_BoW], y=y_te_cat, batch_size=batch_size, verbose=1)
        #scores = model.evaluate(x=[X_test, X_test_BoW, X_test_API], y=to_categorical(y_test, num_classes=len(Family2Id)),batch_size=batch_size, verbose=1)
        #scores = predict_subsampling(model, X_test, y_test)

        #predicted = model.predict(X_test_BoW, batch_size=batch_size)
        #total_cm += confusion_matrix(y_test, np.argmax(predicted, axis=1), labels=range(len(Family2Id.keys())))

        loss_scores.append(scores[0])
        acc_scores.append(scores[1] * 100)
        f1_scores.append(scores[2] * 100)
        print("%s: %.3f" % (model.metrics_names[0], scores[0]))
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))

        row_measures = [fold_i, scores[1] * 100, scores[0]]
        writer.writerow(row_measures)

    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(acc_scores), np.std(acc_scores)))
    print("Loss: %.3f (+/- %.3f)" % (np.mean(loss_scores), np.std(loss_scores)))
    print("F1: %.2f%% (+/- %.2f%%)" % (np.mean(f1_scores), np.std(f1_scores)))

    #avg_acc = np.trace(total_cm) / float(np.sum(total_cm))
    #print (avg_acc)

    #cm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
    #reduced_normalized_cm = cm[largest_ids, :][:, largest_ids]

    # Plot normalized confusion matrix
    #plot_confusion_matrix(reduced_normalized_cm, classes=largest_families, normalize=True,title='Normalized Confusion Matrix (%s)' % "Pkg2vec")
    '''
    print (cm)
    thres = 0.1
    
    for i in range(cm.shape[0]):
        row_list = []
        for j in range(cm.shape[1]):
            if cm[i, j] > thres:
                row_list.append((str(round(cm[i, j], 2)), Id2Family[j]))
        row_list.sort(reverse=True)
        row_measures = [Id2Family[i], str(round(cm[i, i], 2)), row_list]
        #print row_measures
        writer.writerow(row_measures)
    '''
    '''
    reudced_cm = 0.0#[[0] for i in largest_families]
    x=0
    y=0
    for i in range(total_cm.shape[0]):
        if Id2Family[i] in largest_families:
            for j in range(total_cm.shape[1]):
                if Id2Family[j] in largest_families:
                    reudced_cm[x,y] = total_cm[i,j]
                    y+=1
            x+=1
    print reudced_cm
    '''

    row_measures = [n_folds, np.mean(acc_scores), np.std(acc_scores), np.mean(loss_scores), np.std(loss_scores)]
    writer.writerow(row_measures)

    #file.close()


if __name__ == '__main__':
    main()
