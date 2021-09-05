import numpy as np
import csv
from sklearn.model_selection import StratifiedKFold
from constants import Id2File, Family2Id, MAX_FEATURES, MAX_SEQUENCE_LENGTH, MAX_FUNCTIONS, Id2BoW, Id2Family, largest_ids, largest_families
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from file_model import build_bow, build_2steps, build_half_hierarchical_model
import datetime
from data_generator import load_smart_pkg
from sklearn.metrics import confusion_matrix

batch_size = 32
epochs = 1000

# Write results to a file
f = open('Families_code.csv', 'a')
writer = csv.writer(f, lineterminator='\n')

import matplotlib.pyplot as plt
import itertools as it
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
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
                     fontsize=12,
                     color="black" if cm[i, j] > thresh else "black")

    # fig = plt.gcf()
    # fig.set_size_inches(25, 18, forward=True)
    plt.tight_layout()
    plt.ylabel('True Label', fontsize='x-large')
    plt.xlabel('Predicted Label', fontsize='x-large')
    #plt.show()
    fig = "fig55.png"
    plt.savefig(fig, bbox_inches='tight', dpi=100)


def write_families(total_cm, model_desc):
    cm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
    print (cm)
    thres = 0.1

    for i in range(cm.shape[0]):
        row_list = []
        for j in range(cm.shape[1]):
            if cm[i, j] > thres:
                row_list.append((str(round(cm[i, j], 2)), Id2Family[j]))
        row_list.sort(reverse=True)
        row_measures = [model_desc, Id2Family[i], str(round(cm[i, i], 2)), row_list]
        writer.writerow(row_measures)

def main():
    # callbacks
    early_stopCB = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    # load data
    X = np.zeros((len(Id2File.keys()), MAX_FUNCTIONS, MAX_SEQUENCE_LENGTH), dtype='float32')  # int32
    y = np.zeros((len(Id2File.keys())), dtype='float32')  # int
    X_BoW = np.zeros((len(Id2File.keys()), MAX_FEATURES), dtype='float32')  # int

    for i, (k, v) in enumerate(Id2File.items()):
        file_path = v[0]
        # X[int(i),] = load_cutted_pkg(file_path)
        X[int(i),], list1 = load_smart_pkg(file_path)
        X_BoW[int(i),] = np.array(Id2BoW[k])
        y[int(i)] = Family2Id[v[2]]
    print("X Shape: ", X.shape)
    print("X_BoW Shape: ", X_BoW.shape)

    print("Func length: %.2f (+/- %.2f)" % (np.mean(list1), np.std(list1)))

    n_folds = 5
    train_test = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_i = 0
    e_acc_scores = []
    e_f1_scores = []
    e_loss_scores = []
    b_acc_scores = []
    b_f1_scores = []
    b_loss_scores = []
    c_acc_scores = []
    c_loss_scores = []
    c_f1_scores = []
    e_total_cm = 0.0
    b_total_cm = 0.0
    c_total_cm = 0.0

    for training_index, testing_index in train_test.split(X, y):
        fold_i += 1
        print(datetime.datetime.now())
        print("Running Fold", fold_i, "/", n_folds)

        print("train_index length: ", len(training_index))
        print("test_index length: ", len(testing_index))
        X_training, X_test = X[training_index], X[testing_index]
        y_training, y_test = y[training_index], y[testing_index]
        X_training_BoW, X_test_BoW = X_BoW[training_index], X_BoW[testing_index]

        # Split Train to Train/Validation
        train_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        train_index, val_index = next(train_val.split(X_training, y_training))
        X_train, X_val = X_training[train_index], X_training[val_index]
        y_train, y_val = y_training[train_index], y_training[val_index]
        X_train_BoW, X_val_BoW = X_training_BoW[train_index], X_training_BoW[val_index]
        y_tr_cat = to_categorical(y_train, num_classes=len(Family2Id))
        y_v_cat = to_categorical(y_val, num_classes=len(Family2Id))
        y_te_cat = to_categorical(y_test, num_classes=len(Family2Id))

        # Train model on dataset
        print("Build Step1_models")
        model_embedding = build_half_hierarchical_model()
        model_bow = build_bow()

        print("fit1...")
        mcp_save1 = ModelCheckpoint('.m1_e.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        hist = model_embedding.fit(x=X_train,
                                   y=to_categorical(y_train, num_classes=len(Family2Id)),
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=0,
                                   callbacks=[early_stopCB, mcp_save1],
                                   validation_data=[X_val, to_categorical(y_val, num_classes=len(Family2Id))],
                                   shuffle=True)

        # load best model
        model_embedding.load_weights(filepath='.m1_e.hdf5')

        print("fit2...")
        mcp_save2 = ModelCheckpoint('.m2_b.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        #hist = model_bow.fit(x=X_train_BoW,
        #                     y=to_categorical(y_train, num_classes=len(Family2Id)),
        #                     batch_size=batch_size,
        #                     epochs=epochs,
        #                     verbose=0,
        #                     callbacks=[early_stopCB, mcp_save2],
        #                     validation_data=[X_val_BoW, to_categorical(y_val, num_classes=len(Family2Id))],
        #                     shuffle=True)

        # load best model
        model_bow.load_weights(filepath='.m2_b.hdf5')

        print("Build X for Second Step...")
        print("Predict Train:")
        X_e_tr = model_embedding.predict(X_train, batch_size=batch_size)
        X_b_tr = model_bow.predict(X_train_BoW, batch_size=batch_size)

        print("Predict Val:")
        X_e_v = model_embedding.predict(X_val, batch_size=batch_size)
        X_b_v = model_bow.predict(X_val_BoW, batch_size=batch_size)

        X_c_tr = np.concatenate((X_e_tr, X_b_tr), axis=-1)
        X_c_v = np.concatenate((X_e_v, X_b_v), axis=-1)

        print("X_C_Tr Shape: ", X_c_tr.shape)
        print("X_C_Val Shape: ", X_c_v.shape)

        print("Build Step2_models")
        model_combined = build_2steps()

        print("fit Step2...")
        mcp_save3 = ModelCheckpoint('.m3_c.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        #hist = model_combined.fit(x=X_c_tr,
        #                         y=to_categorical(y_train, num_classes=len(Family2Id)),
        #                         batch_size=batch_size,
        #                         epochs=epochs,
        #                         verbose=0,
        #                         callbacks=[early_stopCB, mcp_save3],
        #                         validation_data=[X_c_v, y_v_cat],
        #                         shuffle=True)


        #print("Build Step2_models") # TEMP
        #model_embedding = build_half_hierarchical_model()
        #model_bow = build_bow()
        #model_combined = build_2steps()
        # load best model
        model_bow.load_weights(filepath='.m2_b.hdf5')
        # load best model
        model_embedding.load_weights(filepath='.m1_e.hdf5')

        # load best model
        model_combined.load_weights(filepath='.m3_c.hdf5')

        print ("Evaluate Embedding:")
        scores = model_embedding.evaluate(x=X_test, y=to_categorical(y_test, num_classes=len(Family2Id)), batch_size=batch_size,verbose=0)
        e_loss_scores.append(scores[0])
        e_acc_scores.append(scores[1] * 100)
        e_f1_scores.append(scores[2] * 100)
        print("%s: %.2f%%" % (model_embedding.metrics_names[2], scores[2] * 100))
        print("%s: %.2f%%" % (model_embedding.metrics_names[1], scores[1] * 100))
        print("%s: %.3f" % (model_embedding.metrics_names[0], scores[0]))

        predicted = model_embedding.predict(X_test, batch_size=batch_size)
        e_total_cm += confusion_matrix(y_test, np.argmax(predicted, axis=1), labels=range(len(Family2Id.keys())))

        print("Evaluate BoW:")
        scores = model_bow.evaluate(x=X_test_BoW, y=to_categorical(y_test, num_classes=len(Family2Id)),batch_size=batch_size, verbose=0)
        b_loss_scores.append(scores[0])
        b_acc_scores.append(scores[1] * 100)
        b_f1_scores.append(scores[2] * 100)
        print("%s: %.2f%%" % (model_bow.metrics_names[2], scores[2] * 100))
        print("%s: %.2f%%" % (model_bow.metrics_names[1], scores[1] * 100))
        print("%s: %.3f" % (model_bow.metrics_names[0], scores[0]))

        predicted = model_bow.predict(X_test_BoW, batch_size=batch_size)
        b_total_cm += confusion_matrix(y_test, np.argmax(predicted, axis=1), labels=range(len(Family2Id.keys())))

        print("Evaluate Combined:")
        X_e_te = model_embedding.predict(X_test, batch_size=batch_size)
        X_b_te = model_bow.predict(X_test_BoW, batch_size=batch_size)

        X_c_te = np.concatenate((X_e_te, X_b_te), axis=-1)

        scores = model_combined.evaluate(x=X_c_te, y=to_categorical(y_test, num_classes=len(Family2Id)),
                                          batch_size=batch_size, verbose=0)

        c_loss_scores.append(scores[0])
        c_acc_scores.append(scores[1] * 100)
        c_f1_scores.append(scores[2] * 100)
        print("%s: %.2f%%" % (model_combined.metrics_names[2], scores[2] * 100))
        print("%s: %.2f%%" % (model_combined.metrics_names[1], scores[1] * 100))
        print("%s: %.3f" % (model_combined.metrics_names[0], scores[0]))

        predicted = model_combined.predict(X_c_te, batch_size=batch_size)
        c_total_cm += confusion_matrix(y_test, np.argmax(predicted, axis=1), labels=range(len(Family2Id.keys())))


    print("Embedding")
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(e_acc_scores), np.std(e_acc_scores)))
    print("F1: %.2f%% (+/- %.2f%%)" % (np.mean(e_f1_scores), np.std(e_f1_scores)))
    print("Loss: %.3f (+/- %.3f)" % (np.mean(e_loss_scores), np.std(e_loss_scores)))
    print("BoW")
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(b_acc_scores), np.std(b_acc_scores)))
    print("F1: %.2f%% (+/- %.2f%%)" % (np.mean(b_f1_scores), np.std(b_f1_scores)))
    print("Loss: %.3f (+/- %.3f)" % (np.mean(b_loss_scores), np.std(b_loss_scores)))
    print("Combined")
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(c_acc_scores), np.std(c_acc_scores)))
    print("F1: %.2f%% (+/- %.2f%%)" % (np.mean(c_f1_scores), np.std(c_f1_scores)))
    print("Loss: %.3f (+/- %.3f)" % (np.mean(c_loss_scores), np.std(c_loss_scores)))

    cm = e_total_cm.astype('float') / e_total_cm.sum(axis=1)[:, np.newaxis]
    reduced_normalized_cm = cm[largest_ids, :][:, largest_ids]
    #np.save('confusionM', reduced_normalized_cm)
    print(largest_families)
    # Plot normalized confusion matrix
    plot_confusion_matrix(reduced_normalized_cm, classes=largest_families, normalize=True,title='')

    #write_families(e_total_cm, "Hieraricy")
    #write_families(b_total_cm, "BoW")
    #write_families(c_total_cm, "Meta")
    f.close()


if __name__ == '__main__':
    main()
