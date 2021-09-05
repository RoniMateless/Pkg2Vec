# Shared Feature Extraction Layer
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Lambda
from keras.layers.recurrent import LSTM, GRU
from keras.layers.merge import concatenate, add, maximum, average
from constants import nb_classes, MAX_FEATURES, MAX_SEQUENCE_LENGTH, MAX_FUNCTIONS, MAX_FUNCTIONS_PER_FILE, MAX_FILES, EMBEDDING_MAX_FEATURES, SPLITS
from attention_layer import AttentionWithContext
from zero_masked_layer import ZeroMaskedEntries, mask_aware_mean, mask_aware_mean_output_shape
from keras.layers import Conv1D, Reshape
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD, RMSprop
from keras.constraints import maxnorm
from keras import backend as K

embedded_dim = 100#100
encoded_dim = 100
#encoded_dim2 = 1000
nb_filter = 10#100#150
kernel_sizes = [1,2,3]#,3)#, 5)



def get_lstm_stateful_model():

    input = Input(batch_shape=(1, 1))

    embedded = Embedding(output_dim=10, input_dim=MAX_FEATURES+1, mask_zero=True)(input)
    encoded = LSTM(units=10, stateful=True)(embedded)
    output = Dense(nb_classes, activation='softmax')(encoded)

    model = Model(inputs=input, outputs=output)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model

def build_cnn_above_tf():

    #input = Input(shape=(MAX_FEATURES,1), dtype='floatX')
    input = Input(shape=(MAX_FEATURES,), dtype='floatX')
    reshape = Reshape((MAX_FEATURES, 1))(input)

    convs = []
    for ksz in kernel_sizes:
        conv = Conv1D(filters=nb_filter, kernel_size=ksz, activation='relu')(reshape)#kernel_regularizer=l2(0.03),
        pool = MaxPooling1D(pool_size=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    if len(kernel_sizes) > 1:
        pkg_encoded = concatenate(convs)
    else:
        pkg_encoded = flatten
    pkg_encoded = Dropout(0.5)(pkg_encoded)

    pkg_encoded = Dense(1000)(pkg_encoded)

    # Prediction
    prediction = Dense(nb_classes, activation='softmax')(pkg_encoded)
    #model = Model([pkg_input, bow_input, api_input], prediction)
    model = Model(input, prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy', f1])

    print (model.summary())
    # plot_model(model, to_file='shared_feature_extractor.png')

    return model



def build_1_sequence():

    input = Input(shape=(MAX_FUNCTIONS*MAX_SEQUENCE_LENGTH,), dtype='floatX')
    embedded_sequence = Embedding(output_dim=embedded_dim, input_dim=EMBEDDING_MAX_FEATURES + 1)(input)
    '''
    convs = []
    for ksz in kernel_sizes:
        conv = Conv1D(filters=nb_filter, kernel_size=ksz, activation='relu')(embedded_sequence)#kernel_regularizer=l2(0.03),
        pool = MaxPooling1D(pool_size=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    if len(kernel_sizes) > 1:
        pkg_encoded = concatenate(convs)
    else:
        pkg_encoded = flatten
    pkg_encoded = Dropout(0.5)(pkg_encoded)
    '''
    pkg_encoded = LSTM(units=encoded_dim)(embedded_sequence)
    '''
    seq_model = Model(input, pkg_encoded)

    pkg_input = Input(shape=(MAX_FUNCTIONS, MAX_SEQUENCE_LENGTH), dtype='floatX')  # float64
    pkg_encoded = TimeDistributed(seq_model)(pkg_input)
    pkg_encoded = Dropout(0.5)(pkg_encoded)
    pkg_encoded = Flatten()(pkg_encoded)

    bow_input = Input(shape=(MAX_FEATURES,), dtype='floatX')#float64
    bow_encoded = Dense(int(MAX_FEATURES/2))(bow_input)

    api_input = Input(shape=(MAX_FEATURES,), dtype='floatX')#float64
    api_encoded = Dense(int(MAX_FEATURES/2))(api_input)

    pkg_encoded = concatenate([pkg_encoded, bow_encoded, api_encoded])
    '''
    #pkg_encoded = Dense(1000)(pkg_encoded)
    # Prediction
    prediction = Dense(nb_classes, activation='softmax')(pkg_encoded)
    #model = Model([pkg_input, bow_input, api_input], prediction)
    model = Model(input, prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy', f1])

    print (model.summary())
    # plot_model(model, to_file='shared_feature_extractor.png')

    return model


def build_long_sequence():
    # Encode each timestep
    input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='floatX')

    #embedded_function = Embedding(output_dim=embedded_dim, input_dim=MAX_FEATURES+1, mask_zero=True)(input)
    embedded_function = Embedding(output_dim=embedded_dim, input_dim=MAX_FEATURES + 1)(input)

    #out_function = LSTM(units=encoded_dim)(embedded_function)
    convs = []
    for ksz in kernel_sizes:
        conv = Conv1D(filters=nb_filter, kernel_size=ksz, activation='relu')(embedded_function)#kernel_regularizer=l2(0.03),
        pool = MaxPooling1D(pool_size=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    if len(kernel_sizes) > 1:
        pkg_encoded = concatenate(convs)
    else:
        pkg_encoded = flatten
    out_function = Dropout(0.5)(pkg_encoded)


    #embed_zeroed = ZeroMaskedEntries()(embedded_function)
    #out_function = Lambda(mask_aware_mean, mask_aware_mean_output_shape)(embed_zeroed)
    # Prediction
    prediction = Dense(nb_classes, activation='softmax')(out_function)
    model = Model(input, prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', f1])

    print (model.summary())
    #plot_model(model, to_file='shared_feature_extractor.png')

    return model


def build_yang_model():
    # Encode each timestep
    in_function = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='floatX')  # float64

    embedded_function = Embedding(output_dim=embedded_dim, input_dim=MAX_FEATURES + 1, mask_zero=True)(in_function)
    out_function = Bidirectional(GRU(units=encoded_dim, return_sequences=True))(embedded_function)
    out_function = TimeDistributed(Dense(50))(out_function)
    out_function = AttentionWithContext()(out_function)

    function_model = Model(in_function, out_function)

    print(function_model.summary())

    pkg_input = Input(shape=(MAX_FUNCTIONS, MAX_SEQUENCE_LENGTH), dtype='floatX')  # float64
    pkg_encoded = TimeDistributed(function_model)(pkg_input)

    pkg_encoded = Dropout(0.2)(pkg_encoded)

    pkg_encoded = Bidirectional(GRU(units=encoded_dim, return_sequences=True))(pkg_encoded)
    pkg_encoded = TimeDistributed(Dense(50))(pkg_encoded)
    pkg_encoded = AttentionWithContext()(pkg_encoded)

    # Prediction
    main_output = Dense(nb_classes, activation='softmax', name='main_output')(pkg_encoded)
    model = Model(pkg_input, main_output)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', f1])

    print(model.summary())

    return model


def build_tang_cnn_model():
    # Encode each timestep
    in_function = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='floatX')  # float64

    embedded_function = Embedding(output_dim=embedded_dim, input_dim=MAX_FEATURES + 1)(in_function)

    convs = []
    for ksz in [1,2,3]:
        conv = Conv1D(filters=nb_filter, kernel_size=ksz, activation='relu')(embedded_function)  # kernel_regularizer=l2(0.03),
        pool = MaxPooling1D(pool_size=2)(conv)
        pool = Flatten()(pool)
        convs.append(pool)

    out_function = concatenate(convs)

    function_model = Model(in_function, out_function)

    print(function_model.summary())

    pkg_input = Input(shape=(MAX_FUNCTIONS, MAX_SEQUENCE_LENGTH), dtype='floatX')  # float64
    pkg_encoded = TimeDistributed(function_model)(pkg_input)

    pkg_encoded = Bidirectional(GRU(units=encoded_dim))(pkg_encoded)

    # Prediction
    main_output = Dense(nb_classes, activation='softmax', name='main_output')(pkg_encoded)
    model = Model(pkg_input, main_output)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', f1])

    print(model.summary())

    return model

def build_tang_lstm_model():
    # Encode each timestep
    in_function = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='floatX')  # float64

    embedded_function = Embedding(output_dim=embedded_dim, input_dim=MAX_FEATURES + 1, mask_zero=True)(in_function)

    out_function = LSTM(units=encoded_dim, return_sequences=True)(embedded_function)

    embed_zeroed = ZeroMaskedEntries()(out_function)
    out_function = Lambda(mask_aware_mean, mask_aware_mean_output_shape)(embed_zeroed)

    function_model = Model(in_function, out_function)

    print(function_model.summary())

    pkg_input = Input(shape=(MAX_FUNCTIONS, MAX_SEQUENCE_LENGTH), dtype='floatX')  # float64
    pkg_encoded = TimeDistributed(function_model)(pkg_input)

    pkg_encoded = Bidirectional(GRU(units=encoded_dim))(pkg_encoded)

    # Prediction
    main_output = Dense(nb_classes, activation='softmax', name='main_output')(pkg_encoded)
    model = Model(pkg_input, main_output)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', f1])

    print(model.summary())

    return model

def build_kim_model():
    # Encode each timestep
    in_function = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='floatX')  # float64
    embedded_function = Embedding(output_dim=embedded_dim, input_dim=MAX_FEATURES + 1)(in_function)
    convs = []
    for ksz in kernel_sizes:
        conv = Conv1D(filters=nb_filter, kernel_size=ksz, activation='relu')(embedded_function)#kernel_regularizer=l2(0.03),
        pool = MaxPooling1D(pool_size=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    out_function = concatenate(convs)

    function_model = Model(in_function, out_function)

    print(function_model.summary())

    pkg_input = Input(shape=(MAX_FUNCTIONS, MAX_SEQUENCE_LENGTH), dtype='floatX')  # float64
    pkg_encoded = TimeDistributed(function_model)(pkg_input)

    pkg_encoded = Dropout(0.8)(pkg_encoded)

    pkg_encoded = AttentionWithContext()(pkg_encoded)

    # Prediction
    main_output = Dense(nb_classes, activation='softmax', name='main_output')(pkg_encoded)
    model = Model(pkg_input, main_output)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', f1])

    print(model.summary())

    return model


def build_half_hierarchical_model():
    # Encode each timestep
    in_function = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='floatX')#float64

    embedded_function = Embedding(output_dim=embedded_dim, input_dim=MAX_FEATURES+1, mask_zero=True)(in_function)
    embedded_function = Dropout(0.1)(embedded_function)

    embed_zeroed = ZeroMaskedEntries()(embedded_function)
    out_function = Lambda(mask_aware_mean, mask_aware_mean_output_shape)(embed_zeroed)

    function_model = Model(in_function, out_function)

    print (function_model.summary())

    pkg_input = Input(shape=(MAX_FUNCTIONS, MAX_SEQUENCE_LENGTH), dtype='floatX')#float64
    pkg_encoded = TimeDistributed(function_model)(pkg_input)

    pkg_encoded = Dropout(0.8)(pkg_encoded)

    pkg_encoded = TimeDistributed(Dense(embedded_dim))(pkg_encoded)
    pkg_encoded = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(pkg_encoded)

    #pkg_encoded = AttentionWithContext()(pkg_encoded)

    #pkg_encoded = Dense(50)(pkg_encoded)


    # pkg_encoded = Lambda(mask_aware_mean, mask_aware_mean_output_shape)(pool)


    #seq_output = Dense(nb_classes, activation='softmax', name='seq_output')(pkg_encoded)

    #bow_input = Input(shape=(MAX_FEATURES,), dtype='floatX')#float64
    #bow_encoded = Dense(int(MAX_FEATURES/2))(bow_input)
    #bow_encoded = Dense(nb_classes)(bow_encoded)

    #bow_output = Dense(nb_classes, activation='softmax', name='bow_output')(bow_encoded)

    #pkg_encoded = concatenate([seq_output, bow_output])
    #pkg_encoded = average([pkg_encoded])
    #main_output = maximum([seq_output, bow_output],name='main_output')
    #pkg_encoded = Dense(100)(pkg_encoded)

    # Prediction
    #prediction = TimeDistributed(Dense(nb_classes, activation='softmax'))(pkg_encoded)
    main_output = Dense(nb_classes, activation='softmax', name='main_output')(pkg_encoded)
    #model = Model([pkg_input, bow_input], main_output)
    #model = Model(inputs=[pkg_input, bow_input], outputs=[main_output, seq_output, bow_output])
    model = Model(pkg_input, main_output)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy', f1])
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

    print (model.summary())
    #plot_model(model, to_file='shared_feature_extractor.png')

    return model



def build_hierarchical_model():
    # Encode each timestep
    in_function = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='floatX')
    embedded_function = Embedding(output_dim=embedded_dim, input_dim=MAX_FEATURES+1, mask_zero=True)(in_function)
    embedded_function = Dropout(0.1)(embedded_function)

    out_function = AttentionWithContext()(embedded_function)
    #embed_zeroed = ZeroMaskedEntries()(embedded_function)
    #out_function = Lambda(mask_aware_mean, mask_aware_mean_output_shape)(embed_zeroed)

    function_model = Model(in_function, out_function)

    print (function_model.summary())


    in_file = Input(shape=(MAX_FUNCTIONS_PER_FILE, MAX_SEQUENCE_LENGTH), dtype='floatX')
    embedded_file = TimeDistributed(function_model)(in_file)
    embedded_file = Dropout(0.1)(embedded_file)

    #embed_zeroed = ZeroMaskedEntries()(embedded_file)
    #file_encoded = Lambda(mask_aware_mean, mask_aware_mean_output_shape)(embed_zeroed)
    file_encoded = AttentionWithContext()(embedded_file)
    #file_encoded = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(embedded_file)

    file_model = Model(in_file, file_encoded)
    print(file_model.summary())


    pkg_input = Input(shape=(MAX_FILES, MAX_FUNCTIONS_PER_FILE, MAX_SEQUENCE_LENGTH), dtype='floatX')
    pkg_encoded = TimeDistributed(file_model)(pkg_input)

    pkg_encoded = Dropout(0.8)(pkg_encoded)
    pkg_encoded = AttentionWithContext()(pkg_encoded)

    # Prediction
    main_output = Dense(nb_classes, activation='softmax', name='main_output')(pkg_encoded)
    model = Model(pkg_input, main_output)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy', f1])
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

    print (model.summary())

    return model


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    pre = precision(y_true, y_pred)
    re = recall(y_true, y_pred)
    return 2*((pre*re)/(pre+re+K.epsilon()))

def build_bow():

    bow_input = Input(shape=(MAX_FEATURES,), dtype='floatX')
    bow_encoded = Dense(int(MAX_FEATURES / 2), activation='tanh')(bow_input)
    prediction = Dense(nb_classes, activation='softmax')(bow_encoded)
    model = Model(bow_input, prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', f1])

    print (model.summary())
    #plot_model(model, to_file='shared_feature_extractor.png')

    return model

def build_2steps():

    input = Input(shape=(nb_classes*2,), dtype='floatX')
    prediction = Dense(nb_classes, activation='softmax')(input)
    model = Model(input, prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', f1])
    return model

#build_hierarchical_model()
#build_bow()

