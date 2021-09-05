import numpy as np
from constants import MAX_FEATURES, MAX_SEQUENCE_LENGTH, MAX_FUNCTIONS, MAX_FUNCTIONS_PER_FILE, MAX_FILES, Token2Id, BoFunc, EoFile, SPLITS, Id2Token

def load_cutted_pkg(file_path):
    'load apk file contains multiple files, each file multiple functions and leftovers'
    with open(file_path, 'r') as f:
        seq = [int(i) for i in (f.read().strip().split(' ')) if i != '']
    # neto seq for fixed size w/o prior knowledge
    neto_seq = [x for x in seq if x != Token2Id[BoFunc] and x != Token2Id[EoFile]]
    #print ("File path %s, length: %d" % (file_path, len(neto_seq)))
    return prepare_sequences(neto_seq, MAX_SEQUENCE_LENGTH)#, np.array(binary_representation(neto_seq))

def reshape_sequences(X,y):
    new_X = X.reshape(-1, X.shape[-1])
    #new_X = X.reshape(X.shape[0], -1)
    new_y = [[i] * MAX_FUNCTIONS for i in y]
    flat_list = [item for sublist in new_y for item in sublist]
    return new_X, np.array(flat_list)

def X_to_one_hot(X):
    one_hot = np.zeros((X.shape[0], X.shape[1], X.shape[2], MAX_FEATURES + 1), dtype='float32')
    one_hot[:, :, np.arange(MAX_SEQUENCE_LENGTH), X] = 1
    return one_hot


def reshape_file_sequences(X,y):
    #new_X = X.reshape(-1, X.shape[-2], X.shape[-1])
    new_X = X.reshape(int(X.shape[0]*SPLITS), int(MAX_FUNCTIONS / SPLITS), X.shape[-1])
    new_y = [[i] * SPLITS for i in y]
    flat_list = [item for sublist in new_y for item in sublist]
    return new_X, np.array(flat_list)

def load_bow_pkg(file_path):
    'load apk file contains multiple files, each file multiple functions and leftovers'
    with open(file_path, 'r') as f:
        seq = [int(i) for i in (f.read().strip().split(' ')) if i != '']
    # neto seq for fixed size w/o prior knowledge
    neto_seq = [x for x in seq if x != Token2Id[BoFunc] and x != Token2Id[EoFile]]
    #leftovers = binary_representation(neto_seq)# BoW based all sequence(w/o -1/-2), need to examine only the 'sequence - funcs'
    leftovers = TF_representation(neto_seq)

    return np.array(leftovers)

def prepare_sequences(sequence, window_length):
    windows = [[0 for j in range(MAX_SEQUENCE_LENGTH)] for i in range(MAX_FUNCTIONS)]
    window_length = min(window_length, len(sequence))
    window_length = max(1, window_length)
    i = 0
    for window_start in range(0, len(sequence) - window_length + 1, window_length):
        window_end = window_start + window_length
        window = sequence[window_start:window_end] + [0 for t in range(MAX_SEQUENCE_LENGTH - len(sequence))]
        windows[i] = window
        i += 1
        if i >= MAX_FUNCTIONS:
            return np.array(windows[:MAX_FUNCTIONS])
    #   print "Really max_functions", i
    return np.array(windows[:MAX_FUNCTIONS])


def load_smart_pkg(file_path):
    'load apk file contains multiple files, each file multiple functions and leftovers'
    with open(file_path, 'r') as f:
        seq = [int(i) for i in (f.read().strip().split(' ')) if i != '']
        #seq = [int(i) for i in seq if i in Id2Token.keys()]
    #return prepare_functions_sequences(seq)
    return prepare_files_sequences(seq)
count1500=0
count2000=0
list = []
def prepare_functions_sequences(sequence):

    func = []
    in_function = False
    min_func_size = 20#20#20
    funcs = [[0 for j in range(MAX_SEQUENCE_LENGTH)] for i in range(MAX_FUNCTIONS)]
    leftovers = [0 for i in range(MAX_FEATURES)]
    funcs_count = 0
    leftovers_count = 0
    funcs_count_val = []

    for num in sequence:
        if num == Token2Id[BoFunc] or num == Token2Id[EoFile]:  # Begin of function / End Of file
            if in_function:
                funcs_count_val.append(len(func))
            if len(func) >= min_func_size and funcs_count < MAX_FUNCTIONS:  # long function
                chunks = [func[x:x + MAX_SEQUENCE_LENGTH] for x in range(0, len(func), MAX_SEQUENCE_LENGTH)]
                list.append(len(chunks[0]))
                chunks = [(chunk + [0] * (MAX_SEQUENCE_LENGTH - len(chunk))) for chunk in chunks] # padding with zeros

                #funcs[funcs_count:funcs_count+len(chunks)] = chunks
                #funcs_count += len(chunks)

                # load only first sequence from long function
                funcs[funcs_count:funcs_count + 1] = [chunks[0]]
                funcs_count += 1


            else: # small function => add to leftovers
                #leftovers[leftovers_count:] = func
                leftovers_count += len(func)
            func = []  # Clear function

            in_function = num == Token2Id[BoFunc]  # in_function = True if begin of function. in_function = False if end of fils
        else:  # Regular number
            if in_function == True:
                func.append(num)#PATCH-49

    #print ("funcs_count: ", funcs_count)
#    if funcs_count > 1500:
#        global count1500
#        count1500 +=1
#        print ('1500', count1500)
#    if funcs_count > 2000:
#        global count2000
#        count2000 +=1
#        print ('2000', count2000)
    #print "funcs_count median: ", np.median(funcs_count_val, axis=0), np.percentile(funcs_count_val, 50), np.percentile(funcs_count_val, 90)
    return np.array(funcs[:MAX_FUNCTIONS])

def prepare_files_sequences(sequence):

    func = []
    in_function = False
    min_func_size = 20
    funcs = [[0 for j in range(MAX_SEQUENCE_LENGTH)] for i in range(MAX_FUNCTIONS_PER_FILE)]
    files = [[[0 for j in range(MAX_SEQUENCE_LENGTH)] for i in range(MAX_FUNCTIONS_PER_FILE)] for f in range(MAX_FILES)]
    funcs_count = 0
    files_count=0

    for num in sequence:
        if num == Token2Id[BoFunc] or num == Token2Id[EoFile]:  # Begin of function / End Of file
            if len(func) >= min_func_size and funcs_count < MAX_FUNCTIONS_PER_FILE:  # long function
                chunks = [func[x:x + MAX_SEQUENCE_LENGTH] for x in range(0, len(func), MAX_SEQUENCE_LENGTH)]
                chunks = [(chunk + [0] * (MAX_SEQUENCE_LENGTH - len(chunk))) for chunk in chunks] # padding with zeros

                # load only first sequence from long function
                funcs[funcs_count:funcs_count + 1] = [chunks[0]]
                funcs_count += 1

                #funcs[funcs_count:funcs_count+len(chunks)] = chunks
                #funcs_count += len(chunks)

            func = []  # Clear function
            if num == Token2Id[EoFile] and files_count < MAX_FILES: # End Of file
                if funcs_count > 1: # have meat
                    #print ("Function count: ", funcs_count)
                    funcs_padd = funcs[:funcs_count] + [[0] * MAX_SEQUENCE_LENGTH for i in range(MAX_FUNCTIONS_PER_FILE-funcs_count)] # padding with zeros
                    files[files_count] = funcs_padd
                    funcs_count=0
                    files_count+=1
                else:
                    funcs_count = 0 #reset funcs

            in_function = num == Token2Id[BoFunc]  # in_function = True if begin of function. in_function = False if end of fils
        else:  # Regular number
            if in_function == True:
                func.append(num)

    #print ("files_count: ", files_count)
    return np.array(files[:MAX_FILES])


def binary_representation(list):
    new_list = [0 for i in range(MAX_FEATURES)]
    for i in list:
        new_list[i-1:i] = [1]

    return new_list

def TF_representation(list):
    new_list = [0 for i in range(MAX_FEATURES)]
    for i in list:
        new_list[i-1] += 1
    max_val = max(new_list)
    new_list = [i/float(max_val) for i in new_list]

    return new_list
#f = r"D:\APK\Authorship\Sequences\Decompiled_src\6677g_com\Decompiled_Decompiled_com.degoo.android.beach-5.apk.jar\seq.csv"
#arr = load_smart_pkg(f)
#print (arr)
