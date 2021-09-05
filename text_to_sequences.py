
from keras.preprocessing.text import text_to_word_sequence
import re
import string
#from constants import Id2File, Token2Id, BoFunc, EoFile, comment_remover
from ob_constants import Id2File, feature_id, BoFunc, EoFile
import os, os.path
import errno
import numpy as np

def text_to_sequences(file_path):
    orig_tokens=0
    reduced_tokens=0
    functions=0
    files=0


    with open(file_path, 'r', encoding="utf8") as f:
        text = f.read()
    #text = comment_remover(text)

    'mark Begin Of Function and End Of File'
    text = mark_text(text)
    'convert text to sequence'
    # tokenize the document
    word_sequence = text_to_word_sequence(text)


    seq = []
    count_seq=-1
    in_func = False
    func_len_list = []
    func_len_list.append(0)
    for w in word_sequence:
        #id = Token2Id.get(w, None)
        id = feature_id.get(w, None)
        if id == -1:
            functions +=1
            if in_func:
                func_len_list.append(count_seq)
            count_seq = -1
            in_func = True
        elif id == -2:
            files +=1
            if in_func:
                func_len_list.append(count_seq)
            count_seq = -1
            in_func = False
        else:
            orig_tokens +=1
        if id != None:
            reduced_tokens +=1
            if in_func:
                count_seq+=1
            seq.append(id)

#    family_name = file_path.strip().split("\\")[-3]
    #row = [file_path, family_name, orig_tokens, reduced_tokens, functions, files]
    #func_10 = [c for c in func_len_list if c>=10]
    #func_20 = [c for c in func_len_list if c>=20]
    #func_30 = [c for c in func_len_list if c>=30]
    #row = [file_path, max(func_len_list), min(func_len_list), np.mean(func_len_list), np.percentile(func_len_list, 50), np.percentile(func_len_list, 90), len(func_len_list), len(func_10), len(func_20), len(func_30)]
    #print (row)

    #writer.writerow(row)

    return seq


def mark_text(text):

    text = text.replace('----------------------------------------', EoFile + " ")

    text = re.sub('\d+', '', text)  # numerics

    def replacer(match):
        s = match.group(0)
        return BoFunc + " " + s
    pattern = re.compile(
        "((public|private|protected|static|final|native|synchronized|abstract|transient)+\\s)+[\\$_\\w\\<\\>\\[\\]]*\\s+[\\$_\\w]+\\([^\\)]*\\)?\\s*\\{?[^\\}]*\\}?")
    return re.sub(pattern, replacer, text)

#text_to_sequences('D:\\APK\\Authorship\\Sources\\Decompiled_src\\Big_Fish_Games\Decompiled_Decompiled_com.bigfishgames.android.bcasffree-30.apk.jar\\mergedWhole.java_useable.java')
# Taken from https://stackoverflow.com/a/600612/119527
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

def create_files_sequences():
    for k, v in Id2File.items():
        file_path = v[0]
        print (file_path)
        seq = text_to_sequences(file_path)
        str1 = ' '.join(str(e) for e in seq)
        new_path = file_path.replace('AMD_Sources','AMD_Sequences_100_features')
        new_new_path = new_path.replace('mergedWholeUsable.java','seq.csv')
#        new_new_path = new_path.replace('.java', '.csv')
        #with safe_open_w(new_new_path) as f:
        #    f.write(str1)

# Write results to a file
import csv
f = open('t2987_distibution.csv', 'a')
writer = csv.writer(f, lineterminator='\n')
#fields = ('package', 'family','orig tokens', 'reduced tokens', 'functions', 'files')
#fields = ('package', 'max_seq_functions', 'min_seq_functions', 'mean_seq_functions', 'precentile_50', 'precentile_90', 'functions count', 'func count more than 10','func count more than 20', 'func count more than 30')
#writer.writerow(fields)

create_files_sequences()

#f = r"D:\APK\Authorship\SubSources_1215_mul3\Decompiled_src\6677g_com\Decompiled_Decompiled_com.degoo.android.beach-5.apk.jar\mergedWhole.java_useable_1.java"
#text_to_sequences(f)