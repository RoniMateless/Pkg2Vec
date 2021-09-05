import yaml
import csv

VOC_PATH = ".\\constants\\VocId_amd.txt"#VocId_random_1215_mul3.txt"

#VOC_PATH = ".\\constants\\VocId_random_1215_mul3.txt"#VocId_random_1215_mul3.txt"
#CLASS_IDSEQ_PATH = ".\\constants\\idSeq_1215_mul3.txt"##idSeq_amd
CLASS_IDSEQ_PATH = ".\\constants\\idSeq_amd.txt"#
#CLASS_IDFILE_PATH = ".\\constants\\idFile_amd.txt"#idFile_1215_mul3.txt"
#CLASS_IDFILE_PATH = ".\\constants\\idFile_1215_mul3.txt"#idFile_1215_mul3.txt"
PKG_API_PATH = ".\\constants\\package-list"
CLASS_API_PATH = ".\\constants\\Classes_API.txt"
FUNCTION_API_PATH = ".\\constants\\Functions_API.txt"
RESERVED_WORDS_PATH = ".\\constants\\reserved_words.txt"
ID_API_CALLS_PATH = ".\\constants\\ID_API_Calls_1215.csv"
#ID_TF_MATRIX_PATH = ".\\constants\\TF_Matrix_random_1215_mul3.csv"#4K.csv"
ID_TF_MATRIX_PATH = ".\\constants\\TF_Matrix_amd.csv"#4K.csv"
ID_DOC2VEC_PATH = ".\\constants\\Id2Doc2Vec.csv"

'''
Playbook:
Run Constants to build Id2File
change constants to run W/O building
Run authorship_classifier to build VOC_PATH
change constants to load Token2id
run text2sequences to create the seqs
change constants to load seqs instead of files
copy CLASS_IDFILE_PATH to CLASS_IDFILE_PATH, replace the strings accordingly
'''


#Id2File = yaml.safe_load(open(CLASS_IDFILE_PATH))
Id2File = yaml.safe_load(open(CLASS_IDSEQ_PATH))
#Id2File = {k:v for k, v in Id2File.items() if int(k) < 19400 and int(k) > 19000} #252< 252
similar_families = ['BabyBus', 'Gameloft', 'Greencopper', 'BLUEPIN']#, 'Tenlogix_Games', '7day', 'Play_Ink_Studio', 'Goodia_Inc', 'ANDROID_PIXELS', 'PlayScape']
Id2File = {k:v for k,v in Id2File.items() if v[2] not in similar_families}

#print ("ID 2 File:",Id2File)
print ("IDs Length:", len(Id2File))
Id2Apk = {k:v[1:] for k, v in Id2File.items()}# if int(k) < 236}
#print "ID 2 APK:",Id2Apk
Family2Id = {}
Id2Family = {}
counter=0
for k,v in Id2File.items():
    if v[2] in Family2Id:
        continue
    Family2Id[v[2]] = counter
    counter += 1
print ("Family 2 id: ", Family2Id)
print ("Families Length: ", len(Family2Id.keys()))
Id2Family = {v:k for k, v in Family2Id.items()}



nb_classes=len(Family2Id)
MAX_SEQUENCE_LENGTH = 40#40#100#30
MAX_FUNCTIONS = 3#1000#7#100#2000#2000
MAX_FEATURES = 100#280
DOC2VEC_FEATURES = 100
EMBEDDING_MAX_FEATURES = 1000
MAX_FUNCTIONS_PER_FILE = 3#4
MAX_FILES = 500#5000
SPLITS=5

Family2count = {}
for k, v in Id2File.items():
    Family2count[v[2]] = Family2count.get(v[2], 0) + 1
print ("Family 2 count: ", Family2count)
list1 = sorted(Family2count, key=Family2count.get, reverse=True)
largest_families = list1[:20]
print ("largest_families: ", largest_families)

largest_ids = []
for i in largest_families:
    largest_ids.append(int(Family2Id[i]))
#print largest_ids

from collections import defaultdict
import operator
import pandas as pd
#select the top features
MAX_FEATURES=1000
with open(ID_TF_MATRIX_PATH) as f:
    reader = csv.reader(f)
    orig_features_names = next(reader)[1:MAX_FEATURES+1]  # skip the header
    #print ("Features names", orig_features_names)
    d = defaultdict(float)
    for row in reader:
        if row[0] in Id2File.keys():
            for col, feature_name in zip(row[1:], orig_features_names):
                if feature_name[:1].isalpha():
                    d[feature_name] += float(col)

    sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    top_selected_features = [i[0] for i in sorted_d[:100]]
    print (top_selected_features)

MAX_FEATURES = len(top_selected_features)
print ("MAX_FEATURES: ", MAX_FEATURES)

#populate Id2BoW with the selected top features
with open(ID_TF_MATRIX_PATH) as f:
    df = pd.read_csv(f, usecols=top_selected_features+['IDs'], index_col=['IDs'])
    #print (df.columns)

    id_feature = {i:col for i, col in enumerate(df.columns)}
    BoFunc = "beginoffunction"
    EoFile = "endoffile"
    id_feature[-1] = BoFunc
    id_feature[-2] = EoFile
    feature_id = {v:k for k,v in id_feature.items()}
    filtered_df = df[df.index.isin(Id2File.keys())]
    print (filtered_df.shape)

    Id2BoW = filtered_df.T.to_dict('list')

obfuscated = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg']
obfuscated_features  = set(top_selected_features).intersection(obfuscated)
print (len(obfuscated_features ))
'''
Id2Token = yaml.safe_load(open(VOC_PATH))
#Token2Id = {k:v for k, v in Token2Id.items()}
print ("Id 2 Token:",Id2Token)
print ("Vocabulary Size:", len(Id2Token))
Token2Id = {v: k for k, v in Id2Token.items()}
BoFunc = "beginoffunction"
EoFile = "endoffile"
Token2Id[BoFunc] = -1
Token2Id[EoFile] = -2


with open(PKG_API_PATH, "r") as f:
    pkgs_API_list = f.read().splitlines()
with open(CLASS_API_PATH) as f:
    classes_API_list = list(set(f.read().splitlines()))
with open(FUNCTION_API_PATH) as f:
    functions_API_list = list(set(f.read().splitlines()))
combined_API_list = list(set(functions_API_list))# + classes_API_list))# + pkgs_API_list))
API_list = list(set(functions_API_list))# + classes_API_list + pkgs_API_list))
API_list = [s.lower() for s in API_list]
#print "combined_pkgs_and_classes API Calls length: ", len(combined_API_list)

text_file = open(RESERVED_WORDS_PATH, "r")
lines = text_file.readlines()
reserved_words = [l.split(',')[0] for l in lines]#.split(',')[0]

list = []
for k,v in Family2Id.items():
    list.append(k)
    list.append(k.lower())
    list.extend(k.split('_'))
    list.extend(k.lower().split('_'))
    list.extend(k.split('.'))
    list.extend(k.lower().split('.'))
    list.extend(k.split('-'))
    list.extend(k.lower().split('-'))
family_tokens = set(list)


#indicated_tokens = ['variables', 'df', 'jm', 'firstsprite', 'jn', 'jx', 'bd', 'a', 'serializer', 'arrayofbyte', 'goevent', 'groupelementindex', 'dt', 'global', 'cocosd', 'gamemanager', 'naver', 'indicators', 'comus', 'netmarble', 'mobage','tencent','greencopper', 'mominis', 'cocosd', 'gamemanager', 'naver', 'indicators', 'comus', 'flurry', 'paramparcel', 'hsp', 'xdrexception', 'codehaus', 'jackson', 'localbasicsprite', 'xffffffff', 'gms', 'parambasicsprite', 'jp', 'basicsprite', 'google', 'arrayofint', 'org', 'cgpoint', 'adobe', 'paramfloat', 'kv']

Id2BoW = {}
with open(ID_TF_MATRIX_PATH) as f:
    reader = csv.reader(f)
    orig_features_names = next(reader)[1:MAX_FEATURES+1]  # skip the header
    features_names = [f for f in orig_features_names]# if f in API_list or f in reserved_words]
    #features_names = [f for f in orig_features_names if f in API_list]
    #features_names = [f for f in orig_features_names[220:] if len(f) > 2 ]# if f not in reserved_words]
    #api_indexes = [i for i, f in enumerate(orig_features_names) if f not in API_list and f not in reserved_words]
    #included_indexes = [i for i, f in enumerate(orig_features_names[220:]) if len(f) > 2 ]
    included_indexes = [i for i, f in enumerate(orig_features_names) if f in features_names]
    print ("Features names", features_names)

    for row in reader:
        if row[0] in Id2File.keys():
            float_list = [float(f) for i,f in enumerate(row[1:MAX_FEATURES+1]) if i in included_indexes]
            Id2BoW[row[0]] = float_list

MAX_FEATURES = len(features_names)
print ("MAX_FEATURES: ", MAX_FEATURES)
'''
#when inserting only API calls skip the keywords +50
Id2Token = {i+1:f for i, f in enumerate(top_selected_features)}
print ("Vocabulary Length:", len(Id2Token))
#Id2Token = {i+1:f for i, f in enumerate(features_names)}
BoFunc = "beginoffunction"
EoFile = "endoffile"
Id2Token[-1] = BoFunc
Id2Token[-2] = EoFile
#Id2Token = {i+1:f for i, f in enumerate(features_names)}
#Id2Token = yaml.safe_load(open(VOC_PATH))
#Token2Id = {k:v for k, v in Token2Id.items()}
print ("Id 2 Token:",Id2Token)
Token2Id = {v: k for k, v in Id2Token.items()}
'''



Id2APICalls = {}
with open(ID_API_CALLS_PATH) as f:
    reader = csv.reader(f)
    next(reader)  # skip the header
    for row in reader:
        if row[0] in Id2File.keys():
            float_list = [float(i) for i in row[1:MAX_FEATURES+1]]
            Id2APICalls[row[0]] = float_list

Id2Doc2vec = {}
with open(ID_DOC2VEC_PATH) as f:
    reader = csv.reader(f)
    for row in reader:
        float_list = [float(i) for i in row[1:]]
        Id2Doc2vec[row[0]] = float_list
'''
'''
work_dir = r"D:\APK\AMD_Sources"
import os
import json
import ntpath
counter=0
d1 = {}
Id2File = {}
for src_dir, dirs, files in os.walk(work_dir):
    for file_ in files:
        if file_ == "mergedWholeUsable.java":  # "mergedSpecific.java":
            src_file = os.path.join(src_dir, file_)
            apk_name = src_dir.strip().split("\\")[-1]
            family_name = src_dir.strip().split("\\")[-3]
            Id2File[counter] = (src_file, apk_name, family_name)
            counter += 1
print (counter)
json.dump(Id2File, open(CLASS_IDFILE_PATH, 'w'))







work_dir = "D:\\APK\\Authorship\\Sources\\Decompiled_src"
import os
import json
import ntpath
counter=0
d1 = {}
#Id2Family = {}
Id2Apk = {}
for src_dir, dirs, files in os.walk(work_dir):
    #family_names = sorted(dirs)
    #print family_names
    #for fam in family_names:
    #    Id2Family[counter] = fam
    #    counter +=1
    for dir in dirs:
        dir_ext = ntpath.basename(dir).split('.')[-1]
        if dir_ext == "jar":
            print dir
            family_name = os.path.basename(src_dir)
            Id2Apk[counter] = (dir, family_name)
            counter += 1
            #src_file = os.path.join(src_dir, file_)
        #print (src_file)
    #print src_dir
    #for _dir in dirs:
    #    Id2Family[counter] = _dir
    #json.dump(Id2Family, open(CLASS_MAP_PATH,'w'))
print counter
json.dump(Id2Apk, open(CLASS_IDMAP_PATH, 'w'))
#    with open(CLASS_MAP_PATH, 'w') as file:
#        file.write(json.dumps(Id2Family))  # use `json.loads` to do the reverse
    #d1 = json.loads(open(CLASS_MAP_PATH))
#d1 = yaml.safe_load(open(CLASS_IDMAP_PATH))
    #print d1
    #exit(0)



Id2Family = {}
work_dir = "F:\\APK\\Authorship\\Sources\\Decompiled_src"
import os
import json
import ntpath
counter=0
d1 = {}

for src_dir, dirs, files in os.walk(work_dir):
    family_names = sorted(dirs)
    print family_names
    for fam in family_names:
        Id2Family[counter] = fam
        counter +=1
    json.dump(Id2Family, open(CLASS_MAP_PATH,'w'))
    d1 = yaml.safe_load(open(CLASS_MAP_PATH))
    print d1

    exit(0)

#json.dump(Id2Apk, open(CLASS_IDMAP_PATH, 'w'))
#    with open(CLASS_MAP_PATH, 'w') as file:
#        file.write(json.dumps(Id2Family))  # use `json.loads` to do the reverse
    #d1 = json.loads(open(CLASS_MAP_PATH))
#d1 = yaml.safe_load(open(CLASS_IDMAP_PATH))
'''
import re
def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            print (s)
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE)
    return re.sub(pattern, replacer, text)

