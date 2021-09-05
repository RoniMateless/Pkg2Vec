
import os
from constants import Id2Apk, Id2APICalls, Id2File, MAX_FEATURES, Family2count, features_names, MAX_SEQUENCE_LENGTH, Family2Id, ID_TF_MATRIX_PATH,  combined_API_list, reserved_words, family_tokens,largest_families, largest_ids, VOC_PATH, comment_remover, Id2BoW
from collections import defaultdict
import csv
f2c = defaultdict(list)
for k, v in Id2File.items():
    fn = v[2]
    id = k

    f2c[fn] = [sum(i) for i in zip(Id2BoW[id], f2c.get(fn, [0] * MAX_FEATURES))]

fn2avg = {k: [c / float(Family2count[k]) for c in v] for k,v in f2c.items()}

sorted_l = ['final', 'float', 'static', 'public', 'int', 'string', 'import', 'return', 'throws', 'native', 'if', 'new', 'void', 'private', 'protected', 'null', 'byte', 'class', 'case', 'abstract', 'exception', 'this', 'boolean', 'long', 'catch', 'char', 'throw', 'try', 'package', 'short', 'continue', 'super', 'for', 'break', 'false', 'extends', 'while', 'else', 'double', 'true', 'instanceof', 'implements', 'finally', 'interface', 'do', 'enum', 'volatile', 'default', 'switch']
sorted_indexes = [features_names.index(w) for w in sorted_l]
sort_l = [0 for i in range(len(sorted_indexes))]
for i, o in enumerate(sorted_indexes):
    sort_l[o] = i

#print (f2c)
#print (fn2avg)

with open('fn2avg1.csv', 'a') as csv_file:
    writer = csv.writer(csv_file, lineterminator='\n')
    print(sorted_l)
    fields = ['family_name'] + sorted_l
    writer.writerow(fields)
    for k,v in fn2avg.items():

        row = [k] + [str(i) for _, i in sorted(zip(sort_l, v))]#[str(f) for f in v]
        #print (row)
        writer.writerow(row)