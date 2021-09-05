

import os
import ntpath
from constants import pkgs_API_list, classes_API_list, functions_API_list, combined_pkgs_and_classes_list

print "Pkgs API Calls length: ", len(pkgs_API_list)
print "Classes API Calls length: ", len(classes_API_list)
print "Functions API Calls length: ", len(functions_API_list)
print "combined_pkgs_and_classes API Calls length: ", len(combined_pkgs_and_classes_list)

import csv
file = open('authorship_api_calls_1215.csv', 'a')
fields = ('apk_name','family', 'variety', 'pkgs api calls', 'classes api calls', 'functions api calls1', 'functions api calls2')
writer = csv.writer(file, lineterminator='\n')
writer.writerow(fields)


#work_dir = "E:\\APK\\APK Benign and Limited AMD\\Data\\Source\\VirusShare_Malicious\\Malicious\\"
work_dir = "G:\\APK\\Authorship\\Sources\\Decompiled_src"
import datetime
import re
for src_dir, dirs, files in os.walk(work_dir):
    for file_ in files:
        file_name =  ntpath.basename(file_)
        #if file_name == "mergedWholeUsable.java":
        if file_name == "mergedWhole.java_useable.java":
            src_file = os.path.join(src_dir, file_)
            print src_file
            with open(src_file, 'r') as fp:
                #file_list = set(fp.readlines())
                file_str = fp.read()

                file_str = re.sub('[^0-9a-zA-Z\.]+', ' ', file_str)
                file_tokens = set(file_str.split())
            #print datetime.datetime.now()
            pkgs_data = [token for token in pkgs_API_list if token in file_tokens] #token in file_str
            classes_data = [token for token in combined_pkgs_and_classes_list if token in file_tokens]
            functions_data = [token for token in functions_API_list if token in file_tokens]
            #pkgs_data = list(filter(lambda x: x in file_str, pkgs_API_list))
            #classes_data = list(filter(lambda x: x in file_str, combined_pkgs_and_classes_list))
            #functions_data = list(filter(lambda x: x in file_str, functions_API_list))
            #print datetime.datetime.now()
            #print "Pkgs API Calls Found: ", len(pkgs_data)
            #print "Classes API Calls Found: ", len(classes_data)
            #print "Functions API Calls Found: ", len(functions_data)
            parent_path = os.path.dirname(src_file)
            apk_name = os.path.basename(parent_path)
            par_par_path = os.path.dirname(parent_path)
            variety = os.path.basename(par_par_path)
            par_par_par_path = os.path.dirname(par_par_path)
            family = os.path.basename(par_par_par_path)
            func_part1 = functions_data[:len(functions_data)/2]
            func_part2 = functions_data[len(functions_data)/2:]
            row_features = [apk_name,family, variety, pkgs_data, classes_data, func_part1, func_part2]
            #print (row_features)
            writer.writerow(row_features)
