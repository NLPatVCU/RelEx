# Author : Samantha Mahendran for RelEx

import os
import csv
import pandas as pd
import sys
import fnmatch

def append(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if fnmatch.fnmatch(filename, '*.ann'):
            print(filename)
            if os.stat(input_folder + filename).st_size == 0:
                continue
            else:
                df = pd.read_csv(input_folder + filename, header = None, sep="\t", quoting=csv.QUOTE_NONE)
                df.columns = ['key', 'body', 'label']
                df['key'] = df.index+1
                df['key'] = 'T' + df['key'].astype(str)
                df.to_csv(output_folder  + filename, sep='\t', index=False, header=False, mode='a')