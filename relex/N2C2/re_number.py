# Author : Samantha Mahendran for RelEx

import os
import pandas as pd
import sys
def append(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        print(filename)
        if os.stat(input_folder + filename).st_size == 0:
            continue
        else:
            df = pd.read_csv(input_folder + filename, header = None, sep="\t")
            df.columns = ['key', 'body']
            df['key'] = df.index
            df['key'] = 'R' + df['key'].astype(str)
            df.to_csv(output_folder  + filename, sep='\t', index=False, header=False, mode='a')