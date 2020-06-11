# Author : Samantha Mahendran for RelEx

import os
import pandas as pd
import sys

input_folder = sys.argv[1]
output_folder = sys.argv[2]

for filename in os.listdir(input_folder):
    print(filename)
    if os.stat(input_folder + filename).st_size == 0:
        continue
    else:
        df = pd.read_csv(input_folder + filename, sep="\t")
        df.columns = ['key', 'body']
        df['key'] = df.index+1
        df['key'] = 'R' + df['key'].astype(str)
        df.to_csv(output_folder  + filename, sep='\t', index=False, header=False, mode='a')