import pandas as pd
from scipy.stats import ttest_ind

ip = 'edited_data.h5'

df = pd.read_hdf(ip, 'LDLC_Comparision','r+')

age_groups = ['40-60 years old','60-80 years old','80-100 years old','20-40 years old']
for a in age_groups:
    idx1 = df['LDL Record'] == 'Baseline'
    idx2 = df['LDL Record'] == 'Latest'
    idx3 = df['Age Group'] == a
    cat1 = df[idx1&idx3]
    cat2 = df[idx2&idx3]
    print(a,ttest_ind(cat1['LDL'],cat2['LDL']))
