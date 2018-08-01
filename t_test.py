import pandas as pd
from scipy.stats import ttest_ind

ip = 'edited_data.h5'

df = pd.read_hdf(ip, 'LDLC_Comparision','r+')
cat1 = df[df['LDL Record'] == 'Baseline']
cat2 = df[df['LDL Record'] == 'Latest']

print(ttest_ind(cat1['LDL'],cat2['LDL']))
