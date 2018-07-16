from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np


# prepare df
ip = 'edited_data.xlsx'
df = pd.read_excel(ip, sheet_name = "BP Improvement")

cols = ['Age','Gender_Male','Gender_Female','Baseline Diastolic','Baseline Systolic',
        'Dyslipidaemia','Pre-Diabetes']
x = df[cols]
y = df['Diastolic Improvement']
