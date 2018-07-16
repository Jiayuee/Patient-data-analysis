from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np


# prepare train data
ip = 'random_data.xlsx'
vitals = pd.read_excel(ip, sheet_name='Vitals', skiprows=3)
sub_vitals = vitals.drop_duplicates(subset=['Patient no.'], keep = 'last')
sub_vitals = sub_vitals.rename(index = sub_vitals['Patient no.'])

df_list = pd.read_excel(ip, sheet_name='List', skiprows=4)
df_list = df_list.rename(index = df_list['Patient no.'])

como = pd.read_excel(ip, sheet_name='Comorbidities', skiprows=3)
como = como.rename(index = como['Patient no.'])

vitals['BP Diastolic'] = vitals['BP Diastolic'].apply(clean_bp)
vitals['BP Systolic'] = vitals['BP Systolic'].apply(clean_bp)

# clean bmi data
sub_vitals['BMI'] = sub_vitals['BMI'].apply(clean_bmi)

### add 'gender' variable into sub_vitals
sub_vitals['Gender'] = df_list['Gender']
sub_vitals.loc[sub_vitals['Gender'] == 'M','Gender'] = 'Male'
sub_vitals.loc[sub_vitals['Gender'] == 'F','Gender'] = 'Female'

### add 'age' variable into sub_vitals
sub_vitals['DOB'] = df_list['Patient DOB']
sub_vitals['Age'] = (sub_vitals['Reg Calendar Date'] - sub_vitals['DOB']).dt.days/365

group_size = 20
age_groups = ['20-40 years old', '40-60 years old',
                '60-80 years old', '80-100 years old']
sub_vitals['Age Group'] = sub_vitals['Age'].apply(get_age_group)

sub_vitals['BP Diastolic'] = sub_vitals['BP Diastolic'].apply(clean_bp)
sub_vitals['BP Systolic'] = sub_vitals['BP Systolic'].apply(clean_bp)

baseline_bp_all = sub_vitals['Patient no.'].apply(get_baseline_bp)
sub_vitals = pd.concat([sub_vitals, baseline_bp_all], axis=1)
bbp_notnull =  sub_vitals[sub_vitals['Baseline Systolic'].notnull()]
bp_notnull = bbp_notnull[bbp_notnull['BP Diastolic'].notnull()]
train_data = bp_notnull.copy()
map_como_status(train_data)
train_data['Diastolic Improvement'] = train_data['BP Diastolic'] - train_data['Baseline Diastolic']
train_data['Systolic Improvement'] = train_data['BP Systolic'] - train_data['Baseline Systolic']


cols = ['Age','Gender','Baseline Diastolic','Baseline Systolic',
        'Dyslipidaemia','Pre-Diabetes']
x = train_data[cols]
x['Dyslipidaemia'] = x['Dyslipidaemia'].apply(lambda i: int(i))
x['Pre-Diabetes'] = x['Pre-Diabetes'].apply(lambda i: int(i))
y = train_data['Diastolic Improvement']

dummy_fields = ['Gender']
for each in dummy_fields:
    dummies = pd.get_dummies(x.loc[:, each], prefix=each )
    x = pd.concat( [x, dummies], axis = 1 )

fields_to_drop = ['Gender']
x = x.drop(fields_to_drop, axis = 1)

Si = sobol.analyze(x, y)
