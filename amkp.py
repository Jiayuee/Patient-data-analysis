import os
import pandas as pd
import numpy as np
import xlsxwriter
from pandas import DataFrame

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

# Instruction
# '###' add on new variable in research
# '##' hiden function, alternative method
# '#' comment

def clean_bmi(x):
    if x == '-':
        return np.NaN
    else:
        # print(bmi)
        return float(x)

def clean_bp(bp):
    if bp == '-':
        return np.NaN
    bp_numeric = bp.split()[0] # remove unit
    return float(bp_numeric)

def get_age_group(age):
    return age_groups[(int(age/(group_size)))-1]

def get_picname(x,y,hue,plot_type):
    if hue == None:
        return x +' vs '+ y +' '+ plot_type + '.png'
    else:
        return x +' vs '+ y +' controling '+ hue +' '+ plot_type +'.png'

def plt_save_basic_plot(plot_type, data, x, y, hue, order, folder):
    plt.figure()
    if plot_type == 'boxplot':
        ax = sns.boxplot(x= x, y = y, hue = hue, data = data, order = order)
        ax.set_title(x+' vs '+y)
        if hue != None:
            ax.legend(loc='upper right', title=None)
    elif plot_type == 'swarmplot':
        sns.swarmplot(x = x, y = y, hue = hue, data = data, order = order)
    fname = os.path.join(folder, get_picname(x,y,hue,plot_type))
    plt.savefig(fname, dpi=300)

def plt_save_bp_plot(plot_type, data, x, y, hue, order, folder):
    plt.figure()
    if plot_type == 'boxplot':
        ax = sns.boxplot(x= x, y = y, hue = hue, data = data, order = order)
        ax.set_title(x + ' vs ' + y)
        ax.set(ylabel=y + ' [mmHg]')
        if hue != None:
            ax.legend(loc='upper right', title=None)
        if y == 'BP Diastolic':
            plt.plot([-1,5],[90,90],'--', linewidth = 0.5)
        elif y == 'BP Systolic':
            plt.plot([-1,6],[140,140],'--', linewidth = 0.5)
        plt.plot([-1,6],[90,90],'--', linewidth = 0.5)
    elif plot_type == 'swarmplot':
        sns.swarmplot(x = x, y = y, hue = hue, data = data, order = order)
    fname = os.path.join(folder, get_picname(x,y,hue,plot_type))
    plt.savefig(fname, dpi=300)

def get_basic_plots(data,folder): # BP Diastolic, BP Systolic
    plt_save_basic_plot('boxplot', data, 'Age Group','BMI', None,
                    age_groups, folder)
    plt_save_basic_plot('boxplot', data, 'Gender','BMI', None,
                    None, folder)
    plt_save_basic_plot('swarmplot', data, 'Age Group','BMI', 'Gender',
                    age_groups, folder)
    plt_save_basic_plot('boxplot', data, 'Age Group','BMI', 'Gender',
                    age_groups, folder)

def get_plots_for_bp(data,folder,diastolic, systolic):
    plt_save_bp_plot('boxplot', data, 'Age Group', diastolic,
                    None, age_groups, folder)
    plt_save_bp_plot('boxplot', data, 'Age Group',systolic,
                    None, age_groups, folder)
    plt_save_bp_plot('boxplot', data, 'Age Group',diastolic,
                    'Gender', age_groups, folder)
    plt_save_bp_plot('boxplot', data, 'Age Group',systolic,
                    'Gender', age_groups,folder)

def get_plots_for_lab(data,folder,lab):
    plt_save_basic_plot('boxplot', data, 'Age Group', lab,
                    None, age_groups, folder)
    plt_save_basic_plot('boxplot', data, 'Age Group',lab,
                    'Gender', age_groups, folder)


def get_baseline_bp(patient_number):
    ## baseline bp is the bp on the day when  hypertension was detected
    idx1 = vitals['Patient no.'] == patient_number
    try:
        detection_date = como['Hypertension Detect Date'][patient_number]
    except KeyError:
        return pd.Series({'Baseline Systolic' : np.NaN,
                        'Baseline Diastolic' : np.NaN})
    idx2 = vitals['Reg Calendar Date'] == detection_date
    idx = idx1 & idx2
    if idx.sum() == 0:
        # this means detection date was before first reg calender Date
        # or patient does not have a bp detect Date
        return pd.Series({'Baseline Systolic' : np.NaN,
                        'Baseline Diastolic' : np.NaN})
    else:
        return pd.Series({'Baseline Diastolic':vitals.loc[idx,'BP Diastolic'].iloc[0],
                        'Baseline Systolic':vitals.loc[idx,'BP Systolic'].iloc[0]})

def get_baseline_ldl(patient_number):
    ## baseline ldl is the 'LDL-C' on the day when hypertension was detected
    idx1 = ldl['Patient no.'] == patient_number
    try:
        detection_date = como['Dyslipidaemia Detect Date'][patient_number]
    except KeyError:
        return pd.Series({'Baseline LDL' : np.NaN}) #patient no does not exist
    idx2 = ldl['Lab Collection Calendar Date'] == detection_date
    idx = idx1 & idx2
    if idx.sum() == 0:
        # this means detection date was before first Lab Collection Calendar Date
        # or patient does not have a hypertension detect Date
        return pd.Series({'Baseline LDL' : np.NaN}) # detection date is not recorded
    else:
        return  pd.Series({'Baseline LDL':ldl.loc[idx,'Test Result (Numeric)'].iloc[0]})


def get_age_gender_bmi(patient_number):
    idx = sub_vitals['Patient no.'] == patient_number
    return pd.Series({'Age Group':sub_vitals.loc[idx,'Age Group'].iloc[0],
                    'Gender':sub_vitals.loc[idx,'Gender'].iloc[0],
                    'BMI':sub_vitals.loc[idx,'BMI'].iloc[0]})

def get_df_for_lab(lab_type):
    lab_idx = lab['Lab Test Desc'] == lab_type
    return lab[lab_idx].copy()

def map_age_gender_bmi(df):
    age_gender_bmi = df['Patient no.'].apply(get_age_gender_bmi)
    df['Age Group'] = age_gender_bmi['Age Group']
    df['Gender'] = age_gender_bmi['Gender']
    df['BMI'] = age_gender_bmi['BMI']

def get_como_status(patient_number):
    idx = como['Patient no.'] == patient_number
    como1 = como.loc[idx,'Dyslipidaemia Detect Date'].iloc[0]
    como2 = como.loc[idx,'Pre-Diabetes Mellitus Detect Date'].iloc[0]
    return pd.Series({'Dyslipidaemia':pd.isnull(como1),
                    'Pre-Diabetes':pd.isnull(como2)})

def map_como_status(df):
    dyslip_predia = df['Patient no.'].apply(get_como_status)
    df['Dyslipidaemia'] =  dyslip_predia['Dyslipidaemia']
    df['Pre-Diabetes'] =  dyslip_predia['Pre-Diabetes']

# def mkdir(dirname):
#     if os.pa

if not os.path.exists('figures_whole_group'):
    os.mkdir('figures_whole_group')

if not os.path.exists('figures_bp'):
    os.mkdir('figures_bp')

if not os.path.exists('figures_LDL-C'):
    os.mkdir('figures_LDL-C')

if not os.path.exists('figures_hba1c'):
    os.mkdir('figures_hba1c')

if not os.path.exists('figures_glucose'):
    os.mkdir('figures_glucose')

# build df_vital from sheet 'Vitals', df as basic dataframe
## ip for input, op for outout
excel_name = 'random_data.xlsx'
vitals = pd.read_excel(excel_name, sheet_name='Vitals', skiprows=3)
sub_vitals = vitals.drop_duplicates(subset=['Patient no.'], keep = 'last')
sub_vitals = sub_vitals.rename(index = sub_vitals['Patient no.'])

# build df_list from sheet 'List' to get patient gender
df_list = pd.read_excel(excel_name, sheet_name='List', skiprows=4)
df_list = df_list.rename(index = df_list['Patient no.'])

# build df_como for Comorbidities
como = pd.read_excel(excel_name, sheet_name='Comorbidities', skiprows=3)
como = como.rename(index = como['Patient no.'])

# build df_lab from sheet 'Lab' to get lab tests
lab = pd.read_excel(excel_name, sheet_name='Lab', skiprows = 3)

# build df_drugs for Drugs
# drugs = pd.read_excel(excel_name, sheet_name = 'Drugs',skiprows=3)

# clean data with unit, clean bp in df_vital
vitals['BP Diastolic'] = vitals['BP Diastolic'].apply(clean_bp)
vitals['BP Systolic'] = vitals['BP Systolic'].apply(clean_bp)

# clean bmi data
sub_vitals['BMI'] = sub_vitals['BMI'].apply(clean_bmi)

### add 'gender' variable into df
sub_vitals['Gender'] = df_list['Gender']
sub_vitals.loc[sub_vitals['Gender'] == 'M','Gender'] = 'Male'
sub_vitals.loc[sub_vitals['Gender'] == 'F','Gender'] = 'Female'

### add 'age' variable into df
sub_vitals['DOB'] = df_list['Patient DOB']
sub_vitals['Age'] = (sub_vitals['Reg Calendar Date'] - sub_vitals['DOB']).dt.days/365

group_size = 20
age_groups = ['20-40 years old', '40-60 years old',
                '60-80 years old', '80-100 years old']
sub_vitals['Age Group'] = sub_vitals['Age'].apply(get_age_group)

### add BP into df
sub_vitals['BP Diastolic'] = sub_vitals['BP Diastolic'].apply(clean_bp)
sub_vitals['BP Systolic'] = sub_vitals['BP Systolic'].apply(clean_bp)

## plot and save all boxplots and scatterplots required
# get_basic_plots(sub_vitals, 'figures_whole_group')
# get_plots_for_bp(sub_vitals, 'figures_whole_group','BP Diastolic','BP Systolic')

## Get no of patient whose BP is in certain range
# c = 0 # c for count, i for patient no.
# for i in range(1, len(df)+1):
#     if sub_vitals['BP Diastolic'][i] < 90 :
#         if sub_vitals['BP Systolic'][i] < 140:
#             c +=1
# print(c)

### add 'BP Diastolic before' as BP before drug from BP at hypertension detect date
baseline_bp_all = sub_vitals['Patient no.'].apply(get_baseline_bp)
sub_vitals = pd.concat([sub_vitals, baseline_bp_all], axis=1)

# define BBP_notnull for patient whose BP before drug is recorded
bbp_notnull =  sub_vitals[sub_vitals['Baseline Systolic'].notnull()]
## plot and save all boxplots and scatterplots required
# get_basic_plots(bbp_notnull, 'figures_bp')
# get_plots_for_bp(bbp_notnull, 'figures_bp','BP Diastolic','BP Systolic')
# get_plots_for_bp(bbp_notnull, 'figures_bp','Baseline Diastolic',
#                 'Baseline Systolic')
# plot grouped boxplot to compare BP before and after drug
baseline_bp = bbp_notnull.copy()
baseline_bp['BP record'] = 'Baseline BP'
baseline_bp['BP Diastolic'] = baseline_bp['Baseline Diastolic']
baseline_bp['BP Systolic'] = baseline_bp['Baseline Systolic']
latest_bp = bbp_notnull.copy()
latest_bp['BP record'] = 'Latest BP'
baseline_and_latest_bp = baseline_bp.append(latest_bp)

## Plot boxplots to compare baseline BP and present BP
# plt_save_bp_plot('boxplot', baseline_and_latest_bp, 'Age Group', 'BP Diastolic',
#                  'BP record', age_groups, 'figures_bp')
# plt_save_bp_plot('boxplot', baseline_and_latest_bp, 'Age Group', 'BP Systolic',
#                  'BP record', age_groups, 'figures_bp')

### add variable 'ldl'
ldl = get_df_for_lab('LDL-C')

# sub_ldl: without duplicates
sub_ldl = ldl.drop_duplicates(subset=['Patient no.'], keep = 'last')
sub_ldl['Baseline LDL'] = sub_ldl['Patient no.'].apply(get_baseline_ldl)

# ldl_notnull: drop null values
ldl_notnull =  sub_ldl[sub_ldl['Baseline LDL'].notnull()]
ldl_notnull = ldl_notnull.rename(columns = {'Test Result (Numeric)':'Latest LDL'})

# add basic info for ldl_notnull
map_age_gender_bmi(ldl_notnull)

# # basic plot: distribution
# get_basic_plots(ldl_notnull, 'figures_LDL-C')
# get_plots_for_lab(ldl_notnull,'figures_LDL-C','Baseline LDL')
# get_plots_for_lab(ldl_notnull,'figures_LDL-C','Latest LDL')

# plots for ldl: comparision with baseline
baseline_ldl = ldl_notnull.copy()
baseline_ldl['LDL'] = baseline_ldl['Baseline LDL']
baseline_ldl['LDL Type'] = 'Baseline'
latest_ldl = ldl_notnull.copy()
latest_ldl['LDL'] = latest_ldl['Latest LDL']
latest_ldl['LDL Type'] = 'Latest'
baseline_and_latest_ldl = baseline_ldl.append(latest_ldl)
# plt_save_basic_plot('boxplot', baseline_and_latest_ldl, 'Age Group', 'LDL',
#                  'LDL Type', age_groups, 'figures_LDL-C')

### add variable 'hba1c'
hba1c = get_df_for_lab('HbA1c')

# sub_hba1c: drop duplicates
sub_hba1c = hba1c.drop_duplicates(subset=['Patient no.'], keep = 'last')

# hba1c_notnull: drop null values
hba1c_notnull =  sub_hba1c[sub_hba1c['Test Result (Numeric)'].notnull()]
hba1c_notnull = hba1c_notnull.rename(columns = {'Test Result (Numeric)':'Latest HbA1c'})

# add basic info for hba1c_notnull
map_age_gender_bmi(hba1c_notnull)

# #basic plot: distribution
# get_basic_plots(hba1c_notnull, 'figures_hba1c')
# get_plots_for_lab(hba1c_notnull,'figures_hba1c','Latest HbA1c')

######## ONLY 6 PATIENT HAVE TAKEN HBA1C TEST!!!!!!!! #########

### add variable 'glucose'
glucose = get_df_for_lab('Glucose, Fasting, pl')
# sub_glucose: drop duplicates
sub_glucose = glucose.drop_duplicates(subset=['Patient no.'], keep = 'last')

# glucose_notnull: drop null values
glucose_notnull =  sub_glucose[sub_glucose['Test Result (Numeric)'].notnull()]
glucose_notnull = glucose_notnull.rename(columns = {'Test Result (Numeric)':'Latest Glucose'})

# add basic info for glucose_notnull
map_age_gender_bmi(glucose_notnull)

# #basic plot: distribution
# get_basic_plots(glucose_notnull, 'figures_glucose')
# get_plots_for_lab(glucose_notnull,'figures_glucose','Latest Glucose')


### Build predictive model for improvement in BP
# input: age, gender, baseline_bp, dyslipidaemia, prediabetes
# simple logistic regression

# prepare train data
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

model1 = LinearRegression(copy_X=True)
model2 = RandomForestRegressor(max_depth=3, random_state=0)
model3 = XGBRegressor()
model4 = LGBMRegressor()
models = [model1, model2, model3, model4]
model_names = []
scores2 = {}
for m in models:
    model_names.append(m.__class__.__name__)
    scores2[m.__class__.__name__] = {}
## there are two methods to cross validate. First method only gives you there
## the score, but does not give you a trained model. It is used for understanding
## the problem - which predictors and which models work better
# sklearn provides a function cross_val_score for this
# scores1 = {} # scores1 is scores by method 1
# for i in range(len(models)):
#     scores1[i] = cross_val_score(models[i], x, y,
#                                 scoring='neg_mean_squared_error', cv=5)
#
# for i in range(len(models)):
#     print('{}: {}'.format(models[i].__class__.__name__, scores1[i]))

## method 2 involves going through train and test splis one by one
## sklearn provides a class KFold to help with that
kf = KFold(n_splits=5, shuffle=True)
# scores2= pd.DataFrame(np.zeros((4,5)), columns = model_names)
# models_collection = pd.DataFrame({},columns = model_names)

for n, (train_index, test_index) in enumerate(kf.split(x)):
    ## we will make n models, one for each fold
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    for i in range(len(models)):
        model = models[i]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        this_score = mean_squared_error(y_test, y_pred)
        scores2[model_names[i]][n] = this_score
        # models_collection[model_names[i]].append(model)

scores2 = pd.DataFrame(scores2)

## now we have n models which can be used for future data,
## and average of outputs of these models can be used as the prediction

# etl.py (extract, transform, load) -? take raw data and prepare processed data
# descriptive_analysis.py -> genrates tables and charts
# predictive_models.py ->
