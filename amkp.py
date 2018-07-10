import os
import pandas as pd
import xlsxwriter
from pandas import DataFrame
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

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
        if x == 'Age Group':
            ax.set_title('Age Group vs '+y)
            ax.set(xlabel='Age Group', ylabel=y)
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
        ax.set_title('Age Group vs ' + y)
        ax.set(xlabel='Age Group', ylabel=y + ' [mmHg]')
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
    plt_save_basic_plot('boxplot', data, 'gender','BMI', None,
                    None, folder)
    plt_save_basic_plot('swarmplot', data, 'Age Group','BMI', 'gender',
                    age_groups, folder)
    plt_save_basic_plot('boxplot', data, 'Age Group','BMI', 'gender',
                    age_groups, folder)

def get_plots_for_bp(data,folder,diastolic, systolic):
    plt_save_bp_plot('boxplot', data, 'Age Group', diastolic,
                    None, age_groups, folder)
    plt_save_bp_plot('boxplot', data, 'Age Group',systolic,
                    None, age_groups, folder)
    plt_save_bp_plot('boxplot', data, 'Age Group',diastolic,
                    'gender', age_groups, folder)
    plt_save_bp_plot('boxplot', data, 'Age Group',systolic,
                    'gender', age_groups,folder)

def get_plots_for_lab(data,folder,lab):
    plt_save_basic_plot('boxplot', data, 'Age Group', lab,
                    None, age_groups, folder)
    plt_save_basic_plot('boxplot', data, 'Age Group',lab,
                    'gender', age_groups, folder)


def get_baseline_bp(patient_number):
    ## baseline bp is the bp on the day when  hypertension was detected
    idx1 = df_vital['Patient no.'] == patient_number
    try:
        detection_date = df_como['Hypertension Detect Date'][patient_number]
    except KeyError:
        return pd.Series({'baseline_systolic' : np.NaN,
                        'baseline_diastolic' : np.NaN})
    idx2 = df_vital['Reg Calendar Date'] == detection_date
    idx = idx1 & idx2
    if idx.sum() == 0:
        # this means detection date was before first reg calender Date
        # or patient does not have a bp detect Date
        return pd.Series({'baseline_systolic' : np.NaN,
                        'baseline_diastolic' : np.NaN})
    else:
        return pd.Series({'baseline_diastolic':df_vital.loc[idx,'BP Diastolic'].iloc[0],
                        'baseline_systolic':df_vital.loc[idx,'BP Systolic'].iloc[0]})

def get_baseline_ldl(patient_number):
    ## baseline ldl is the 'LDL-C' on the day when hypertension was detected
    idx1 = ldl['Patient no.'] == patient_number
    try:
        detection_date = df_como['Dyslipidaemia Detect Date'][patient_number]
    except KeyError:
        return pd.Series({'baseline_ldl' : np.NaN}) #patient no does not exist
    idx2 = ldl['Lab Collection Calendar Date'] == detection_date
    idx = idx1 & idx2
    if idx.sum() == 0:
        # this means detection date was before first Lab Collection Calendar Date
        # or patient does not have a hypertension detect Date
        return pd.Series({'baseline_ldl' : np.NaN}) # detection date is not recorded
    else:
        return  pd.Series({'baseline_ldl':ldl.loc[idx,'Test Result (Numeric)'].iloc[0]})
        # return ldl.loc[idx,'Test Result (Numeric)'].iloc[0]

def get_age_gender_bmi(patient_number):
    idx = df['Patient no.'] == patient_number
    return pd.Series({'Age Group':df.loc[idx,'Age Group'].iloc[0],
                    'gender':df.loc[idx,'gender'].iloc[0],
                    'BMI':df.loc[idx,'BMI'].iloc[0]})

def get_df_for_lab(lab):
    lab_idx = df_lab['Lab Test Desc'] == lab
    return df_lab[lab_idx].copy()

def map_age_gender_bmi(df):
    age_gender_bmi = df['Patient no.'].apply(get_age_gender_bmi)
    df['Age Group'] = age_gender_bmi['Age Group']
    df['gender'] = age_gender_bmi['gender']
    df['BMI'] = age_gender_bmi['BMI']

if not os.path.exists('figures_whole_group'):
    os.mkdir('figures_whole_group')

if not os.path.exists('figures_not_null'):
    os.mkdir('figures_not_null')

if not os.path.exists('figures_LDL-C'):
    os.mkdir('figures_LDL-C')

if not os.path.exists('figures_hba1c'):
    os.mkdir('figures_hba1c')

if not os.path.exists('figures_glucose'):
    os.mkdir('figures_glucose')

# build df_vital from sheet 'Vitals', df as basic dataframe
excel_name = 'SR181549 Summary anonymized.xlsx'
df_vital = pd.read_excel(excel_name, sheet_name='Vitals', skiprows=3)
df = df_vital.drop_duplicates(subset=['Patient no.'], keep = 'last')
df = df.rename(index = df['Patient no.'])

# build df_list from sheet 'List' to get patient gender
df_list = pd.read_excel(excel_name, sheet_name='List', skiprows=4)
df_list = df_list.rename(index = df_list['Patient no.'])

# build df_como for Comorbidities
df_como = pd.read_excel(excel_name, sheet_name='Comorbidities', skiprows=3)
df_como = df_como.rename(index = df_como['Patient no.'])

# build df_lab from sheet 'Lab' to get lab tests
df_lab = pd.read_excel(excel_name, sheet_name='Lab', skiprows = 3)

# build df_drugs for Drugs
# df_drugs = pd.read_excel(excel_name, sheet_name = 'Drugs',skiprows=3)

# clean data with unit, clean bp in df_vital
df_vital['BP Diastolic'] = df_vital['BP Diastolic'].apply(clean_bp)
df_vital['BP Systolic'] = df_vital['BP Systolic'].apply(clean_bp)

# clean bmi data
df['BMI'] = df['BMI'].apply(clean_bmi)

### add 'gender' variable into df
df['gender'] = df_list['Gender']
df.loc[df['gender'] == 'M','gender'] = 'Male'
df.loc[df['gender'] == 'F','gender'] = 'female'

### add 'age' variable into df
df['DOB'] = df_list['Patient DOB']
df['age'] = (df['Reg Calendar Date'] - df['DOB']).dt.days/365

group_size = 20
age_groups = ['20-40 years old', '40-60 years old',
                '60-80 years old', '80-100 years old']
df['Age Group'] = df['age'].apply(get_age_group)

### add BP into df
df['BP Diastolic'] = df['BP Diastolic'].apply(clean_bp)
df['BP Systolic'] = df['BP Systolic'].apply(clean_bp)

## plot and save all boxplots and scatterplots required
# get_basic_plots(df, 'figures_whole_group')
# get_plots_for_bp(df, 'figures_whole_group','BP Diastolic','BP Systolic')

## Get no of patient whose BP is in certain range
# c = 0 # c for count, i for patient no.
# for i in range(1, len(df)+1):
#     if df['BP Diastolic'][i] < 90 :
#         if df['BP Systolic'][i] < 140:
#             c +=1
# print(c)

### add 'BP Diastolic before' as BP before drug from BP at hypertension detect date
baseline_bp_all = df['Patient no.'].apply(get_baseline_bp)
df = pd.concat([df, baseline_bp_all], axis=1)

# define df_notnull for patient whose BP before drug is recorded
df_notnull =  df[df['baseline_systolic'].notnull()]
## plot and save all boxplots and scatterplots required
# get_basic_plots(df_notnull, 'figures_not_null')
# get_plots_for_bp(df_notnull, 'figures_not_null','BP Diastolic','BP Systolic')
# get_plots_for_bp(df_notnull, 'figures_not_null','baseline_diastolic',
#                 'baseline_systolic')
# plot grouped boxplot to compare BP before and after drug
baseline_bp = df_notnull.copy()
baseline_bp['BP record'] = 'Baseline BP'
baseline_bp['BP Diastolic'] = baseline_bp['baseline_diastolic']
baseline_bp['BP Systolic'] = baseline_bp['baseline_systolic']
latest_bp = df_notnull.copy()
latest_bp['BP record'] = 'Latest BP'
baseline_and_latest_bp = baseline_bp.append(latest_bp)

## Plot boxplots to compare baseline BP and present BP
# plt_save_bp_plot('boxplot', baseline_and_latest_bp, 'age_group', 'BP Diastolic',
#                  'BP record', age_groups, 'figures_not_null')
# plt_save_bp_plot('boxplot', baseline_and_latest_bp, 'age_group', 'BP Systolic',
#                  'BP record', age_groups, 'figures_not_null')

### add variable 'ldl'
ldl = get_df_for_lab('LDL-C')

# sub_ldl: without duplicates
sub_ldl = ldl.drop_duplicates(subset=['Patient no.'], keep = 'last')
sub_ldl['baseline_ldl'] = sub_ldl['Patient no.'].apply(get_baseline_ldl)

# ldl_notnull: drop null values
ldl_notnull =  sub_ldl[sub_ldl['baseline_ldl'].notnull()]
ldl_notnull = ldl_notnull.rename(columns = {'Test Result (Numeric)':'latest_ldl'})

# add basic info for ldl_notnull
map_age_gender_bmi(ldl_notnull)

# basic plot: distribution
# get_basic_plots(ldl_notnull, 'figures_LDL-C')
# get_plots_for_lab(ldl_notnull,'figures_LDL-C','baseline_ldl')
# get_plots_for_lab(ldl_notnull,'figures_LDL-C','latest_ldl')

# plots for ldl: comparision with baseline
baseline_ldl = ldl_notnull.copy()
baseline_ldl['LDL'] = baseline_ldl['baseline_ldl']
baseline_ldl['LDL record'] = 'Baseline'
latest_ldl = ldl_notnull.copy()
latest_ldl['LDL'] = latest_ldl['latest_ldl']
latest_ldl['LDL record'] = 'Latest'
baseline_and_latest_ldl = baseline_ldl.append(latest_ldl)
# plt_save_basic_plot('boxplot', baseline_and_latest_ldl, 'age_group', 'LDL',
                 # 'LDL record', age_groups, 'figures_LDL-C')

### add variable 'hba1c'
hba1c = get_df_for_lab('HbA1c')

# sub_hba1c: drop duplicates
sub_hba1c = hba1c.drop_duplicates(subset=['Patient no.'], keep = 'last')

# hba1c_notnull: drop null values
hba1c_notnull =  sub_hba1c[sub_hba1c['Test Result (Numeric)'].notnull()]
hba1c_notnull = hba1c_notnull.rename(columns = {'Test Result (Numeric)':'Latest hba1c'})

# add basic info for hba1c_notnull
map_age_gender_bmi(hba1c_notnull)

# basic plot: distribution
get_basic_plots(hba1c_notnull, 'figures_hba1c')
get_plots_for_lab(hba1c_notnull,'figures_hba1c','Latest hba1c')

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

# basic plot: distribution
get_basic_plots(glucose_notnull, 'figures_glucose')
get_plots_for_lab(glucose_notnull,'figures_glucose','Latest Glucose')
