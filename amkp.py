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
    return age_groups[int(age/(group_size))]

def get_picname(x,y,hue,plot_type):
    if hue == None:
        return x +'_vs_'+ y +'_'+ plot_type + '.png'
    else:
        return x +'_vs_'+ y +'_controling_'+ hue +'_'+ plot_type +'.png'

def plt_save_plot(plot_type, data, x, y, hue, order, folder):
    plt.figure()
    if plot_type == 'boxplot':
        sns.boxplot(x= x, y = y, hue = hue, data = data, order = order)
    elif plot_type == 'swarmplot':
        sns.swarmplot(x = x, y = y, hue = hue, data = data, order = order)
    fname = os.path.join(folder, get_picname(x,y,hue,plot_type))
    plt.savefig(fname, dpi=300)

def get_all_plots(data,folder): # BP Diastolic, BP Systolic
    plt_save_plot('boxplot', data, 'age_group','BMI', None,
                    age_groups, folder)
    plt_save_plot('boxplot', data, 'gender','BMI', None,
                    None, folder)
    plt_save_plot('swarmplot', data, 'age_group','BMI', 'gender',
                    age_groups, folder)
    plt_save_plot('boxplot', data, 'age_group','BMI', 'gender',
                    age_groups, folder)

def get_all_plots_for_bp(data,folder,diastolic, systolic):
    plt_save_plot('boxplot', data, 'age_group',diastolic,
                    None, age_groups, folder)
    plt_save_plot('boxplot', data, 'age_group',systolic,
                    None, age_groups, folder)
    plt_save_plot('boxplot', data, 'age_group',diastolic,
                    'gender', age_groups, folder)
    plt_save_plot('boxplot', data, 'age_group',systolic,
                    'gender', age_groups,folder)

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

if not os.path.exists('figures_whole_group'):
    os.mkdir('figures_whole_group')

if not os.path.exists('figures_not_null'):
    os.mkdir('figures_not_null')

# build df_vital from sheet 'Vitals', df as basic dataframe
df_vital = pd.read_excel('random_data.xlsx',
                    sheet_name='Vitals', skiprows=3)
df = df_vital.drop_duplicates(subset=['Patient no.'], keep = 'last')
df = df.rename(index = df['Patient no.'])

# build df_list from sheet 'List' to get patient gender
df_list = pd.read_excel('random_data.xlsx',
                    sheet_name='List', skiprows=4)
df_list = df_list.rename(index = df_list['Patient no.'])

# build df_como for Comorbidities
df_como = pd.read_excel('random_data.xlsx',
                    sheet_name='Comorbidities', skiprows=3)
df_como = df_como.rename(index = df_como['Patient no.'])

# clean data with unit
# clean bp in df_vital
df_vital['BP Diastolic'] = df_vital['BP Diastolic'].apply(clean_bp)
df_vital['BP Systolic'] = df_vital['BP Systolic'].apply(clean_bp)

# clean bmi data
df['BMI'] = df['BMI'].apply(clean_bmi)

### add 'gender' variable into df
df['gender'] = df_list['Gender']

### add 'age' variable into df
df['DOB'] = df_list['Patient DOB']
df['age'] = (df['Reg Calendar Date'] - df['DOB']).dt.days/365

group_size = 20
age_groups = ['0-20', '20-40', '40-60', '60-80', '80-100']
df['age_group'] = df['age'].apply(get_age_group)

### add BP into df
df['BP Diastolic'] = df['BP Diastolic'].apply(clean_bp)
df['BP Systolic'] = df['BP Systolic'].apply(clean_bp)

# plot and save all boxplots and scatterplots required
get_all_plots(df, 'figures_whole_group')
get_all_plots_for_bp(df, 'figures_whole_group','BP Diastolic','BP Systolic')

## Get no of patient whose BP is in certain range
# c = 0 # c for count, i for patient no.
# for i in range(1, len(df)+1):
#     if df['BP Diastolic'][i] < 90 :
#         if df['BP Systolic'][i] < 140:
#             c +=1
# print(c)

## Check if data in vital is ascending order by date and patient no
# o1 = 0 # no of wrong order in date
# o2 = 0 # no of wrong order in patient no
# for i in range(0, len(df_vital)-1):
#     if df_vital['Patient no.'][i] == df_vital['Patient no.'][i+1]:
#         if df_vital['Reg Calendar Date'][i] > df_vital['Reg Calendar Date'][i+1]:
#             o +=1
#     if df_vital['Patient no.'][i] > df_vital['Patient no.'][i+1]:
#         o2+=1
# print(o1,o2)

### add 'BP Diastolic before' as BP before drug from BP at hypertension detect date
baseline_bp = df['Patient no.'].apply(get_baseline_bp)
df = pd.concat([df, baseline_bp], axis=1)

# define df_notnull for patient whose BP before drug is recorded
df_notnull =  df[df['baseline_systolic'].notnull()]
# plot and save all boxplots and scatterplots required
get_all_plots(df_notnull, 'figures_not_null')
get_all_plots_for_bp(df_notnull, 'figures_not_null','BP Diastolic','BP Systolic')
get_all_plots_for_bp(df_notnull,
                    'figures_not_null','baseline_diastolic','baseline_systolic')
