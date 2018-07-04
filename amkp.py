import os
import pandas as pd
import xlsxwriter
from pandas import DataFrame
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
# altertive method to import df
# # path = (r'D:\Users\deora0287p\Desktop\SR181549 Summary anonymized.xlsx')
# record = pd.ExcelFile(path)
# df = record.parse('Vitals_edited')
## Class_Name, ClassName
#define funtion to clean bmi data

# Instruction
# '###' add on new variable or aspect in research
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

def plt_save_plot(plot_type, data, x, y, hue, order):
    plt.figure()
    if plot_type == 'boxplot':
        sns.boxplot(x= x, y = y, hue = hue, data = data, order = order)
    elif plot_type == 'swarmplot':
        sns.swarmplot(x = x, y = y, hue = hue, data = data, order = order)
    fname = os.path.join('figures', get_picname(x,y,hue,plot_type))
    plt.savefig(fname, dpi=300)

## def plt_save_boxplot(DATA,X,Y,HUE,ORDER):
#     plt.figure()
#     sns.boxplot(x=X, y = Y, hue = HUE, data = DATA, order = ORDER)
#     fname = os.path.join('figures', picname(X,Y,HUE))
#     plt.savefig(fname, dpi=300)

if not os.path.exists('figures'):
    os.mkdir('figures')

# build df_vital from sheet 'Vitals', df as basic dataframe
df_vital = pd.read_excel('SR181549 Summary anonymized.xlsx',
                    sheet_name='Vitals', skiprows=3)
df = df_vital.drop_duplicates(subset=['Patient no.'], keep = 'last')
df = df.rename(index = df['Patient no.'])
# build df_list from sheet 'List' to get patient gender
df_list = pd.read_excel('SR181549 Summary anonymized.xlsx',
                    sheet_name='List', skiprows=4)
df_list = df_list.rename(index = df_list['Patient no.'])

# apply clean_bmi to bmi data
df['BMI'] = df['BMI'].apply(clean_bmi)

### add 'gender' variable into df
df['gender'] = df_list['Gender']

## boxplot of BMI by gender group
# plt_save_plot('boxplot', df, 'gender','BMI', None, None)
## plt and save figure without plt_save_plot function
# plt.figure()
# sns.boxplot(x='gender', y = 'BMI', data = df)
# fname = os.path.join('figures', 'gender_vs_bmi.png')
# plt.savefig(fname, dpi=300)

### add 'age' variable into df
df['DOB'] = df_list['Patient DOB']
df['age'] = (df['Reg Calendar Date'] - df['DOB']).dt.days/365

group_size = 20
age_groups = ['0-20', '20-40', '40-60', '60-80', '80-100']
df['age_group'] = df['age'].apply(get_age_group)

## boxplot of BMI by age_groups
# plt_save_plot('boxplot', df, 'age_group','BMI', None, age_groups)
## plt and save figure without plt_save_plot function
# plt.figure()
# sns.boxplot(x='age_group', y = 'BMI', data = df, order = age_groups )
# fname = os.path.join('figures', 'age_vs_bmi.png')
# plt.savefig(fname, dpi=300)

## boxplot of BMI  by age & gender
# plt_save_plot('boxplot', df, 'age_group','BMI', 'gender', age_groups)
## plt and save figure without plt_save_plot function
# plt.figure()
# sns.boxplot(x='age_group', y = 'BMI', hue = 'gender', data = df)
# plt.savefig('age_gender_bmi.png', dpi=300)

## scatter plot of BMI by age & gender
# plt_save_plot('swarmplot', df, 'age_group','BMI', 'gender', age_groups)
## plt and save figure without plt_save_plot function
# plt.figure()
# sns.swarmplot(x='age_group', y = 'BMI', hue = 'gender', data = df,
#                order = age_groups)
# fname = os.path.join('figures', 'age_gender_bmi_scatterplot.png')
# plt.savefig(fname, dpi=300)
#

### add BP into df
df['BP Diastolic'] = df['BP Diastolic'].apply(clean_bp)
df['BP Systolic'] = df['BP Systolic'].apply(clean_bp)

## plt and save boxplot of age_group vs BP controling gender
# plt_save_plot('boxplot', df, 'age_group','BP Diastolic', None, age_groups)
# plt_save_plot('boxplot', df, 'age_group','BP Systolic', None, age_groups)
# plt_save_plot('boxplot', df, 'age_group','BP Diastolic', 'gender', age_groups)
# plt_save_plot('boxplot', df, 'age_group','BP Systolic', 'gender', age_groups)
## plt.figure()
# sns.boxplot(x='age_group', y = 'BP Diastolic', data = df)
# plt.figure()
# sns.boxplot(x='age_group', y = 'BP Systolic', data = df)

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
# df_vital = pd.read_excel('SR181549 Summary anonymized.xlsx',
#                         sheet_name='Vitals', skiprows=3)
# for i in range(0, len(df_vital)-1):
#     if df_vital['Patient no.'][i] == df_vital['Patient no.'][i+1]:
#         if df_vital['Reg Calendar Date'][i] > df_vital['Reg Calendar Date'][i+1]:
#             o +=1
#     if df_vital['Patient no.'][i] > df_vital['Patient no.'][i+1]:
#         o2+=1
# print(o1,o2)

### BP at hypertension detect date (assume before drug taken)
# build df_como from sheet 'Comorbidities' to get hydertension detect date
df_como = pd.read_excel('SR181549 Summary anonymized.xlsx',
                    sheet_name='Comorbidities', skiprows=3)
df_como = df_como.rename(index = df_como['Patient no.'])

### add 'BP Diastolic before' as BP before drug from BP at hypertension detect date
bp_dia_before = []
j = 0 # hypertension detect day
for i in range(1,5): # i for patient no.
    if df_vital['Patient no.'][j] == i:
        if df_vital['Reg Calendar Date'][j] == df_como['Hypertension Detect Date'][i]:
            bp_dia_before[i] = df_vital['BP Diastolic'][j]
            print(i,j)
            break
        else: 
            j+=1
