import os
import pandas as pd
import numpy as np
import xlsxwriter
from pandas import DataFrame

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

### make direction for figures:
def mkdir(dirname):
   if not os.path.exists(dirname):
       os.mkdir(dirname)

dirnames = ['figures_whole_group','figures_bp','figures_LDL-C',
            'figures_hba1c','figures_glucose']

for d in dirnames:
    mkdir(d)

### import data:
# build df_vital from sheet 'Vitals', df as basic dataframe
## ip for input, op for outout
ip = 'random_data.xlsx'
vitals = pd.read_excel(ip, sheet_name='Vitals', skiprows=3)
sub_vitals = vitals.drop_duplicates(subset=['Patient no.'], keep = 'last')
sub_vitals = sub_vitals.rename(index = sub_vitals['Patient no.'])

# build df_list from sheet 'List' to get patient gender
df_list = pd.read_excel(ip, sheet_name='List', skiprows=4)
df_list = df_list.rename(index = df_list['Patient no.'])

# build df_como for Comorbidities
como = pd.read_excel(ip, sheet_name='Comorbidities', skiprows=3)
como = como.rename(index = como['Patient no.'])

# build df_lab from sheet 'Lab' to get lab tests
lab = pd.read_excel(ip, sheet_name='Lab', skiprows = 3)

# build df_drugs for Drugs
# drugs = pd.read_excel(ip, sheet_name = 'Drugs',skiprows=3)

### clean data:
# clean data with unit, clean bp in vitals
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

### add BP into df
sub_vitals['BP Diastolic'] = sub_vitals['BP Diastolic'].apply(clean_bp)
sub_vitals['BP Systolic'] = sub_vitals['BP Systolic'].apply(clean_bp)

### add 'baseline_bp' as BP before drug from BP at hypertension detect date
baseline_bp_all = sub_vitals['Patient no.'].apply(get_baseline_bp)
sub_vitals = pd.concat([sub_vitals, baseline_bp_all], axis=1)

### define BBP_notnull for patient whose BP before drug is recorded
bbp_notnull =  sub_vitals[sub_vitals['Baseline Systolic'].notnull()]

### build dataframe contain both baseline and latest bp for further comparision
# BP record: baseline bp or latest bp
baseline_bp = bbp_notnull.copy()
baseline_bp['BP record'] = 'Baseline BP'
baseline_bp['BP Diastolic'] = baseline_bp['Baseline Diastolic']
baseline_bp['BP Systolic'] = baseline_bp['Baseline Systolic']
latest_bp = bbp_notnull.copy()
latest_bp['BP record'] = 'Latest BP'
baseline_and_latest_bp = baseline_bp.append(latest_bp)

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

### build dataframe contain both baseline and latest bp for further comparision
# LDL record : baseline or latest
baseline_ldl = ldl_notnull.copy()
baseline_ldl['LDL'] = baseline_ldl['Baseline LDL']
baseline_ldl['LDL Record'] = 'Baseline'
latest_ldl = ldl_notnull.copy()
latest_ldl['LDL'] = latest_ldl['Latest LDL']
latest_ldl['LDL Record'] = 'Latest'
baseline_and_latest_ldl = baseline_ldl.append(latest_ldl)

### add variable 'hba1c'
hba1c = get_df_for_lab('HbA1c')

# sub_hba1c: drop duplicates
sub_hba1c = hba1c.drop_duplicates(subset=['Patient no.'], keep = 'last')

# hba1c_notnull: drop null values
hba1c_notnull =  sub_hba1c[sub_hba1c['Test Result (Numeric)'].notnull()]
hba1c_notnull = hba1c_notnull.rename(columns = {'Test Result (Numeric)':'Latest HbA1c'})

# add basic info for hba1c_notnull
map_age_gender_bmi(hba1c_notnull)

### add variable 'glucose'
glucose = get_df_for_lab('Glucose, Fasting, pl')
# sub_glucose: drop duplicates
sub_glucose = glucose.drop_duplicates(subset=['Patient no.'], keep = 'last')

# glucose_notnull: drop null values
glucose_notnull =  sub_glucose[sub_glucose['Test Result (Numeric)'].notnull()]
glucose_notnull = glucose_notnull.rename(columns = {'Test Result (Numeric)':'Latest Glucose'})

# add basic info for glucose_notnull
map_age_gender_bmi(glucose_notnull)








import seaborn as sns
import matplotlib.pyplot as plt

def get_picname(x,y,hue,plot_type):
    if hue == None:
        return x +' vs '+ y +' '+ plot_type + '.png'
    else:
        return x +' vs '+ y +' controling '+ hue +' '+ plot_type +'.png'

def save_basic_plot(plot_type, data, x, y, hue, order, folder):
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
    plt.close()

def save_bp_plot(plot_type, data, x, y, hue, order, folder):
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
    plt.close()

def get_basic_plots(data,folder): # BP Diastolic, BP Systolic
    save_basic_plot('boxplot', data, 'Age Group','BMI', None,
                    age_groups, folder)
    save_basic_plot('boxplot', data, 'Gender','BMI', None,
                    None, folder)
    save_basic_plot('swarmplot', data, 'Age Group','BMI', 'Gender',
                    age_groups, folder)
    save_basic_plot('boxplot', data, 'Age Group','BMI', 'Gender',
                    age_groups, folder)


def get_plots_for_bp(data,folder,diastolic, systolic):
    save_bp_plot('boxplot', data, 'Age Group', diastolic,
                    None, age_groups, folder)
    save_bp_plot('boxplot', data, 'Age Group',systolic,
                    None, age_groups, folder)
    save_bp_plot('boxplot', data, 'Age Group',diastolic,
                    'Gender', age_groups, folder)
    save_bp_plot('boxplot', data, 'Age Group',systolic,
                    'Gender', age_groups,folder)

def get_plots_for_lab(data,folder,lab):
    save_basic_plot('boxplot', data, 'Age Group', lab,
                    None, age_groups, folder)
    save_basic_plot('boxplot', data, 'Age Group',lab,
                    'Gender', age_groups, folder)

### Boxplots and scatterplots for sub_vitals
get_basic_plots(sub_vitals, 'figures_whole_group')
get_plots_for_bp(sub_vitals, 'figures_whole_group','BP Diastolic','BP Systolic')

### Boxplots and scatterplots for patients with baseline BP
get_basic_plots(bbp_notnull, 'figures_bp')
get_plots_for_bp(bbp_notnull, 'figures_bp','BP Diastolic','BP Systolic')
get_plots_for_bp(bbp_notnull, 'figures_bp','Baseline Diastolic',
                'Baseline Systolic')

### Comparision between baseline BP and present BP
save_bp_plot('boxplot', baseline_and_latest_bp, 'Age Group', 'BP Diastolic',
                 'BP record', age_groups, 'figures_bp')
save_bp_plot('boxplot', baseline_and_latest_bp, 'Age Group', 'BP Systolic',
                 'BP record', age_groups, 'figures_bp')

### LDL-C distribution
get_basic_plots(ldl_notnull, 'figures_LDL-C')
get_plots_for_lab(ldl_notnull,'figures_LDL-C','Baseline LDL')
get_plots_for_lab(ldl_notnull,'figures_LDL-C','Latest LDL')

### Comparision between baseline and latest LDL-C
save_basic_plot('boxplot', baseline_and_latest_ldl, 'Age Group', 'LDL',
                 'LDL Record', age_groups, 'figures_LDL-C')

### HbA1c distribution
get_basic_plots(hba1c_notnull, 'figures_hba1c')
get_plots_for_lab(hba1c_notnull,'figures_hba1c','Latest HbA1c')

### Glucose distribution
get_basic_plots(glucose_notnull, 'figures_glucose')
get_plots_for_lab(glucose_notnull,'figures_glucose','Latest Glucose')
