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
