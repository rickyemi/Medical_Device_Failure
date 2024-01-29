

#########################################################################################################################################
import pandas as pd # data  loading, manipulation and wrangling
pd.set_option('display.max_rows', None) # display all rows in the dataset
pd.set_option('display.max_columns', None) # display all columns in the dataset
pd.set_option('display.float_format', lambda x: '%.2f' % x) # suppress all scientific notations and round to 2 decimal places
import numpy as np

#Visualization
import plotly.express as px #interactive visualization
import seaborn as sns # statistical visualization
#SNS Settings 
sns.set(color_codes = True)
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(10,10)})
sns.set_palette("Set3")
import matplotlib.pyplot as plt # basic visualization
# Command to tell Python to actually display the graphs
%matplotlib inline

# statistical Analysis library
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import shapiro
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
# To ignore unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
import statsmodels.api as sm
from patsy import dmatrices
#SNS Settings 
sns.set(color_codes = True)
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(10,10)})
sns.set_palette("Set3")

!pip install statsmodels
import statsmodels.formula.api as smf

# Import Data Set
import time
time_begin = time.time()
#########################################################################################################################################

df_main = pd.read_csv("/content/drive/MyDrive/Previous_Projects/medical_device_failure.csv")

#######################################################################################################################################

data[['Velocity','Torque(Nm)',"Wear_Measure","Sensor_1","Sensor_2"]] = data[['Velocity','Torque(Nm)',"Wear_Measure","Sensor_1","Sensor_2"]].fillna(data[['Velocity','Torque(Nm)',"Wear_Measure","Sensor_1","Sensor_2"]].median())

data.isnull().sum()

#############################################################################################################################################

#### What's the Proportion of Device Failure ?

print(data['Device_Fail'].value_counts())
print(data['Device_Fail'].value_counts(normalize = True))

sns.countplot(x="Device_Fail",edgecolor=sns.color_palette("dark", 2),facecolor=(0,0,0,0),linewidth=3,data = data)

#########################################################################################################################################

### Of all 4 gear_types which has the minimum and maximum Wear_Measure?

geartype_minmax_wear_measure = data.groupby('Gear_Type').sum()[['Wear_Measure']].sort_values('Wear_Measure',ascending = False)
geartype_minmax_wear_measure

plt.figure(figsize=(14,8))
sns.catplot(data=geartype_minmax_wear_measure,x= 'Gear_Type', y='Wear_Measure', kind='bar',edgecolor=sns.color_palette("dark", 5),facecolor=(0,0,0,0),linewidth=3);
plt.xticks(rotation=30);

#########################################################################################################################################

### What's the average velocity and Torque for all 4 gear types?

geartype_avg_velocity = data.groupby('Gear_Type').mean()[['Velocity']].sort_values('Velocity',ascending = False)
geartype_avg_velocity

plt.figure(figsize=(14,8))
sns.catplot(data=geartype_avg_velocity,x= 'Gear_Type', y='Velocity', kind='bar',edgecolor=sns.color_palette("dark", 5),facecolor=(0,0,0,0),linewidth=3);
plt.xticks(rotation=30);


geartype_avg_torque = data.groupby('Gear_Type').mean()[['Torque(Nm)']].sort_values('Torque(Nm)',ascending = False)
geartype_avg_torque

plt.figure(figsize=(14,8))
sns.catplot(data=geartype_avg_torque,x= 'Gear_Type', y='Torque(Nm)', kind='bar',edgecolor=sns.color_palette("dark", 5),facecolor=(0,0,0,0),linewidth=3);
plt.xticks(rotation=30);

#########################################################################################################################################

### There's a no statistical difference in the average velocity of gear_types T,X, Y,Z, can you confirm this hypothesis ?

# visual analysis of the velocity for the 4 Gear_Type
sns.boxplot(x="Gear_Type", y="Velocity", data = data)
plt.show()


##############################################################################################################################################

gear_velocity_T = data[data['Gear_Type']=='T']['Velocity']
gear_velocity_X = data[data['Gear_Type']=='X']['Velocity']
gear_velocity_Y = data[data['Gear_Type']=='Y']['Velocity']
gear_velocity_Z = data[data['Gear_Type']=='Z']['Velocity']

# find the p-value
test_stat, p_value = stats.kruskal(gear_velocity_T, gear_velocity_X,gear_velocity_Y,gear_velocity_Z)
print(p_value)

# print the conclusion based on p-value
if p_value < 0.05:
    print(f'As the p-value {p_value} is less than the level of significance, we reject the null hypothesis.')
else:
    print(f'As the p-value {p_value} is greater than the level of significance, we fail to reject the null hypothesis.')

###################################################################################################################################################

### Are Gear_Type and Device Failure independent of each other ?
### Hypothesis Test : Chisquare Test of Independence

contigency = pd.crosstab(data['Device_Fail'],data['Gear_Type']) 
contigency

plt.figure(figsize=(10,5)) 
sns.heatmap(contigency, annot=True, cmap='Blues', fmt='g')

# Chi-square test of independence. 
c, p, dof, expected = chi2_contingency(contigency) 
# Print the p-value
print(p)

######################################################################################################################################################

### Build a statistical model to determine the odd ratio of device failure

data['Device_Fail']= data['Device_Fail'].map({'Y': 1, 'N': 0})
data['Device_Fail'].unique()

log_reg = smf.logit("Device_Fail ~ Velocity + Gear_Type +Wear_Measure + Sensor_1 + Sensor_2", data=data).fit()
# Summary of results
print(log_reg.summary())
# Inspect paramaters
print(log_reg.params)

odds_ratios = pd.DataFrame(
    {
        "OR": log_reg.params,
        "Lower CI": log_reg.conf_int()[0],
        "Upper CI": log_reg.conf_int()[1],
    }
)
odds_ratios = np.exp(odds_ratios)

print(odds_ratios)




