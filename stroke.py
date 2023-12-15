# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:01:32 2022

@author: Mehak
"""

# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
# dataset in CSV format
df = pd.read_csv('E:\Msc Advanced Computer Science\Semester 2\Data Science\Project\healthcare-dataset-stroke-data.csv')

pd.options.display.max_columns = None


# Droping column name 'id' becasue it is useless information for us
df.drop("id", axis=1, inplace=True)

#finding statistic measures about dataset
print(df.describe(include='all'))
# printing dimension
print(df.shape)

#There are 5110 instances with 12 variables.
#There are 8 categorical variables and 3 numerical variables (age, avg_glucose_level and bmi).

# printing top 5 records
print(df.head(5))

# an array of categorical data
cat_var = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
# loop for ploting bar charts 
for i in cat_var:
    # this line is for separating each categorical variable bar chart so that no bar chart will overlap
    plt.figure(i)
    # ploting bar chart of each variable
    sns.countplot(df[i])


plt.figure(figsize=(10,6))
ax=sns.countplot(x='smoking_status',data=df, palette='mako',hue='stroke')
plt.title("Count of people in each Smoking Group, Separated by Stroke")
for p in ax.patches:
        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+50))

plt.figure(figsize=(10,6))
g = sns.catplot(x='work_type',y='stroke', col = 'Residence_type', data=df, kind='bar', palette='rocket', saturation =2.5)
(g.set_axis_labels("Work Type", "Stroke Rate").set_titles("{col_name}").set(ylim=(0,0.15)))
g.fig.set_figwidth(10)
g.fig.set_figheight(2)

plt.figure(figsize=(10,6))
sns.distplot(df['age'], kde=False, color='black', bins=40)


sns.heatmap(df.corr(), annot = True, cmap = 'rocket')


# Dropping 'Other' from Gender
df.drop(df[df.gender == 'Other'].index, axis = 0, inplace=True)
data_copy = df.copy()


# Dummy Encoding of gender, ever_married and residence type
dummy_df = pd.get_dummies(df.iloc[:, [0, 4, 6]], drop_first=True)
# Dummy Encoding of Gender (replacing 'male' and 'female' with binary values 0 and 1)
df['gender'] = dummy_df.iloc[:, 0]
# Dummy encoding of ever_married
df['ever_married'] = dummy_df.iloc[:, 1]
# Dummy encoding of Residence_type
df['Residence_type'] = dummy_df.iloc[:, 2]
print(df.head())
# Frequency Encoding
freq_smoking = (df.groupby('smoking_status').size()) / len(df)
freq_smoking['Unknown'] = 0 
df['smoking_status'] = df['smoking_status'].apply(lambda x : freq_smoking[x])

freq_work = (df.groupby('work_type').size()) / len(df)
df['work_type'] = df['work_type'].apply(lambda x : freq_work[x])

print(df.head())

#Findind outliers of numeric data
print("age")
upper_fence_age = df["age"].mean() + 3*df["age"].std()
lower_fence_age = df["age"].mean() - 3*df["age"].std()
print("Highest allowed: ", upper_fence_age)
print("Lowest allowed: ", lower_fence_age)
outliers_age = df[(df["age"] > upper_fence_age) | (df["age"] < lower_fence_age)]
print(outliers_age.shape[0])


print("avg_glucose_level")
upper_fence_glucose = df["avg_glucose_level"].mean() + 3*df["avg_glucose_level"].std()
lower_fence_glucose = df["avg_glucose_level"].mean() - 3*df["avg_glucose_level"].std()
print("Highest allowed",upper_fence_glucose)
print("Lowest allowed", lower_fence_glucose)
outliers_glucose = df[(df["avg_glucose_level"] > upper_fence_glucose) | (df["avg_glucose_level"] < lower_fence_glucose)]
print(outliers_glucose.shape[0])

print("bmi")
upper_fence_bmi = df["bmi"].mean() + 3*df["bmi"].std()
lower_fence_bmi = df["bmi"].mean() - 3*df["bmi"].std()
print("Highest allowed: ", upper_fence_bmi)
print("Lowest allowed: ", lower_fence_bmi)
outliers_bmi = df[(df["bmi"] > upper_fence_bmi) | (df["bmi"] < lower_fence_bmi)]
print(outliers_bmi.shape[0])

#removing outliers from avg_glucose_level and bmi
outliers = pd.concat([outliers_glucose,outliers_bmi]).drop_duplicates()
df.drop(outliers.index, axis=0, inplace=True)

# age, avg_glucose_level and bmi need to be scaled
scaler = MinMaxScaler()
df[["age", "avg_glucose_level", "bmi"]] = scaler.fit_transform(df[["age", "avg_glucose_level", "bmi"]])
print(df.head(3))

#Lets find missing data
msno.matrix(df, sparkline=False, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0));

n_miss = df["bmi"].isnull().sum()
perc = n_miss / df.shape[0] * 100
print('> %s, Missing: %d (%.1f%%)' % (df["bmi"], n_miss, perc))

#Only bmi (body mass index) column has missing data. Let's find out whether it can be simply removed or must be replaced with some value:
print('Missing values of BMI: %s' % (df.loc[df.stroke == 1].shape[0]))

print('Positive Stroke cases of missed BMI: %s' % (df.loc[(df.bmi.isna() == True) & (df.stroke == 1)].shape[0]))

#replacing null bmi with mean value
#df['bmi'] = df['bmi'].fillna(np.round(df.bmi.mean(), 2))
df=df.fillna(np.mean(df['bmi']))

#checking bmi values has been replaced by mean value
print(df.isnull().sum())
msno.matrix(df, sparkline=False, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0));

pie_colors = ('Green', 'Red')

pred_classes = df.stroke.value_counts()
 
plt.figure(figsize=(17, 12))
patches, texts, pcts = plt.pie(pred_classes,
                               labels=['no', 'yes'],
                               colors=pie_colors,
                               pctdistance=0.65,
                               shadow=True,
                               startangle=90,
                               autopct='%1.1f%%',
                               textprops={'fontsize': 20,
                                          'color': 'black',
                                          'weight': 'bold',
                                          'family': 'serif'})
plt.setp(pcts, color='white')

hfont = {'fontname':'serif', 'weight': 'bold'}
plt.title('Patient had a stroke', size=45, **hfont)

centre_circle = plt.Circle((0,0),0.40,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)


plt.figure(figsize=(17, 12))

#Model Training
training_data=df.copy()
x= training_data.drop(['stroke'],axis=1)
y= df['stroke']

#Splitting into Training and Test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print(x_train.shape, x_test.shape,  y_train.shape, y_test.shape)

#Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#sm = SMOTE()
#x_train, y_train = sm.fit_resample(x_train,y_train)

#Model Training using DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()   
decision_tree.fit(x_train,y_train)
dt_pred = decision_tree.predict(x_test)
dt_acc = accuracy_score(dt_pred, y_test)
print('DecisionTreeClassifier: %s' % dt_acc)




#Model Training using RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 25)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_acc = accuracy_score(rf_pred, y_test)
print('RandomForestClassifier: %s' % rf_acc)

#Model Training using KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_acc = accuracy_score(knn_pred, y_test)
print('KNeighborsClassifier %s' % knn_acc)

#Model Training using LGBMClassifier
lgbm = LGBMClassifier(random_state = 42)
lgbm.fit(x_train, y_train)
lgbm_pred = lgbm.predict(x_test)
lgbm_acc = accuracy_score(lgbm_pred, y_test)
print('LGBMClassifier: %s' % lgbm_acc)

#Model Training using XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
xgb_acc = accuracy_score(xgb_pred, y_test)
print('XGBClassifier: %s ' % xgb_acc)

#Model Training using GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
gnb_acc = accuracy_score(y_pred, y_test)
print('Accuracy score: %s ' % gnb_acc)
      

#visualization for all model with their Accuracy
model_acc=[]
model_name=[]

model_acc.append(dt_acc)
model_name.append("DecisionTreeClassifier")
model_acc.append(gnb_acc)
model_name.append("GaussianNaiveBayes")
model_acc.append(rf_acc)
model_name.append("RandomForestClassifier")

model_acc.append(knn_acc)
model_name.append("KNeighborsClassifier")
model_acc.append(lgbm_acc)
model_name.append("LGBMClassifier") 


#models_names = ["LogisticRegression",'DecisionTreeClassifier','RandomForestClassifier','XGBClassifier',
#                    'KNeighborsClassifier','LGBMClassifier','SVC']
#models_acc=[lr_acc,dt_acc,rf_acc,xgb_acc,knn_acc,lgbm_acc,svm_acc]

plt.rcParams['figure.figsize']=8,6

#colors = ['#1b9e77', '#a9f971', '#fdaa48','#6890F0','#A890F0']
ax = sns.barplot(x=model_name, y=model_acc, palette = "rainbow", saturation =1.5)
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

plt.xlabel('Classifier Models' )
plt.ylabel('Accuracy')
plt.title('Accuracy of different Classifier Models \n')

plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = '16')
plt.show()


#Making confusion matrix
rf_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, knn_pred)
print(cm)
print(accuracy_score(y_test, knn_pred))


#Applying K-Fold Cross Validation
accuracies = cross_val_score(estimator=knn , X=x_train , y=y_train , cv=10)
print('Accuracy: {:.2f} %'.format(accuracies.mean()*100))
print('Standard deviation: {:.2f} %'.format(accuracies.std()*100))


