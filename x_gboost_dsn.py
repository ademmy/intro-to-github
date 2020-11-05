import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


# reading in the data
train = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

full_dataset = train.copy()

# re sampling
minority_sample = full_dataset[full_dataset.Promoted_or_Not == 1]
majority_sample = full_dataset[full_dataset.Promoted_or_Not == 0]
minority_sampled = resample(minority_sample, replace=True, random_state=42, n_samples=len(majority_sample))
full_dataset1 = pd.concat([majority_sample, minority_sampled])


full_dataset1['years_spent_in_organisation'] = 2019 - full_dataset1['Year_of_recruitment']
full_dataset1['Performance_by_trainings_attended'] = full_dataset1['Last_performance_score'] / full_dataset1['Trainings_Attended']
full_dataset1['years_spent_by_trainings_attended'] = full_dataset1['years_spent_in_organisation'] /full_dataset1['Trainings_Attended']

#  data transformation process

division_to_numbers = {'Commercial Sales and Marketing': 1, 'Customer Support and Field Operations': 2,
                       'Sourcing and Purchasing': 3, 'Information Technology and Solution Support': 4,
                       'Information and Strategy': 5, 'Business Finance Operations': 6, 'People/HR Management': 7,
                       'Regulatory and Legal services': 8, 'Research and Innovation': 9}
# merging Not_sure to single, so the model doesn't learn gabbage as stated above
maritalstatus_to_nums = {'Married': 1, 'Single': 2, 'Not_Sure': 2}

qualification_to_number={'First Degree or HND':2, 'MSc, MBA and PhD':3, 'Non-University Education':1}

states_to_tribe = {
    'ABIA': "IGBO",
    'ADAMAWA': "HAUSA",
    'AKWA IBOM': "IGBO",
    'ANAMBRA': "IGBO",
    'BAUCHI': "IGBO",
    'BAYELSA': "IGBO",
    'BENUE': "HAUSA",
    'BORNO': "HAUSA",
    'CROSS RIVER': "IGBO",
    'DELTA': "IGBO",
    'EBONYI': "IGBO",
    'EDO': "IGBO",
    'EKITI': "YORUBA",
    'ENUGU': "IGBO",
    'FCT': "HAUSA",
    'GOMBE': "HAUSA",
    'IMO': "IGBO",
    'JIGAWA': "HAUSA",
    'KADUNA': "HAUSA",
    'KANO': 'HAUSA',
    'KATSINA': "HAUSA",
    'KEBBI': "HAUSA",
    'KOGI': "HAUSA",
    'KWARA': "HAUSA",
    "LAGOS": "YORUBA",
    'NASSARAWA': "HAUSA",
    'NIGER': "HAUSA",
     'OGUN': "YORUBA",
     'ONDO': "YORUBA",
     'OSUN': "YORUBA",
     'OYO': "YORUBA",
     'PLATEAU': "HAUSA",
     'RIVERS': "IGBO",
     'SOKOTO': "HAUSA",
     'TARABA': "HAUSA",
     'YOBE': "HAUSA",
     'ZAMFARA': "HAUSA"
    }

def bucket_age(year_of_birth):
    age = 2019-int(year_of_birth)
    if age < 21:
        return "less than 21"
    elif age >= 21 and age < 28:
        return "21-27"
    elif age >= 28 and age < 36:
        return "28-35"
    elif age >= 36 and age < 50:
        return "36-50"
    else:
        return "greater than 50"


def bucket_experience(year_of_exp):
    years_exp = 2019-int(year_of_exp)
    if years_exp < 5:
        return "1-4"
    elif years_exp >= 5 and years_exp < 10:
        return "5-9"
    elif years_exp >= 10 and years_exp < 15:
        return "10-14"
    elif years_exp >= 15 and years_exp < 20:
        return "15-20"
    else:
        return "greater than 20"

full_dataset1.head(10)

# transformer function

qualification_binariser = LabelBinarizer()
not_biniarize = ['Qualification', "Trainings_Attended", "Last_performance_score", "Training_score_average"]


def transform_data(input_dataset, is_training=True):
    input_data = input_dataset.copy()
    # firstsly drop the id column
    input_data = input_data.drop(['EmployeeNo'], 1)

    # map the qualification
    input_data['Qualification'] = input_data['Qualification'].fillna("Non-University Education")
    input_data['Qualification'] = input_data['Qualification'].map(qualification_to_number)

    # map the Division
    input_data['Division'] = input_data.Division.map(division_to_numbers)
    # map the Marital_Status
    input_data['Marital_Status'] = input_data.Marital_Status.map(maritalstatus_to_nums)

    if is_training:
        # Floor the trainings attended at 6
        input_data['Trainings_Attended'] = input_data['Trainings_Attended'].map(lambda x: x if x < 5 else 6)

    # roof the number of trainings attended at 5 for only training data
    # i.e if the person has attended > 5 trainings make 5 to force the model to understand that 5 is a lot pf trainings
    if is_training:
        input_data['Qualification'].map(lambda x: 5 if x > 5 else x)

        # temporarily testing if bucketing the age is going to give me better data
    input_data['Year_of_birth'] = input_data['Year_of_birth'].map(bucket_age)

    # temporarily testing if bucketing experience is going to help
    input_data['Year_of_recruitment'] = input_data['Year_of_recruitment'].map(bucket_experience)

    # map the states respectively to the three geo zones
    input_data['State_Of_Origin'] = full_dataset1['State_Of_Origin'].map(lambda x: states_to_tribe.get(x, "YORUBA"))

    return input_data

print('done...')

df = full_dataset1.copy()
df1 = transform_data(df)

df_cat = df1.dtypes == 'object'
categorical_cols = df1.columns[df_cat].tolist()

lb = LabelEncoder()
df1[categorical_cols] = df1[categorical_cols].apply(lambda col: lb.fit_transform(col))

df_label = df1.Promoted_or_Not
df2 = df1.drop('Promoted_or_Not', axis=1)
df_scaled = df2.copy()
df_scaled['Training_score_average_log'] = (df_scaled.Training_score_average).transform(np.log)
minmax = MinMaxScaler()
std = StandardScaler()
df_minmax_scaled = std.fit_transform(df_scaled)
df_scaled1 = pd.DataFrame(df_minmax_scaled, columns=df_scaled.columns)

print()
print(len(df_scaled1.columns))
print('Done with the training side of things')
print()

# test data
test = test_df.copy()

test['years_spent_in_organisation'] = 2019 - test['Year_of_recruitment']
test['Performance_by_trainings_attended'] = test['Last_performance_score'] / test['Trainings_Attended']
test['years_spent_by_trainings_attended'] = test['years_spent_in_organisation'] /test['Trainings_Attended']


test.Qualification = test.Qualification.fillna("Non-University Education")
test.Qualification = test.Qualification.map(qualification_to_number)

# temporarily testing if bucketing the age is going to give me better data
test['Year_of_birth'] = test['Year_of_birth'].map(bucket_age)

# temporarily testing if bucketing experience is going to help
test['Year_of_recruitment'] = test['Year_of_recruitment'].map(bucket_experience)

# map the states respectively to the three geo zones
test['State_Of_Origin'] = test['State_Of_Origin'].map(lambda x: states_to_tribe.get(x, "YORUBA"))

# map the Division
test['Division'] = test.Division.map(division_to_numbers)

# map the Marital_Status
test['Marital_Status'] = test.Marital_Status.map(maritalstatus_to_nums)

# Floor the trainings attended at 6
test['Trainings_Attended'] = test['Trainings_Attended'].map(lambda x: x if x < 5 else 6)

test['Qualification'].map(lambda x: 5 if x > 5 else x)

test_cat = test.dtypes == 'object'
cat_cols = test.columns[test_cat].tolist()
test[cat_cols] = test[cat_cols].apply(lambda cols: lb.fit_transform(cols))
test['Training_score_average_log'] = (test.Training_score_average).transform(np.log)

test1 = test.drop('EmployeeNo', axis=1)

# scaling with StandardScaler
test_scaled = std.fit_transform(test1)
test_scaled_df = pd.DataFrame(test_scaled, columns=test1.columns)
print()
print(len(test_scaled_df.columns))

clf = XGBClassifier(base_score=0.7, booster='dart', objective='binary:logistic', colsample_bytree=0.8, learning_rate=0.1,
                    max_depth=8, subsample=0.8, eta=0.05, min_child_weight=3,
                    reg_lambda=0.03, reg_alpha=0.1, gamma=0, n_estimators=100)

clf.fit(df_scaled1, df_label)
print(clf.score(df_scaled1, df_label))





sub.to_csv('logistic_regression3.csv', index=False)

dsc = DecisionTreeClassifier()
dsc.fit(df, df_label)
print(dsc.score(df, df_label))

dsc_pred = dsc.predict(df_test)
dsc_pred = pd.Series(dsc_pred, name='Promoted_or_Not')
sub1 = eid.join(dsc_pred, how='left')
sub1.to_csv('decision_trees.csv', index=False)

clf = XGBClassifier(base_score=0.7, booster='dart', objective ='binary:logistic', colsample_bytree = 0.8, learning_rate = 0.1,
                max_depth = 8, subsample=0.8, eta=0.05, min_child_weight=3, reg_lambda=0.03, reg_alpha=0.1, gamma=0, n_estimators = 100)

clf.fit(df, df_label)
print(clf.score(df, df_label))
predy = clf.predict(df_test)
xg_pred = pd.Series(predy, name='Promoted_or_Not')
sub6 = eid.join(xg_pred)
sub6.to_csv('xgboost1.csv', index=False)
