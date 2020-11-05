# Data Collection
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
testo = pd.read_csv('test.csv')


lb = LabelEncoder()
# minority_sample = train[train.Promoted_or_Not == 1]
# majority_sample = train[train.Promoted_or_Not == 0]
# minority_sampled = resample(minority_sample, replace=True, random_state=42, n_samples=len(majority_sample))
# train = pd.concat([majority_sample, minority_sampled])


train['years_spent_in_organisation'] = 2019 - train['Year_of_recruitment']
train['Performance_by_trainings_attended'] = train['Last_performance_score'] / train['Trainings_Attended']
train['years_spent_by_trainings_attended'] = train['years_spent_in_organisation'] /train['Trainings_Attended']


# test['years_spent_in_organisation'] = 2019 - test['Year_of_recruitment']
# test['Performance_by_trainings_attended'] = test['Last_performance_score'] / test['Trainings_Attended']
# test['years_spent_by_trainings_attended'] = test['years_spent_in_organisation'] /test['Trainings_Attended']


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

division_to_numbers = {'Commercial Sales and Marketing':1,'Customer Support and Field Operations':2,
                       'Sourcing and Purchasing':3,'Information Technology and Solution Support':4,
                       'Information and Strategy':5,'Business Finance Operations':6,'People/HR Management':7,
                       'Regulatory and Legal services': 8, 'Research and Innovation': 9}

maritalMap = {"Married": 1, "Single": 0}
qualification_to_number = {'First Degree or HND': 2, 'MSc, MBA and PhD': 3, 'Non-University Education': 1}
genderMap = {"Male": 1, "Female": 0}
yesnoMap = {"Yes": 1, "No": 0}

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

def bucket_train_score(training_score_average):
    score = int(training_score_average)
    if score <= 44:
        return 0
    if score >= 45 and score <= 49:
        return 1
    if score >= 50 and score <= 59:
        return 2
    if score >= 60 and score <= 69:
        return 3
    if score >= 70:
        return 4

def bucket_perf_score(last_performance_score):
    score = int(last_performance_score)
    if score <= 2.4:
        return 0
    if score > 2.4 and score <= 4.9:
        return 1
    if score > 4.9 and score <= 7.4:
        return 2
    if score > 7.4 and score <= 9.9:
        return 3
    if score > 9.9 and score <= 12.0:
        return 4
    if score > 12.0:
        return 5

def map_employers(x):
    try:
        return int(x)
    except:
        return 5

def bucket_experience(year_of_exp):
    years_exp = 2019-int(year_of_exp)
    if years_exp < 5:
        return "1-4"
    elif years_exp>=5 and years_exp<10:
        return "5-9"
    elif years_exp>=10 and years_exp<15:
        return "10-14"
    elif years_exp>=15 and years_exp<20:
        return "15-20"
    else:
        return "greater than 20"


def transform_data(input_dataset):
    input_data = input_dataset.copy()
    # firstly drop the id column
    input_data = input_data.drop(['EmployeeNo'], 1)

    # map the qualification
    input_data['Qualification'] = input_data['Qualification'].fillna("Non-University Education")
    input_data['Qualification'] = input_data['Qualification'].map(qualification_to_number)

    # map and binarise the dvision
    # label
    input_data['Division'] = lb.fit_transform(input_data['Division'])
    # input_data['Division'] = input_data.Division.map(division_to_numbers)

    # categorise the gender
    input_data["Gender"] = input_data["Gender"].map(genderMap)

    # map past disciplinary actions
    input_data['Past_Disciplinary_Action'] = input_data['Past_Disciplinary_Action'].map(yesnoMap)

    # Floor the trainings attended at 6
    input_data['Trainings_Attended'] = input_data['Trainings_Attended'].map(lambda x: x if x < 5 else 6)

    # roof the number of trainings attended at 5 for only training data i.e if the person has attended > 5 trainings
    # make 5 to force the model to understand that 5 is a lot pf trainings
    input_data['Qualification'].map(lambda x: 5 if x > 5 else x)

    # temporarily testing if bucketing the age is going to give me better data
    input_data['Year_of_birth'] = input_data['Year_of_birth'].map(bucket_age)

    # map employer
    input_data['No_of_previous_employers'] = input_data['No_of_previous_employers'].map(map_employers)

    # map interdep movement
    input_data['Previous_IntraDepartmental_Movement'] = input_data['Previous_IntraDepartmental_Movement'].map(yesnoMap)

    # temporarily testing if bucketing experience is going to help
    input_data['Year_of_recruitment'] = input_data['Year_of_recruitment'].map(bucket_experience)

    # temporarily testing if bucketing training score is going to help
    input_data['Training_score_average'] = input_data['Training_score_average'].map(bucket_train_score)

    # temporarily testing if bucketing performance score is going to help
    input_data['Last_performance_score'] = input_data['Last_performance_score'].map(bucket_perf_score)

    # transform channel of recruitment
    input_data['Channel_of_Recruitment'] = lb.fit_transform(input_data['Channel_of_Recruitment'])

    # map foreign schooled
    input_data['Foreign_schooled'] = input_data['Foreign_schooled'].map(yesnoMap)

    # map marital status
    input_data['Marital_Status'] = input_data['Marital_Status'].map(lambda x: "Single" if x == "Not_Sure" else x).map(
        maritalMap)

    # map state of origin to tribe and binarise it
    input_data['State_Of_Origin'] = lb.fit_transform(
        input_data['State_Of_Origin'].map(lambda x: states_to_tribe.get(x, "YORUBA")))

    # encode categorical data
    input_cat = input_data.dtypes == object
    categorical_cols = input_data.columns[input_cat].tolist()
    input_data[categorical_cols] = input_data[categorical_cols].apply(lambda col: lb.fit_transform(col))

    return input_data


df_train = transform_data(train)
df_test = transform_data(test)


df_label = df_train.Promoted_or_Not
df = df_train.drop('Promoted_or_Not', axis=1)
x_train, x_test, y_train, y_test = train_test_split(df, df_label, test_size=0.3, random_state=123)

log_reg = LogisticRegression(C=0.7, max_iter=200)
log_reg.fit(x_train, y_train)
print('score of log_reg', log_reg.score(x_train, y_train))
train_pred = log_reg.predict(x_test)
print()
print('the f1 score is', f1_score(y_test, train_pred))
print()
print('this is the classification report', classification_report(y_test, train_pred))
print()
print('the cross validation score is: ', cross_val_score(estimator=log_reg, X=x_train, y=y_train, scoring='accuracy',
                                                         cv=10, ).mean())


