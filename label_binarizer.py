from sklearn.preprocessing import LabelEncoder, MinMaxScaler, LabelBinarizer
print('done importing packages')


class  MultiLabelBinarizer():

    def __init__(self, columns= None):
        self.columns= columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output= X.copy()

        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelBinarizer().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname]= LabelBinarizer().fit_transform(col)

        return output

    def fit_transform(self,X, y=None):
        return self.fit(X, y).transform(X)



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

#  roof the number of trainings attended at 5 for only training data
#  i.e if the person has attended > 5 trainings make 5 to force the model to understand that 5 is a lot pf trainings

test['Qualification'].map(lambda x: 5 if x > 5 else x)

test['years_spent_in_organisation'] = 2019 - test['Year_of_recruitment']
test['Performance_by_trainings_attended'] = test['Last_performance_score'] / test['Trainings_Attended']
test['years_spent_by_trainings_attended'] = test['years_spent_in_organisation'] /test['Trainings_Attended']

test_cat = test.dtypes == 'object'
cat_cols = test.columns[test_cat].tolist()
#test[cat_cols] = test[cat_cols].apply(lambda cols: lb.fit_transform(cols))
test['Training_score_average_log'] = ( test.Training_score_average).transform(np.log)
#test1 = test.drop('EmployeeNo', axis=1)
#test_scaled = std.fit_transform(test1)
#test_scaled_df = pd.DataFrame(test_scaled, columns=test1.columns)
print()
#print(len(test_scaled_df.columns))

