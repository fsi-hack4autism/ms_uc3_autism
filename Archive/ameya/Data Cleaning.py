



import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
m_query_20200512 = dataiku.Dataset("MQuery_05152020_RK")
m_query_20200512_df = m_query_20200512.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

m_query_clean_df = m_query_20200512_df # For this sample code, simply copy input to output


m_query_clean_df.head(10)

import pandas as pd

# Preserve only those rows having the GoalType = 'Target'
m_query_clean_df = m_query_clean_df.loc[m_query_clean_df['GoalType'] == 'Target']
set(list(m_query_clean_df['GoalType']))


# Total number of unique clients and authors
print("Unique Clients: ", m_query_clean_df['ClientId'].nunique())
print("Unique Authors: ", m_query_clean_df['TrialAuthorId'].nunique())

# Clean GoalMetDate column
dataset_new = m_query_clean_df[m_query_clean_df['GoalMetDate'].map(type)!=float]
dataset_new = dataset_new[(dataset_new['GoalMetDate']!='Target') & (dataset_new['GoalMetDate']!='Verbal Behavior Milestone Assessment and Placement Program (VB-MAPP)') & (dataset_new['GoalMetDate']!='Functional Behavior Assessment (FBA)') & (dataset_new['GoalMetDate']!='51:44.2') & (dataset_new['GoalMetDate']!='54:33.3') & (dataset_new['GoalMetDate']!='58:02.6') & (dataset_new['GoalMetDate']!='25:48.2') & (dataset_new['GoalMetDate']!='33:08.5') & (dataset_new['GoalMetDate']!='15:11.3')]
dataset_new['GoalMetDate'].map(type).value_counts()

# Convert the list to dataframe column to further do the testing
dataset_new["GoalInitiatedDateNew"] = goal_initiated_date_list
dataset_new["GoalMetDateNew"] = goal_met_date_list

#Check the datatype of the columns and if not datetime type then convert them
dataset_new["GoalInitiatedDateNew"] = dataset_new["GoalInitiatedDateNew"].astype('datetime64[ns]')
dataset_new["GoalMetDateNew"] = dataset_new["GoalMetDateNew"].astype('datetime64[ns]')

dataset_new['TimeTakenToCompleteGoal'] = dataset_new['GoalMetDateNew']-dataset_new['GoalInitiatedDateNew']


dataset_new['TimeTakenToCompleteGoal']

dataset_new.columns


set(list(dataset_new['GoalAssessment']))


clean_dataset = dataset_new[['TrialId','TrialGroupId','TrialTargetId','TrialPhase','TrialValue','Trial','TrialAuthorId','ClientId','GoalType','CurrentGoalStatus','GoalDomain','GoalAssessment','TimeTakenToCompleteGoal']]

