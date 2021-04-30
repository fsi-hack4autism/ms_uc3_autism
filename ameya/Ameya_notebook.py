
# coding: utf-8




# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '5579dd4a-7aa1-47ce-a82c-656e773f31e8'
resource_group = 'FSIHackathonRG'
workspace_name = 'FSIHackathonWorkspace'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='Ameya_Clean_dataset')



clean_dataset = dataset.to_pandas_dataframe()
clean_dataset.head()



temp_list=[]
for index, row in clean_dataset.iterrows():
    temp_list.append(int(row['TimeTakenToCompleteGoal'][0:2]))
clean_dataset['TimeTakenToCompleteGoalNew'] = temp_list
clean_dataset.groupby('TrialPhase')['TimeTakenToCompleteGoalNew'].mean()


##### How much time does each trial phase take on an average #########

clean_dataset.groupby('TrialPhase')['TimeTakenToCompleteGoalNew'].mean()


######## Which GoalDomain has a high chance of meeting the goal (% format) #######

clean_dataset[clean_dataset['CurrentGoalStatus']=='Met']['GoalDomain'].value_counts(normalize=True)*100


######## How many trials are needed for the GoalDomain to be met? 7 Trials on an average is needed for 'Cognition' GoalDomain to be completed ######

trail_perGoalDomain_dict = clean_dataset[clean_dataset['CurrentGoalStatus']=='Met'].groupby(['GoalDomain'])['Trial'].count()
trail_perGoalDomain_dict_denom = clean_dataset[clean_dataset['CurrentGoalStatus']=='Met'].groupby(['GoalDomain'])['TrialGroupId'].nunique()
trail_perGoalDomain_dict = trail_perGoalDomain_dict.to_dict()
trail_perGoalDomain_dict_denom = trail_perGoalDomain_dict_denom.to_dict()
for key,value in trail_perGoalDomain_dict.items():
    trail_perGoalDomain_dict[key] = int(value/trail_perGoalDomain_dict_denom[key])
trail_perGoalDomain_dict


######## Which author has the highest success rate? #########
# 
# Author Id 613587 has 100% success rate with 369 cases and 61% of them are in the 'Language' GoalDomain



author_to_total_dict = {} #{author:[success,total]}
for index, row in clean_dataset.iterrows():
    if row['TrialAuthorId'] not in author_to_total_dict:
        author_to_total_dict[row['TrialAuthorId']] = [0,0]
    if row['CurrentGoalStatus']=='Met':
        author_to_total_dict[row['TrialAuthorId']][0]+=1
    author_to_total_dict[row['TrialAuthorId']][1]+=1
author_to_success_dict = {}
for author,author_list in author_to_total_dict.items():
    author_to_success_dict[int(author)]= (author_list[0]/author_list[1])*100


####### How long does it take to transition######


temp_dataset = clean_dataset[clean_dataset['TrialAuthorId']==613587]
goal_domain_success_dict= {}
total_entries = len(temp_dataset)
for index, row in temp_dataset.iterrows():
    if row['GoalDomain'] not in goal_domain_success_dict:
        goal_domain_success_dict[row['GoalDomain']] = 0
    if row['CurrentGoalStatus']=='Met':
        goal_domain_success_dict[row['GoalDomain']]+=1

temp_dict={}
for goal_domain, value in goal_domain_success_dict.items():
    temp_dict[goal_domain] = (float(value)/float(total_entries))*100

sorted(temp_dict.items(), key=operator.itemgetter(1),reverse=True)    


