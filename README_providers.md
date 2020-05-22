```python
import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import sklearn
from sklearn import preprocessing
from collections import defaultdict
```

### Importing Data and Basic Analysis


```python
import csv
df_raw = pd.read_csv('data/M_query_20200512.csv', sep=',',skipinitialspace=True,
                    engine='python', error_bad_lines=False,quoting=csv.QUOTE_ALL, warn_bad_lines=False)
```


```python
df_raw.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 547854 entries, 0 to 547853
    Data columns (total 23 columns):
     #   Column                   Non-Null Count   Dtype  
    ---  ------                   --------------   -----  
     0   TrialId                  547854 non-null  int64  
     1   TrialGroupId             547854 non-null  int64  
     2   TrialTargetId            547854 non-null  int64  
     3   TrialPhase               547854 non-null  object 
     4   TrialValue               547854 non-null  float64
     5   Trial                    547854 non-null  int64  
     6   TrialCreatedDate         547854 non-null  object 
     7   TrialDataDate            547854 non-null  object 
     8   TrialAuthorId            547854 non-null  int64  
     9   GoalName                 547854 non-null  object 
     10  ClientId                 547854 non-null  int64  
     11  GoalType                 547854 non-null  object 
     12  CurrentGoalStatus        547854 non-null  object 
     13  GoalDomain               547854 non-null  object 
     14  GoalAssessment           547854 non-null  object 
     15  GoalCreatedDate          547854 non-null  object 
     16  GoalInitiatedDate        547854 non-null  object 
     17  GoalMetDate              357161 non-null  object 
     18  GoalInProgressDate       124408 non-null  object 
     19  GoalHoldDate             28308 non-null   object 
     20  GoalDiscontinuedDate     15126 non-null   object 
     21  GoalDataType             547854 non-null  object 
     22  GoalPercentCorrectTrend  547854 non-null  float64
    dtypes: float64(2), int64(6), object(15)
    memory usage: 96.1+ MB



```python
df_raw.replace("?",np.nan,inplace=True)
```


```python
#helper function
def success_rate(sumval, total):
    return (sumval/total)*100
```


```python
def normalize_values(df):
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df
```


```python
df_sum = df_raw.groupby(['ClientId','TrialDataDate','GoalDomain','TrialAuthorId','CurrentGoalStatus'],as_index=False)['TrialValue'].agg('sum')
df_total = df_raw.groupby(['ClientId','TrialDataDate','GoalDomain','TrialAuthorId','CurrentGoalStatus']).size().reset_index(name='TrialsPerDatePerAuthor')

```


```python
df_total.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClientId</th>
      <th>TrialDataDate</th>
      <th>GoalDomain</th>
      <th>TrialAuthorId</th>
      <th>CurrentGoalStatus</th>
      <th>TrialsPerDatePerAuthor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>88550</td>
      <td>2018-02-26 15:54:00</td>
      <td>Language</td>
      <td>470123</td>
      <td>Met</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88550</td>
      <td>2018-03-01 15:53:00</td>
      <td>Language</td>
      <td>470123</td>
      <td>Met</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>88550</td>
      <td>2018-03-05 15:51:00</td>
      <td>Language</td>
      <td>470123</td>
      <td>Met</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>88550</td>
      <td>2018-03-09 06:45:00</td>
      <td>Language</td>
      <td>344083</td>
      <td>Met</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>88550</td>
      <td>2018-03-16 07:42:00</td>
      <td>Language</td>
      <td>344083</td>
      <td>Met</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df_sum['TrialValue_norm'] = normalize_values(df_sum['TrialValue'])
#df_total['TrialsPerDatePerAuthor_norm'] = normalize_values(df_total['TrialsPerDatePerAuthor'])
```


```python
df_sum['TrialValue'] = df_sum['TrialValue']
df_total['TrialsPerDatePerAuthor'] = df_total['TrialsPerDatePerAuthor']
```


```python
df_sum.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClientId</th>
      <th>TrialDataDate</th>
      <th>GoalDomain</th>
      <th>TrialAuthorId</th>
      <th>CurrentGoalStatus</th>
      <th>TrialValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>88550</td>
      <td>2018-02-26 15:54:00</td>
      <td>Language</td>
      <td>470123</td>
      <td>Met</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88550</td>
      <td>2018-03-01 15:53:00</td>
      <td>Language</td>
      <td>470123</td>
      <td>Met</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>88550</td>
      <td>2018-03-05 15:51:00</td>
      <td>Language</td>
      <td>470123</td>
      <td>Met</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>88550</td>
      <td>2018-03-09 06:45:00</td>
      <td>Language</td>
      <td>344083</td>
      <td>Met</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>88550</td>
      <td>2018-03-16 07:42:00</td>
      <td>Language</td>
      <td>344083</td>
      <td>Met</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_total.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClientId</th>
      <th>TrialDataDate</th>
      <th>GoalDomain</th>
      <th>TrialAuthorId</th>
      <th>CurrentGoalStatus</th>
      <th>TrialsPerDatePerAuthor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>88550</td>
      <td>2018-02-26 15:54:00</td>
      <td>Language</td>
      <td>470123</td>
      <td>Met</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88550</td>
      <td>2018-03-01 15:53:00</td>
      <td>Language</td>
      <td>470123</td>
      <td>Met</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>88550</td>
      <td>2018-03-05 15:51:00</td>
      <td>Language</td>
      <td>470123</td>
      <td>Met</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>88550</td>
      <td>2018-03-09 06:45:00</td>
      <td>Language</td>
      <td>344083</td>
      <td>Met</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>88550</td>
      <td>2018-03-16 07:42:00</td>
      <td>Language</td>
      <td>344083</td>
      <td>Met</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sum['SuccessPercentage'] = success_rate((df_sum['TrialValue']),(df_total['TrialsPerDatePerAuthor']))
                                           
```


```python
#df_sum['SuccessPercentage_norm'] = normalize_values(success_rate((df_sum['TrialValue_norm']),(df_total['TrialsPerDatePerAuthor_norm'])))
#df_sum['SuccessPercentage'] = normalize_values(df_sum['SuccessPercentage_raw'])                                          
```


```python
#df_sum.head(20)
```


```python
df_tot_cases = df_total.groupby(['ClientId']).size().reset_index(name='TotalClientCases')
df_totalclient_suc = df_sum.groupby(['ClientId'],as_index=False)['SuccessPercentage'].mean()
```

#### Identify clients with low success rate and see if it is possible to refer to domain experts of the problem area


```python
df_totalclient_suc.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClientId</th>
      <th>SuccessPercentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1208</th>
      <td>1264327</td>
      <td>96.527778</td>
    </tr>
    <tr>
      <th>1209</th>
      <td>1272784</td>
      <td>19.333333</td>
    </tr>
    <tr>
      <th>1210</th>
      <td>1275765</td>
      <td>95.000000</td>
    </tr>
    <tr>
      <th>1211</th>
      <td>1278577</td>
      <td>27.525253</td>
    </tr>
    <tr>
      <th>1212</th>
      <td>1282868</td>
      <td>66.666667</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_totalclient_suc.sort_values(by=['SuccessPercentage'],ascending=False).tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClientId</th>
      <th>SuccessPercentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1154</th>
      <td>1124604</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1079</th>
      <td>1044572</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>673</th>
      <td>687874</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>729</th>
      <td>718414</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>856</th>
      <td>819281</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Let's check out the clientId 718414 with zero success rate. Let's grab the TraiAuthorId to see the expertise of 


```python
df_raw[df_raw.ClientId==819281].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TrialId</th>
      <th>TrialGroupId</th>
      <th>TrialTargetId</th>
      <th>TrialPhase</th>
      <th>TrialValue</th>
      <th>Trial</th>
      <th>TrialCreatedDate</th>
      <th>TrialDataDate</th>
      <th>TrialAuthorId</th>
      <th>GoalName</th>
      <th>...</th>
      <th>GoalDomain</th>
      <th>GoalAssessment</th>
      <th>GoalCreatedDate</th>
      <th>GoalInitiatedDate</th>
      <th>GoalMetDate</th>
      <th>GoalInProgressDate</th>
      <th>GoalHoldDate</th>
      <th>GoalDiscontinuedDate</th>
      <th>GoalDataType</th>
      <th>GoalPercentCorrectTrend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>152954</th>
      <td>2325064478</td>
      <td>533873395</td>
      <td>81407697</td>
      <td>baseline</td>
      <td>0.0</td>
      <td>1</td>
      <td>2019-12-11 17:41:30.597000000</td>
      <td>2019-12-11 12:41:00</td>
      <td>1110314</td>
      <td>Goes to desk to work on ind work</td>
      <td>...</td>
      <td>Learning Readiness</td>
      <td>Verbal Behavior Milestone Assessment and Place...</td>
      <td>2019-12-11 16:13:36.863000000</td>
      <td>2019-12-11</td>
      <td>NaN</td>
      <td>2019-12-11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>datapercent</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>152955</th>
      <td>2325064479</td>
      <td>533873395</td>
      <td>81407697</td>
      <td>baseline</td>
      <td>0.0</td>
      <td>2</td>
      <td>2019-12-11 17:41:30.597000000</td>
      <td>2019-12-11 12:41:00</td>
      <td>1110314</td>
      <td>Goes to desk to work on ind work</td>
      <td>...</td>
      <td>Learning Readiness</td>
      <td>Verbal Behavior Milestone Assessment and Place...</td>
      <td>2019-12-11 16:13:36.863000000</td>
      <td>2019-12-11</td>
      <td>NaN</td>
      <td>2019-12-11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>datapercent</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>152956</th>
      <td>2325064480</td>
      <td>533873395</td>
      <td>81407697</td>
      <td>baseline</td>
      <td>0.0</td>
      <td>3</td>
      <td>2019-12-11 17:41:30.597000000</td>
      <td>2019-12-11 12:41:00</td>
      <td>1110314</td>
      <td>Goes to desk to work on ind work</td>
      <td>...</td>
      <td>Learning Readiness</td>
      <td>Verbal Behavior Milestone Assessment and Place...</td>
      <td>2019-12-11 16:13:36.863000000</td>
      <td>2019-12-11</td>
      <td>NaN</td>
      <td>2019-12-11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>datapercent</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>152957</th>
      <td>2325654310</td>
      <td>534011102</td>
      <td>81407697</td>
      <td>Intervention</td>
      <td>0.0</td>
      <td>1</td>
      <td>2019-12-11 18:43:13.970000000</td>
      <td>2019-12-11 13:43:00</td>
      <td>1110314</td>
      <td>Goes to desk to work on ind work</td>
      <td>...</td>
      <td>Learning Readiness</td>
      <td>Verbal Behavior Milestone Assessment and Place...</td>
      <td>2019-12-11 16:13:36.863000000</td>
      <td>2019-12-11</td>
      <td>NaN</td>
      <td>2019-12-11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>datapercent</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>152958</th>
      <td>2325654311</td>
      <td>534011102</td>
      <td>81407697</td>
      <td>Intervention</td>
      <td>0.0</td>
      <td>2</td>
      <td>2019-12-11 18:43:13.970000000</td>
      <td>2019-12-11 13:43:00</td>
      <td>1110314</td>
      <td>Goes to desk to work on ind work</td>
      <td>...</td>
      <td>Learning Readiness</td>
      <td>Verbal Behavior Milestone Assessment and Place...</td>
      <td>2019-12-11 16:13:36.863000000</td>
      <td>2019-12-11</td>
      <td>NaN</td>
      <td>2019-12-11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>datapercent</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



### Ranking expertise of authors in a specific Goal Domain


```python
df_tot_cases_author = df_total.groupby(['TrialAuthorId','GoalDomain']).size().reset_index(name='TotalClientCases')
df_totalauthor_suc = df_sum.groupby(['TrialAuthorId','GoalDomain'],as_index=False)['SuccessPercentage'].mean()
```


```python
df_totalauthor_suc.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TrialAuthorId</th>
      <th>GoalDomain</th>
      <th>SuccessPercentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>81006</td>
      <td>Academic</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>81852</td>
      <td>Communication</td>
      <td>75.555556</td>
    </tr>
    <tr>
      <th>2</th>
      <td>81852</td>
      <td>Imitation</td>
      <td>26.666667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>81852</td>
      <td>Learning Readiness</td>
      <td>61.111111</td>
    </tr>
    <tr>
      <th>4</th>
      <td>95238</td>
      <td>Language</td>
      <td>89.583333</td>
    </tr>
  </tbody>
</table>
</div>



### Ranking experts based on goal domain


```python
df_auth_rank_pre = df_totalauthor_suc.sort_values(by=['SuccessPercentage'],ascending = False)
```


```python
df_auth_rank = df_auth_rank_pre[['GoalDomain', 'TrialAuthorId', 'SuccessPercentage']].sort_values(by=['GoalDomain'])
```


```python
df_domain_rank = df_auth_rank.sort_values(by=['SuccessPercentage'],ascending = False)
```


```python
df_domain_rank.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GoalDomain</th>
      <th>TrialAuthorId</th>
      <th>SuccessPercentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1912</th>
      <td>Imitation</td>
      <td>561685</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1424</th>
      <td>Communication</td>
      <td>508433</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>4256</th>
      <td>Communication</td>
      <td>886942</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1280</th>
      <td>Communication</td>
      <td>485433</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1640</th>
      <td>Communication</td>
      <td>528867</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_domain_rank[df_domain_rank.TrialAuthorId==1110314]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GoalDomain</th>
      <th>TrialAuthorId</th>
      <th>SuccessPercentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5536</th>
      <td>Learning Readiness</td>
      <td>1110314</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



#### As we can see above the "TraiAuthorId" of the specific client with low success percentage is zero

##### Now, let's check which providers / AuthorId are having high success percentage in this Goal Domain


```python
df_domain_rank[df_domain_rank.GoalDomain=="Learning Readiness"].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GoalDomain</th>
      <th>TrialAuthorId</th>
      <th>SuccessPercentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1333</th>
      <td>Learning Readiness</td>
      <td>492655</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>Learning Readiness</td>
      <td>508473</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>Learning Readiness</td>
      <td>508468</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1503</th>
      <td>Learning Readiness</td>
      <td>513709</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>3971</th>
      <td>Learning Readiness</td>
      <td>845557</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>



### This work is based on preliminary exploratory analysis and demonstrates a suitable feature for a Machine Learning system to be able to recommend right service providers based on the expertise.


```python

```
