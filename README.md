# FSI Autism Hackathon

## Use Case #3: Application of Machine Learning to ABA Data

### Below are the deliverables of the Hackathon
* KPI for Goal Effectiveness 
* KPI for Provider Effectiveness (rollup of KPI for Goal Effectiveness) 
* Combined measures that predict the likelihood and speed of future success 

### Data
* CR(CentralReach) x Microsoft Clinical Data Project CentralReach
![RAW Data](https://github.com/dipeshtech/ms_uc3_autism/blob/master/images/sample_raw.png)


### Part 1: Supervised Model: To predict the success of a goal

#### Data Cleaning & Pre-Processing

* The bad records from the file with seperators in the data were dropped.
* Bad data was imputed.
* If the goal is met, we flag it as 1. All the other goal statuses are flagged as 0.

#### Feature Engineering:

* Goal Length : Difference between the Goal End Date and Goal Start Date in Days
* Goal Start Date: When the Goal was initiated 
* Goal Met Date, Goal Hold Date, Goal In progress date, Goal Discontinued date (Depending upon the goal status)
* Goal Domain - One hot encoded feature
* Number of unique trials per Trial Target Id
* Number of unique trial groups per Trial Target Id
* Number of unique trial authors per Trial Target Id
* Total number of successful trials
* Total number of failed trials
* Interaction between goal length and number of successful/failed trials
* Interaction between number of unique authors in the goal and number of successful/failed trials

#### Models Trained: 
* Random Forest
* XGBoost
* Logistic Regression

##### Variable Importance:

* Random Forest Variable Importance
![Variable Importance](https://github.com/dipeshtech/ms_uc3_autism/blob/master/images/variable_importance_RF.png)

##### Confusion Martrix
![Confusion Matrix](https://github.com/dipeshtech/ms_uc3_autism/blob/master/images/confusion_matrix_rf.png)

##### ROC-AUC Curve:
![Confusion Matrix](https://github.com/dipeshtech/ms_uc3_autism/blob/master/images/roc_auc_rf.png)


### Part 2: Un-Supervised Model: To predict the success of a goal

#### Data Cleaning & Pre-Processing

* The bad records from the file with seperators in the data were dropped.
* Bad data was imputed.
* If the goal is met, we flag it as 1. All the other goal statuses are flagged as 0.

#### Feature Engineering:

* We tried to cluster the data based on the feature below and Goal Domain.
* Calculated a feature date_diff_to_success = difference between the Goal Met and Trial Date per client
* goals_met_per_trial_phase

##### Variable Importance:

* K-Means Variable Importance
![Variable Importance](https://github.com/dipeshtech/ms_uc3_autism/blob/master/images/variable_importance_kmeans.png)

##### K-Means Summary:
![Confusion Matrix](https://github.com/dipeshtech/ms_uc3_autism/blob/master/images/kmeans_summ.png)

#### Models Trained: 
* K-Means Clustering

### Part 3: Feature Analysis to recommend right service providers based on their expertise.

* Calculates the success_percentage based on the success of specific trials over time.
* ![ReadMe](https://github.com/dipeshtech/ms_uc3_autism/README_providers.md)


### Future Work:
* Extension of part3 to incorporate in the Machine Learning model for Provider Effectiveness
* 