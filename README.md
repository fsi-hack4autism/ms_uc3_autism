# Use Case 3 - Application of Machine Learning to ABA Data

This use case analyzes anonymized therapy data from a repository of Applied Behavior Analysis sessions and cases to draw insights which can be used by individual families to help understand the effectiveness of their programs. 

## Technologies Used

### Languages

1. Python
2. SQL

### Platforms

1. Azure Databricks

### Libraries Used

1. sklearn
2. matplotlib
3. pandas
4. numpy
5. databricks MLFlow

## Deliverables Covered

1. Evaluation of 80% as a set goal for indication of future successes
2. Evaluation of the effect of treatment intensity on trial outcome
3. Analysis on the correlation of therapist/author based descriptors with outcomes

## Data Overview

![Data Overview](images/data.png)

## Part-2: Evaluation of the effect of treatment intensity on trial outcome

### Data Cleaning and Pre-processing

1. Missing values are marked as 0
2. Conversion of string columns to datetime for fields containing 'Date'
3. Conversion of categorical fields to encoded ones using Pandas Label Encoding

### Feature Engineering

1. **sessionCount_byGoal_byMonthYear** : Count of sessions graphed (TrialGroup) per month per year 
2. **sessionCount_byGoal_byDayYear** : Count of sessions graphed (TrialGroup) per day per year 
3. **sessionCount_byGoal_byWeekYear** : Count of sessions graphed (TrialGroup) per week per year
4. **Gender_Encoded** : Demographic Data
5. **Age** : Demographic Data, calculated with the difference of TrialDataDate and BirthYear
6. **TrialTargetId_Encoded** : IDs of the target Goals
7. **GoalDomain_Encoded** : Encoded goal domain value of Adaptive, Communication and Language
8. **goalAssessment_encoded** : Encoded goal ABA assessment
9. **encodedTrialPhase** : Encoded Trial Phase (Baseline' -1,'Intervention' -2,'Generalization' -3,'Maintenance'-4)
10. **goalForced_80Percent** : The target value of outcome to 80% set goal

### Models Used

1. Random Forest
2. Decision Tree

### Feature Importance

![Feature Importance](images/feature_importance.png)

### Confusion Matrix

![Confusion Matrix](images/confusion.png)

### ROC Curve

![ROC Curve](images/roc.png)

### Inferences

1. The feature importance from the Supervised Models show that Session Count/ Treatment Intensity descriptors
are more useful in predicting a successful outcome than the demographic features.

2. The Goal ID is also a useful feature but has limitations due to high cardinality.

3. Age is a more useful predictor for the outcome of a trial than the goal domain or trial phase.

### Multiple therapist changes decreases success rates

![Multiple therapist changes decreases success rates](images/success_ther.png)
