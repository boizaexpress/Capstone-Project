# Multi-model Tuning Results

## logreg
- Best CV score: 0.9583
- Test accuracy: 1.0000
```
{
  "clf__penalty": "l2",
  "clf__C": 10
}
```

### Classification report
```
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000        10
           1     1.0000    1.0000    1.0000        10
           2     1.0000    1.0000    1.0000        10

    accuracy                         1.0000        30
   macro avg     1.0000    1.0000    1.0000        30
weighted avg     1.0000    1.0000    1.0000        30

```
![Confusion matrix](c:\Users\user\Downloads\Capstone project\tuning_confusion_logreg.png)

## rf
- Best CV score: 0.9667
- Test accuracy: 0.9333
```
{
  "clf__n_estimators": 50,
  "clf__max_depth": 10
}
```

### Classification report
```
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000        10
           1     0.9000    0.9000    0.9000        10
           2     0.9000    0.9000    0.9000        10

    accuracy                         0.9333        30
   macro avg     0.9333    0.9333    0.9333        30
weighted avg     0.9333    0.9333    0.9333        30

```
![Confusion matrix](c:\Users\user\Downloads\Capstone project\tuning_confusion_rf.png)

## svc
- Best CV score: 0.9667
- Test accuracy: 0.9667
```
{
  "clf__gamma": "scale",
  "clf__C": 10
}
```

### Classification report
```
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000        10
           1     1.0000    0.9000    0.9474        10
           2     0.9091    1.0000    0.9524        10

    accuracy                         0.9667        30
   macro avg     0.9697    0.9667    0.9666        30
weighted avg     0.9697    0.9667    0.9666        30

```
![Confusion matrix](c:\Users\user\Downloads\Capstone project\tuning_confusion_svc.png)

