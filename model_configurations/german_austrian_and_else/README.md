
# this model classifies german/austrian names from other nationalities


## | nationalities the model can classify:
```json
{ "else": 0, "german_austrian": 1 }
```

## | performance and result metrics:
 - accuracy: 88.1%
 - confusion matrix: <br/> ![confusion_matrix](./confusion_matrix.png)
 - sensitivity/recall:
    - german: 0.964
    - else: 0.798
