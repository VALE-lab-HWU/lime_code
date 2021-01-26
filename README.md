# lime_code

Repository for example code and use of the Lime explainer ([source](https://github.com/marcotcr/lime/))

## Data

The data aren't furnished. Any type of data can be run with Lime.

The different functions provided in the different helper files are able to read pickle files. They are also able to read multiple files and concat the data together, and read those files in separate folder.

The architecture used for testing was:

```
- data
|-------- all_patient.pickle
|-------- patient1
|         |-------file.pickle
|-------- patient2
|         |-------file.pickle
...
```

With the file `all_patient` being the concatenation of all the patient. It is made so we don't have to compute the concatenation every time, to save time and avoid the need of RAM (the concatenation duplicating all data in a new object, )

## Functions

The repository contain multiple helper files to ease the work. 

- `data_helper.py`: contain functions to load the data and extract feature (such as lifetime and intensity)
- `process_helper.py`: contain functions to process the data (such as rgb/gray conversion or scaling)
- `helper.py`: contain functions to do machine learning stuff (such as cross validation, printing confusion matrix or building sklearn pipeline)
- `model_helper.py`: contain functions to build and train model
- `lime_helper.py`: contain a few functions to help with lime visualization
