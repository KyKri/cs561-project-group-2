# cs561-project-group-2
Repo for CS561 project group 2: Amazon Product Review Sentiment Analysis

## Tasks

Data
- Trim down to 50k samples
- Limit to positive and negative only (for classical ML models - makes it easier)
- Preprocess (class standard is fine)
- Train/validation/test split
    - Recommend 80/10/10 split respectively
> Note: For quality of comparison we all need to train, validate and test on the same data

Traditional ML Models
- SVM
- Logistic Regression
- Random Forest (Maybe)

LLM
- API calls
- Prompt engineering
- Output cleanup
> Note: This can get expensive, so this may not be run on the same full dataset as the traditional ML models,
> perhaps we take the first 10/1/1 % respectively from the traditional ML models for comparison to them

Evaluation
- Each model will need the same metrics run:
    - Accuracy
    - Precision
    - Recall
    - F1 score
- Graphs with train, validation and test loss

Summary/Writing
- Short 3-5 page google doc
    - What models we trained
    - What worked best
    - Description of dataset preprocessing
    - LLM pipeline description
- Slide deck based on google doc
