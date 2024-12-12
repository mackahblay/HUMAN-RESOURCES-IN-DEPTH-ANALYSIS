# HR Data Analysis Project

## Overview

This project provides a comprehensive analysis of human resources data for a company, exploring various aspects of employee demographics, performance, compensation, and organizational characteristics. The analysis is conducted using Python, with a focus on data manipulation, visualization, and advanced machine learning techniques for performance prediction.

## Project Structure

The project involves in-depth exploration of a human resources dataset, analyzing key metrics such as:
- Employee demographics
- Hiring and termination trends
- Salary distribution
- Performance metrics
- Departmental insights
- Recruitment sources
- Machine learning-based performance prediction

## Key Technologies

- **Programming Language**: Python
- **Data Manipulation**: Pandas
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Key Libraries**: 
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost


## Machine Learning Performance Prediction

### Feature Selection and Preprocessing

The project employs advanced feature selection techniques to identify the most relevant predictors of employee performance:

1. **Feature Selection Strategy**
   - Used SelectKBest with chi-square and f-classification methods
   - Identified top features: 
     - Salary
     - Position
     - Manager Name
     - Special Projects Count

2. **Data Preparation**
   - Split dataset into training and test sets (80% training, 20% testing)
   - Applied StandardScaler for feature scaling
   - Prepared feature matrix (X) and target variable (performance score)

### Model Development and Evaluation

Implemented and compared multiple machine learning models to predict employee performance:

1. **Logistic Regression**
   - Accuracy: 74.60%
   - F1 Score: 0.638

2. **K-Nearest Neighbors (KNN)**
   - Best Parameters:
     - Leaf Size: 10
     - Neighbors: 12
     - Weights: Uniform
   - Accuracy: 74.60%
   - F1 Score: 0.638

3. **Support Vector Machine (SVM)**
   - Best Parameters:
     - Kernel: Polynomial
     - Complexity (C): 1
     - Degree: 3
   - Accuracy: 74.60%
   - F1 Score: 0.661

4. **Random Forest**
   - Best Parameters:
     - Estimators: 50
     - Max Depth: 5
     - Min Samples Split: 4
   - Accuracy: 74.60%
   - F1 Score: 0.638

5. **XGBoost**
   - Best Parameters:
     - Estimators: 100
     - Learning Rate: 0.01
     - Max Depth: None
     - Min Child Weight: 5
   - Accuracy: 74.60%
   - F1 Score: 0.638

### Performance Prediction Example

Demonstrated the ability to predict performance for a hypothetical new employee:
- Salary: $75,000
- Position: Software Engineer
- Manager: Michael Albert
- Special Projects: 0
- **Predicted Performance**: Exceeding Expectations

### Model Insights

- Consistent accuracy across different algorithms (74.60%)
- SVM showed slightly better F1 score (0.661)
- Key predictors include salary, position, and special project involvement

## Potential Future Enhancements

- Develop more sophisticated feature engineering techniques
- Experiment with ensemble methods
- Incorporate more complex performance indicators
- Create interactive dashboards for workforce analytics
- Expand machine learning capabilities for predictive HR insights

## Technical Implementation Notes

### Feature Selection
```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(score_func=chi2, k=4)
x_new = selector.fit_transform(x, y)
```

### Model Training
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

svm_model = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
svm_model.fit(x_train_scaled, y_train)
```

## Limitations and Considerations

- Performance predictions are probabilistic
- Model accuracy may vary with new or unseen data
- Requires periodic retraining and validation
- Encoding may introduce subtle biases

## Contributing and Extending the Project

Interested in contributing? Consider:
- Implementing advanced machine learning techniques
- Adding more sophisticated performance metrics
- Creating comprehensive visualization dashboards
- Developing predictive retention models

## License

**Creative Commons**

## Contact

melkages@gmail.com
