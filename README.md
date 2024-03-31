# Auto-MPG-Detection
A mpg detection system based on the methodology : Cross-Industry Standard Process for Data Mining

**Project Structure**
The project is organized following industry standards for directory structure and utilizes design patterns to ensure modularity, scalability, and maintainability. The structure comprises the following components:

Data/
  mpg.csv: Raw dataset containing information about miles per gallon.
  processed_data.csv: Processed dataset after data preprocessing steps.
src/
  data_processing/
    data_loader.py: Module for loading the dataset.
    data_pre_processing.py: Module for data cleaning and preprocessing.
  dimensionality_reduction/
    plots/
      2d_projection_pca.png: Visualization of 2D projection using PCA.
      cumulative_explained_variance.png: Cumulative explained variance plot for PCA.
      pca_notebook.ipynb: Jupyter notebook demonstrating PCA.
      pca.py: Module for performing Principal Component Analysis.
  hot_code_encoding/
    plots/
      cylinders_distribution.png: Distribution of cylinders feature.
      mpg_distribution.png: Distribution of mpg feature.
      origin_distribution.png: Distribution of origin feature.
    hot_code_encoding.py: Module for one-hot encoding categorical variables.
  modeling/
    evaluation.py: Module for model evaluation.
    hyperparameter_Tuning.py: Module for hyperparameter tuning.
    modeling.py: Module for building regression models.
  Testing/
	  unit_testing/
		  unit_testing_data_preprocessing.py
		  unit_testing_hotcodeencoding.py
		  unit_testing_featureselection.py
		  Unit_testing_modeling.py
		  Unit_testing_evaluation.py		
		  Unit_testing_hyperparametertuning.py
		  integration_testing.py
  Visualization/
    data_visualization.png: Visualization of the dataset.
    data_visualization.py: Module for data visualization.
    feature_Selection.py: Module for feature selection.
    Recursive_Feature_Elimination.py: Module for Recursive Feature Elimination.
    Select_K_Best.py: Module for SelectKBest feature selection.
    L1_Regularization.py: Module for L1 Regularization.
    Tree-based_methods.py: Module for feature importance using decision trees or random forests.
    VarianceThreshold.py: Module for variance thresholding.
    Mutual_Information.py: Module for mutual information-based feature selection.
    Sequential_Feature_Selection.py: Module for Sequential Feature Selection.
  trained_models/
    Directory containing joblib files of trained models- 
    1. Linear Regression
    2. Ridge Regression
    3. Lasso Regression
    4. Decision Tree Regressor
    5. Random Forest Regressor
    6. Gradient Boosting Regressor
    7. XGBoost Regressor
    8. Support Vector Regressor (SVR)
    9. K-Nearest Neighbors Regressor (KNN)
    10. Neural Network Regressor
    11. Elastic Net
    12. Bayesian Ridge Regression
    13. Huber Regressor
    14. Isotonic Regression
    15. Gaussian Process Regressor
    16. CatBoost Regressor
    17. LightGBM Regressor
    18. Elastic NetCV
    19. LGBM Regressor
    20. AdaBoost Regressor
