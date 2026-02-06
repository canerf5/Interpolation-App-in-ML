# Interpolation-App-in-ML
Applications of Interpolation in Machine Learning - Telco Customer Churn Analysis

Technical Overview: Pseudo-Time Series Imputation Framework
This project explores the mathematical applicability of interpolation techniques on cross-sectional datasets by leveraging a Pseudo-Time Series transformation. The core research objective is to assess the performance of deterministic imputation methods against traditional stochastic and statistical approaches in the context of binary classification.

Methodology: The Pseudo-Time Series Approach


The fundamental challenge in applying interpolation to cross-sectional data is the lack of a natural monotonic ordering. This study proposes a transformation based on internal feature covariance:
Correlation-Based Ordering: A high-correlation auxiliary variable, Tenure (r \approx 0.825), is utilized as the independent ordinal axis (X).

Functional Mapping: The target variable, TotalCharges, is mapped as a function of the ordering variable: $TotalCharges \approx f(Tenure) + \epsilon$.

Deterministic Imputation: Once the sequential topology is established, missing values (simulated at a 10% MCAR rate) are filled using:

  Linear Spline Interpolation: Piecewise linear approximation.
  
  Polynomial Interpolation (Order 2): Global trend fitting via Lagrange forms, limited to the second degree to prevent Runge's phenomenon.
  
  Cubic Spline Interpolation: $C^2$ continuous piecewise third-degree polynomials ensuring curvature equality at nodal points.

Comparative Performance Analysis


The methodology was validated using 10-fold cross-validation across eight distinct machine learning architectures, including Gradient Boosting Machines (LightGBM, CatBoost, XGBoost) and parametric models.
  
  Noise Filtering & Smoothing: Polynomial Interpolation achieved the global maximum performance with Logistic Regression ($ROC\_AUC: 0.8463$), notably outperforming the model trained on the original, complete dataset. This suggests that low-degree interpolation acts as a noise-reduction filter, enhancing model generalization.
  
  Variance Preservation: Mean Imputation demonstrated consistently inferior results due to artificial variance suppression ($\sigma^2 \rightarrow 0$), which diminishes Information Gain in tree-based models. In contrast, Spline and KNN methods effectively preserved the data manifold.
  
  Model-Method Compatibility: A significant alignment was observed between the mathematical structure of the imputation and the learner:
    
      Global Interpolators: Performed optimally with global decision boundary models (e.g., Polynomial with        Logistic Regression).
    
      Local Imputations: Performed better with local learners (e.g., KNN with LightGBM).
  
  Computational Efficiency: Cubic Spline interpolation provides a Pareto-optimal trade-off for industrial applications. It delivers accuracy competitive with KNN but at a significantly lower computational cost, scaling nearly linearly $O(N)$ compared to the quadratic complexity $O(N^2)$ of distance-based methods.



Engineering Implications


The findings confirm that correlation-focused interpolation is a viable and often superior strategy for missing data imputation in sequential cross-sectional environments, particularly for high-stakes customer analytics in telecommunications and finance.
