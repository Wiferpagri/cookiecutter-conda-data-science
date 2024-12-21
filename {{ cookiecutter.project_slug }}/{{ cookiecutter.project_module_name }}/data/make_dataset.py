import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, make_scorer
from sklearn.feature_selection import RFECV

def preprocessing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler_type: str = 'standard',
    encoder_type: dict = None
) -> pd.DataFrame:
    """
    Preprocesses the train and test DataFrames by scaling numerical features and encoding categorical features.
    The function allows for dynamic selection of scalers for numerical features and encoders for categorical features.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        The training dataset containing both numerical and categorical features.
    
    test_df : pd.DataFrame
        The testing dataset containing both numerical and categorical features.
    
    scaler_type : str, optional (default='standard')
        The type of scaler to apply to numerical features. Supported options are:
        - 'standard': StandardScaler (default) - standardizes features by removing the mean and scaling to unit variance.
        - 'minmax': MinMaxScaler - scales features to a range between 0 and 1.
        - 'robust': RobustScaler - scales features using the median and interquartile range, less sensitive to outliers.
        - 'power': PowerTransformer - applies a power transformation to make data more Gaussian-like.
    
    encoder_type : dict, optional (default=None)
        A dictionary specifying the encoding method for each categorical column. If None, all categorical features are 
        encoded using OneHotEncoder by default. The supported encodings for categorical columns are:
        - 'label': LabelEncoder - encodes each label with a unique integer (applies to a single column).
        - 'ordinal': OrdinalEncoder - encodes categories with ordinal values (in ascending order).
        - 'onehot': OneHotEncoder - encodes categorical features as one-hot arrays (used by default).
        Example: 
        encoder_type = {
            'col_categorical_1': 'onehot', 
            'col_categorical_2': 'ordinal', 
            'col_categorical_3': 'label'
        }

    Returns:
    --------
    train_df_preprocessed : pd.DataFrame
        The preprocessed training dataset with scaled numerical features and encoded categorical features.

    test_df_preprocessed : pd.DataFrame
        The preprocessed testing dataset with scaled numerical features and encoded categorical features.
    
    Notes:
    ------
    - Numerical features are scaled according to the specified `scaler_type`.
    - Categorical features are encoded according to the specified `encoder_type` for each column.
    - If no specific encoding is provided for a categorical column, OneHotEncoder with drop='first' is used by default.
    - The function converts the transformed arrays back into DataFrames with appropriate column names.

    Example usage:
    --------------
    encoder_dict = {
        'col_1': 'onehot',
        'col_2': 'ordinal',
        'col_3': 'label'
    }
    train_df_preprocessed, test_df_preprocessed = preprocessing(train_df, test_df, scaler_type='minmax', encoder_type=encoder_dict)
    """
    num_feat = train_df.select_dtypes(include=['number']).columns
    cat_feat = train_df.select_dtypes(include=['object', 'category']).columns
    
    # Selecting scaler
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler == 'power':
        scaler = PowerTransformer() # Normal-like transform
    else:
        scaler = StandardScaler() # Standardization by default
    
    transformers = [('num', scaler, num_feat)]
    
    # Encoding categorical columns
    for cat_col in cat_feat:
        # If the user chosen different encoding for each column
        if encoder_type and cat_col in encoder_type:
            if encoder_type[cat_col] == 'label':
                encoder = LabelEncoder()
            elif encoder_type[cat_col] == 'ordinal':
                encoder = OrdinalEncoder()
            else:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
        else:
            encoder = OneHotEncoder(sparse_output=False, drop='first')
    
        # Adding transformer of the categorical column
        transformers.append(f'cat_{cat_col}', encoder, [cat_col])
    
    # Preprocessing
    preprocessor = ColumnTransformer(transformers=transformers)
    
    train_preprocessed = preprocessor.fit_transform(train_df)
    test_preprocessed = preprocessor.transform(test_df)
    
    # Converting them to DataFrames
    # Obtaining numerical column names
    num_col_names = num_feat.tolist()
    
    # Obtaining categorical column names
    cat_col_names = []
    for cat_col in cat_feat:
        if encoder_type and encoder_type.get(cat_col) == 'onehot':
            cat_col_names.extend(preprocessor.named_transformers_[f'cat_{cat_col}'].get_feature_names_out([cat_col]).tolist())
        else:
            cat_col_names.append(cat_col)
        
    all_col_names = num_col_names + cat_col_names
    
    train_df_preprocessed = pd.DataFrame(train_preprocessed, columns=all_col_names)
    test_df_preprocessed = pd.DataFrame(test_preprocessed, columns=all_col_names)
    
    return train_df_preprocessed, test_df_preprocessed

def rfe_dimensionality_reduction(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series | np.array,
    estimator,
    scorer: str,
    feautre_importance = True | bool,
    step = 1 | int,
    test_size = 0.2 | float,
    min_features_to_select = 1 | int,
    cv = 5 | int,

)  -> pd.DataFrame:
    """Perform dimensionality reduction using Recursive Feature Elimination (RFE) or feature importance.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix containing the independent variables.
    y : pd.DataFrame | pd.Series | np.array
        Target variable corresponding to the feature matrix.
    estimator : object
        A machine learning estimator (e.g., a classifier or regressor) that implements the `fit` method.
    scorer : str
        Scoring metric to evaluate the model's performance. Supported metrics include:
        - 'f1': F1 Score
        - 'roc_auc': Receiver Operating Characteristic Area Under the Curve
        - 'accuracy': Accuracy Score
        - 'recall': Recall Score
        - 'precision': Precision Score
        - 'mse': Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'r2': R-Squared Score
    feautre_importance : bool, optional (default=True)
        If True, use feature importance to reduce features. Otherwise, use RFE with cross-validation.
    step : int, optional (default=1)
        Number of features to remove at each iteration when using RFE.
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split when using feature importance.
    min_features_to_select : int, optional (default=1)
        Minimum number of features to select when using RFE.
    cv : int, optional (default=5)
        Number of cross-validation folds to use when using RFE.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the selected features.

    Raises:
    -------
    NotImplementedError
        If an unsupported scorer is provided.

    Notes:
    ------
    - When `feautre_importance` is True, feature selection is performed based on the feature importance 
      scores of the given estimator. The optimal subset of features is chosen to maximize the specified 
      scoring metric.
    - When `feautre_importance` is False, RFE with cross-validation is used to iteratively eliminate 
      features and select the best subset based on the scoring metric.

    Example:
    --------
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Create a sample dataset
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])

    # Initialize an estimator
    estimator = RandomForestClassifier(random_state=42)

    # Perform dimensionality reduction
    reduced_X = rfe_dimensionality_reduction(
        X=X,
        y=y,
        estimator=estimator,
        scorer='f1',
        feautre_importance=True
    )
    ```
    """
    
     # Define a dictionary to map scorer strings to their corresponding functions
    scorer_dict = {
        'f1': f1_score,
        'roc_auc': roc_auc_score,
        'accuracy': accuracy_score,
        'recall': recall_score,
        'precision': precision_score,
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'r2': r2_score
    }
    
    # Use the dictionary to get the loss function
    try:
        loss_func = scorer_dict[scorer]
    except KeyError:
        raise NotImplementedError(f"Scorer '{scorer}' is not implemented.")
    
    final_scorer = make_scorer(loss_func)
    
    scores = []
    selected_features = range(1, X.shape[1] + 1)
    
    if feautre_importance:
    # Calculating initial feature importance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if hasattr(estimator, 'eval_set'):
            estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        else:
            estimator.fit(X_train, y_train)
        
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': estimator.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        
        for i in selected_features:
            cols = list(feature_importance_df.iloc[:i, 0])
            
            if hasattr(estimator, 'eval_set'):
                estimator.fit(X_train[cols], y_train, eval_set=[(X_test[cols], y_test)])
            else:
                estimator.fit(X_train[cols], y_train)
            
            score = loss_func(y_test, estimator.predict(X_test[cols]))
            scores.append(score)
            
        final_features = list(feature_importance_df.iloc[:scores.index(max(scores)), 0])
        reduced_df = X[final_features]

    else:
        rfecv = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring=final_scorer,
            min_features_to_select=min_features_to_select
        )
        rfecv.fit(X, y)
        scores.extend(list(rfecv.grid_scores_)[::-1])
        final_features = list(X.columns[rfecv.support_])
        reduced_df = X[final_features]
    
    print(f'Optimal number of features: {len(final_features)}')
    print(f'Selected features: {final_features}')
    
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel(f"{scorer.capitalize()} Score")
    plt.plot(selected_features, scores)
    plt.show()
    
    return reduced_df
