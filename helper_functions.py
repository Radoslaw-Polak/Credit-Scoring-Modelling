import pandas as pd
import numpy as np
import seaborn as sns                                                                                                       
from matplotlib import pyplot as plt

def numeric_describe(df):
    pd.set_option('display.float_format', '{:.2f}'.format)
    numerical_data = df.select_dtypes(["int32", "float32", "int64", "float64"]).dropna()
    numerical_data_describe = numerical_data.describe()
    numerical_data_describe.loc['unique'] = numerical_data.nunique()
    numerical_data_describe.loc['var_coeff [%]'] = 100.0 * numerical_data_describe.loc['std'] / numerical_data_describe.loc['mean']
    numerical_data_describe.loc['median'] = numerical_data.median()
    numerical_data_describe.loc['skew'] = numerical_data.skew()
    numerical_data_describe.loc['kurtosis'] = numerical_data.kurtosis()
    return numerical_data, numerical_data_describe
    
"""Adjusting number of rows for subplot"""
def subplot_shape(df, subplot_cols=3):
    df_ncols = df.columns.size
    if df_ncols % subplot_cols == 0:
        return (int(df_ncols / subplot_cols), subplot_cols)
    else:
        return (df_ncols // subplot_cols + 1, subplot_cols)

"""Function to draw distributions for variables (columns)"""
def draw_distribution(
        df, 
        subplot_size, 
        subplot_cols=3, 
        barplot_max_cols=20, 
        label_rot_for_categorical=0, 
        top_n_freq=5, 
        plot_color='tab:blue'
    ):
    plot_shape = subplot_shape(df, subplot_cols=subplot_cols)
    fig, axes = plt.subplots(plot_shape[0], plot_shape[1], figsize=subplot_size)
    axes = axes.flatten()
    num_cols = df.shape[1]
    distribution_type = {}
    keys = df.columns.values
    distribution_type = distribution_type.fromkeys(keys)

    for iter, (ax, col) in enumerate(zip(axes, df.columns)):
        top_frequent = ''
        data = df[col].dropna().reset_index(drop=True)
        val_counts = data.value_counts()
        grid_axis = 'y'
        if data.dtype == 'object':
            if len(val_counts.index) <= barplot_max_cols / 2:
                sns.barplot(
                    x=val_counts.index, 
                    y=val_counts.values, 
                    ax=ax, 
                    color=plot_color,
                    edgecolor='black'
                )
            elif len(val_counts.index) - barplot_max_cols <= 50 - barplot_max_cols:
                sns.barplot(
                    x=val_counts.values,
                    y=val_counts.index,  
                    ax=ax, 
                    color=plot_color,
                    orient='h',
                    height=0.5,
                    edgecolor='black'
                )
                grid_axis = 'x'
            else:
                top_frequent = f" - TOP {top_n_freq} categories"
                sns.barplot(
                    x=val_counts.index[0:top_n_freq], 
                    y=val_counts.values[0:top_n_freq], 
                    ax=ax, 
                    color=plot_color,
                    edgecolor='black'
                )
            ax.tick_params(axis='x', labelrotation=label_rot_for_categorical)
            plot_type = 'barplot'
        else:  # distributions_for_numerics:
            if len(val_counts.index) <= barplot_max_cols:
                sns.barplot(
                    x=val_counts.index, 
                    y=val_counts.values, 
                    ax=ax, 
                    color=plot_color,
                    edgecolor='black'
                )
                plot_type = 'barplot'
            else:
                sns.histplot(
                    data=data, 
                    bins='auto', 
                    ax=ax, 
                    color=plot_color,
                    edgecolor='black'
                )
                plot_type = 'histogram'
        ax.set_title(f"{col}{top_frequent}")
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_axisbelow(True)
        ax.grid(linewidth=0.8, axis=grid_axis)
        distribution_type[col] = plot_type
        # Counter for drawing charts process
        print(f'\rDrawing distribution plots [{iter+1}/{num_cols}]', end='', flush=True)
    
    for i, item in enumerate(distribution_type.items()):
        if i == 0:
            print('\n')
        print(item)
    plt.tight_layout()
    plt.show()

"""Function to determine number of outlier values in a dataframe column"""
def number_of_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return len(outliers)

"""Function for drawing boxplots"""
def draw_boxplots(
        df, 
        subplot_size=(16, 10), 
        subplot_cols=3, 
        width=0.3, 
        plot_color='tab:blue'
    ):
    plot_shape = subplot_shape(df, subplot_cols=subplot_cols)
    fig, axes = plt.subplots(plot_shape[0], plot_shape[1], figsize=subplot_size)
    axes = axes.flatten()
    num_cols = df.shape[1]
    num_outliers = []
    
    for iter, (ax, col) in enumerate( zip(axes, df.columns) ):
        sns.boxplot(
            data=df[col], 
            ax=ax, 
            width=width,
            color=plot_color
        )
        num_outliers.append( number_of_outliers(df[col]) )
        ax.set_title(col)
        ax.set_ylabel('')
        ax.grid(linewidth=0.8)
        print(f'\rDrawing boxplots [{iter+1}/{num_cols}]', end='', flush=True)
    print(f'\n{num_outliers}')
    plt.tight_layout()
    plt.show()

"""function for handling outliers in a dataframe"""
def handle_outliers(
        df, 
        threshold=1.5, 
        remove=True, 
        replace_val=''
    ):
    df_cleaned = df.copy()
    df_numeric = df_cleaned.select_dtypes(include=['int32', 'float32', 'int64', 'float64'])
    
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1

    # data serieses for lower and upper bounds for each numeric column
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    if remove == True:
        # condition for numeric columns for handling outliers
        mask = ~((df_numeric < lower_bound) | (df_numeric > upper_bound)).any(axis=1)
        df_cleaned_no_outliers = df_cleaned[mask]
        print(f"Customer dataframe INPUT size: {len(df_cleaned)}")
        print(f"Deleted rows containing outliers: {len(mask[mask==False])}")
        print(f"Customer dataframe OUTPUT size: {len(df_cleaned_no_outliers)}")
        return df_cleaned_no_outliers

    else:
        for col in df_numeric.columns:
            col_dtype = df_cleaned[col].dtype
            if replace_val == 'mean':
                replacement_val = df_cleaned[col].mean()
            elif replace_val == 'median':
                replacement_val = df_cleaned[col].median()

            replacement = col_dtype.type(replacement_val) # setting type of the replacement value corresponding to given column
            outlier_mask_col = ((df_cleaned[col] < lower_bound[col]) | (df_cleaned[col] > upper_bound[col]))
            df_cleaned.loc[outlier_mask_col, col] = replacement

        return df_cleaned

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

"""Creating linear models for every independent variable between other independent variables"""
def VIF(numerical_data):    
    X = numerical_data
    X = add_constant(X)

    VIFs = pd.DataFrame(
      data=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
      index=X.columns,
      columns=['VIF']
    )
    # Returning VIF values sorted in descending order
    return VIFs[VIFs.index != 'const'].sort_values(by='VIF', ascending=False)

"""
Checking linearity of logit for continuous variables, if p-value of the regression is less than 0.05 then the relationship is non-linear
In the end didn't use it, it would be more useful for logistic model from statsmodels in context of coefficients interpretation, significance 
and confidence intervals, but for sklearn logistic regression it is not that useful, so I decided to use it only for EDA purposes and not for feature engineering
In the case of prediciton based analysis the scikit-learn logistic regression model is much more commonly used, also the fact that I have storngly imbalanced data
and I am using class_weight='balanced' parameter in logistic regression model, so the linearity of logit is not that important for the model performance, 
but it is still good to check it for EDA purposes and to have better understanding of the data and relationships between features and target variable.
"""
def check_logit_linearity(X_cont, y, n_bins=10, target_col='target', plot=False):
    """
    Sprawdza liniowość logitów dla zmiennych ciągłych.
    Zwraca słownik z wykresami i flagami naruszenia założenia.
    """
    df = pd.concat([X_cont, y], axis=1)
    results = {}
    
    for col in X_cont.columns:
        # Bins
        df_temp = df[[col, target_col]].copy()
        df_temp[f'{col}_bin'] = pd.qcut(df_temp[col], q=n_bins, duplicates='drop')
        
        # Calculating empirical log-odds in each bin
        bin_stats = df_temp.groupby(f'{col}_bin')[target_col].agg(['mean', 'count'])
        bin_stats['p'] = bin_stats['mean']
        # We avoid log(0) i log(1)
        eps = 1e-5
        bin_stats['log_odds'] = np.log((bin_stats['p'] + eps) / (1 - bin_stats['p'] + eps))
        bin_stats['bin_center'] = bin_stats.index.map(lambda x: x.mid)
        
        # basic linear regression: log_odds ~ bin_center
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            bin_stats['bin_center'], bin_stats['log_odds']
        )
        
        is_linear = p_value < 0.05  # If p < 0.05 then the relationship is non-linear
        
        results[col] = {
            'p_value_lin': p_value,
            'is_linear': not is_linear,
            'bin_stats': bin_stats
        }
        
        # Plot
        if plot:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(
                x=bin_stats['bin_center'], 
                y=bin_stats['log_odds'], 
                size=bin_stats['count'], 
                sizes=(50, 300), 
                alpha=0.7
            )
            sns.regplot(
                x=bin_stats['bin_center'], 
                y=bin_stats['log_odds'], 
                scatter=False, 
                color='red', 
                ci=None
            )
            plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
            plt.title(f'{col} vs Log-Odds (p={p_value:.3f})')
            plt.xlabel(col)
            plt.ylabel('Empirical Log-Odds')
            plt.grid(True, alpha=0.3)
            plt.show()
        
    return results

from sklearn.preprocessing import OneHotEncoder

"""Encodes categorical columns in the DataFrame using OneHotEncoder."""
def encode_categoric_data(df, ohe=None):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    encoding_values = df[categorical_columns].nunique().values
    # Fitting new OneHotEncoder if not provided, otherwise using the provided one (for test data encoding)
    if ohe is None:
        # drop='first' to avoid dummy variable trap, dropping first category in each categorical column and 
        # handle_unknown='ignore' to avoid errors when test data contains categories not present in training data
        ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        # Fit and transform the categorical columns (for training data encoding)
        one_hot_encoded = ohe.fit_transform(df[categorical_columns])
    else:
        # Transform the categorical columns using the provided OneHotEncoder (for test data encoding)
        one_hot_encoded = ohe.transform(df[categorical_columns])

    # Create a DataFrame with the one-hot encoded columns
    one_hot_df = pd.DataFrame(
        one_hot_encoded, 
        index=df.index,
        columns=ohe.get_feature_names_out(categorical_columns)
    ).astype(int)
    
    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    df_encoded = pd.concat([df, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(columns=categorical_columns)
     
    # Categories for each categorical column are in a list of np.arrays, dropped first category for each feature (in encoding)  
    # so extracting it as a reference category
    ref_cat_values = [ref_cat[0] for ref_cat in ohe.categories_]
    # Reference category values for each categorical column in dictionary, 
    ref_categories = dict( zip(categorical_columns, ref_cat_values) )

    for i, (cat_col, encoded_vals) in enumerate( zip(categorical_columns, encoding_values) ):
        print(f"{i+1}) {cat_col} - encoded {encoded_vals} categories [reference category: '{ref_categories[cat_col]}']")
    
    return df_encoded, ohe


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score, fbeta_score, precision_recall_curve, PrecisionRecallDisplay, classification_report, average_precision_score

"""Obtaining optimal decision threshold for binary classification based on maximizing Fbeta-score, for beta > 1 more weight is given to recall,
for beta < 1 more weight is given to precision, for beta = 1 it is equivalent to maximizing F1-score"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(
        y_true, 
        y_proba, 
        beta=None, 
        plot=True,
        figsize=(6, 4.5)
    ):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    if beta is None:
        # Set beta as the square root of the class frequency ratio
        if hasattr(y_true, 'value_counts'):
            class_counts = y_true.value_counts()
            class_freq_ratio = class_counts.loc[0] / class_counts.loc[1]
        else:
            unique, counts = np.unique(y_true, return_counts=True)
            class_counts = dict(zip(unique, counts))
            class_freq_ratio = class_counts[0] / class_counts[1]
        beta = np.sqrt(class_freq_ratio)

    elif beta <= 0:
        raise ValueError(f'Beta parameter should be a positive value, but got {beta} instead.')
    
    elif not isinstance(beta, (int, float)):
        raise TypeError(f'Beta parameter should be a numeric value, but got {type(beta)} instead.')
    
    beta2 = beta**2
    
    # Zero division handling for F-beta score calculation, if denominator is zero set Fbeta-score to zero for that threshold
    denominator = beta2 * precisions[:-1] + recalls[:-1]
    fbeta_scores = np.divide(
        (1 + beta2) * (precisions[:-1] * recalls[:-1]),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0
    )
    
    # Checking edge case where all F-beta scores are zero or could not be computed due to zero division, in that case raise an error
    if len(fbeta_scores) == 0 or np.all(fbeta_scores == 0):
        raise ValueError("Could not compute valid F-beta scores")
    
    optimal_idx = np.argmax(fbeta_scores)
    best_threshold = thresholds[optimal_idx] 
    best_fbeta_score = fbeta_scores[optimal_idx]
    
    # Plotting
    if plot:
      plt.figure(figsize=figsize)
      plt.plot(
          thresholds, 
          fbeta_scores, 
          color='black', 
          label=f'Fbeta-score [beta = {beta:.2f}]'
      )
      plt.scatter(
          x=best_threshold, 
          y=best_fbeta_score,
          color='red',
          marker='o',
          label=f'Best threshold [{best_threshold:.3f}]'
      )
      plt.xlabel('Threshold', fontsize=12)
      plt.ylabel(f'F{beta:.2f}-score', fontsize=12)
      plt.title("Obtaining best threshold", fontsize=14)
      plt.grid(linewidth=0.4, alpha=0.7)
      plt.legend(loc='center right')
      plt.tight_layout()
      plt.show()
      
    return best_threshold, best_fbeta_score, beta

from sklearn.model_selection import StratifiedKFold

def model_cross_validation(
        model,
        X,
        y,
        k_folds=5,
        fbeta_param=None,
        model_name='',
        random_state=68,
        shuffle=True,
        verbose=True,
        plot=False
    ):

    print(f"Cross validation - {model_name}")

    skf = StratifiedKFold(
        n_splits=k_folds,
        shuffle=shuffle,
        random_state=random_state
    )

    thresholds_per_fold = []
    best_scores_per_fold = []
    precisions_per_fold = []
    recalls_per_fold = []
    fbetas_per_fold = []
    accuracies_per_fold = []

    fold_idx = 1

    for train_idx, valid_idx in skf.split(X, y):

        print(f'\nFOLD {fold_idx}')
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        else:
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

        from sklearn.base import clone
        model_fold = clone(model)

        # training
        model_fold.fit(X_train, y_train)

        # probabilities prediciton
        y_proba = model_fold.predict_proba(X_valid)[:, 1]

        # obtaining best threshold and score for fold
        best_threshold, best_score, beta = find_optimal_threshold(
            y_true=y_valid,
            y_proba=y_proba,
            beta=fbeta_param,
            plot=plot
        )

        y_pred = np.where(y_proba > best_threshold, 1, 0)
        precision = precision_score(y_valid, y_pred)
        recall = recall_score(y_valid, y_pred)
        fbeta = fbeta_score(y_valid, y_pred, beta=beta)
        accuracy = accuracy_score(y_valid, y_pred)

        thresholds_per_fold.append(best_threshold)
        best_scores_per_fold.append(best_score)
        precisions_per_fold.append(precision)
        recalls_per_fold.append(recall)
        fbetas_per_fold.append(fbeta)
        accuracies_per_fold.append(accuracy)

        if verbose:
            print(f"Fold {fold_idx}: threshold={best_threshold:.3f}, score={best_score:.4f}")

        fold_idx += 1

    fbeta_colname = f"F{beta:.2f}-score"
    results = {
        "Fold": np.arange(1, k_folds + 1, 1),
        "Best score": best_scores_per_fold,
        "Threshold": thresholds_per_fold,
        "Precision": precisions_per_fold,
        "Recall": recalls_per_fold,
        fbeta_colname: fbetas_per_fold,
        "Accuracy": accuracies_per_fold
    }
    df_results = pd.DataFrame(results)

    # Aggregated results
    agg_results = {
      "avg": df_results[['Threshold', 'Precision', 'Recall', fbeta_colname, 'Accuracy']].mean(),
      "median": df_results[['Threshold', 'Precision', 'Recall', fbeta_colname, 'Accuracy']].median(),
      "std": df_results[['Threshold', 'Precision', 'Recall', fbeta_colname, 'Accuracy']].std()
    }
    # It is 1 row DataFrame
    df_agg_results = pd.DataFrame(
        agg_results,
        index=agg_results["avg"].index,
        columns=agg_results.keys()
    )
    
    return df_results, df_agg_results, beta

from string import capwords
"""Confusion matrices"""
def display_confusion_matrix(
        y1_true, 
        y1_pred, 
        y2_true, 
        y2_pred, 
        # title='', 
        model_name='',
        cmap='cividis', 
        compare=['Train', 'Validation'], 
        figsize=(15, 6)
    ):
    
    if not isinstance(model_name, str) or not isinstance(compare, list) or len(compare) != 2:
        raise ValueError("Model_name should be a string and compare should be a list of two strings (e.g., ['Train', 'Validation']).")
    
    cm1 = confusion_matrix(y_true=y1_true, y_pred=y1_pred)
    cm2 = confusion_matrix(y_true=y2_true, y_pred=y2_pred)

    cm1_disp = ConfusionMatrixDisplay(confusion_matrix=cm1)
    cm2_disp = ConfusionMatrixDisplay(confusion_matrix=cm2)
    cms_disp = [cm1_disp, cm2_disp]

    fig, axes = plt.subplots(1, 2, figsize=figsize) 
    axes = axes.flatten()
    model_name = capwords(model_name)

    for i, (ax, cm) in enumerate( zip(axes, cms_disp) ):
        cm.plot(ax=ax, cmap=cmap)
        cm_title = f'{model_name} - {compare[i]}' 
        ax.set_xlabel('Predicted target')
        ax.set_ylabel('True target')
        ax.set_title(cm_title)
        ax.invert_yaxis() # Invert y-axis for better readability
    fig.suptitle(t=f'Confusion Matrices - {model_name} | {compare[0]} vs {compare[1]}', fontsize=13)
    plt.tight_layout()
    plt.show()
    return cm1, cm2

"""Quality metrics report"""
def quality_metrics(
        y_true, 
        y_pred, 
        y_probs=None, 
        pos_class_label=1, 
        label='Test data',
        plot=True, 
        pr_curve_figsize=(6, 5), 
        pr_curve_title=''
    ):

    n_classes = len( np.unique(y_true) )
    print(f"{label}:")
    print(f"Accuracy: {accuracy_score(y_true=y_true, y_pred=y_pred):.3f}")
    
    # For multiclass classification, precision, recall and F1-score are calculated for each class"""
    old_format = pd.options.display.float_format
    pd.options.display.float_format = '{:.3f}'.format
    score_results = pd.DataFrame(
        classification_report(
            y_true=y_true, 
            y_pred=y_pred, 
            target_names=[str(i) for i in range(n_classes)], 
            digits=3,
            output_dict=True
        )
    ).round(decimals=3) 
    print(score_results)

    # For binary classification only - draw precision-recall curve
    if plot and n_classes == 2 and (y_probs != None).all():
        precisions, recalls, _ = precision_recall_curve(y_true=y_true, y_score=y_probs, pos_label=pos_class_label) # y_probs[:, 1]
        avg_precision_score = average_precision_score(y_true=y_true, y_score=y_probs)
        plt.figure(figsize=pr_curve_figsize)
        plt.plot(
            recalls, 
            precisions, 
            label=f'AP = {avg_precision_score:.4f}', 
            color='black'
        )
        plt.title(pr_curve_title)
        plt.xlabel('Recall', fontsize=12); plt.ylabel('Precision', fontsize=12)
        plt.legend(loc='center left', fontsize=12)
        plt.grid(linewidth=0.4)
        plt.tight_layout()
        plt.show()

    pd.options.display.float_format = old_format
    return score_results    

from sklearn.inspection import permutation_importance

def make_threshold_scorer(scoring='fbeta', threshold=0.5, beta=1):
    def scorer(model, X, y):
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        if scoring == 'fbeta':
            return fbeta_score(y, y_pred, beta=beta)
        elif scoring == 'recall':
            return recall_score(y, y_pred)
        
    return scorer

"""Plots feature importance calculated using permutation importance"""
def plot_feature_importances(
        model, 
        model_name, 
        X_data, 
        y_data, 
        n_reps=5, 
        scoring=None,   
        max_num_features=15, 
        n_jobs=-1, 
        figsize=(11, 6), 
        random_state=68
    ):
    # Getting features and their importances and sorting them by the importance value in descending order
    # returns sklearn.utils.Bunch object with importances_mean, importances_std and importances, can refer to importances_mean with 
    # dot '.' operator
    try:
        print(f'Calculating feature importances for {model_name.capitalize()} model ...')
    except TypeError as te:
        print(f'{te}: model_name parameter should be a string type but got {type(model_name)} instead.')

    if hasattr(X_data, "columns"):
        feature_names = X_data.columns
    else:
        feature_names = [f"feature_{i}" for i in range(X_data.shape[1])]

    importances = permutation_importance(
        estimator=model, 
        X=X_data, 
        y=y_data, 
        n_repeats=n_reps, 
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs
    ) 
    feature_importances_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances.importances_mean
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        y='Feature',
        x='Importance',
        data=feature_importances_df[:max_num_features],
        ax=ax,
        color='dodgerblue',
        edgecolor='black',
        orient='h'
    )
  
    # For readability, if label is negative still show it on the right side of the bar
    for container in ax.containers:
        for rect in container:
            value = rect.get_width()
            x = value if value >= 0 else 0
            y = rect.get_y() + rect.get_height() / 2

            ax.text(
                x=x + 0.001,
                y=y,
                s=f"{value:.4f}",
                va='center',
                ha='left',
                fontsize=10
            )
    plt.xlabel('Importance', fontsize=14); plt.ylabel('Feature', fontsize=14)
    plt.title(f'Feature Importances (Mean) - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

    # returning list of feature names sorted descending by their importance
    return feature_importances_df['Feature'].tolist()


import shap 
from sklearn.model_selection import train_test_split

"""Shapley values analysis and visualization for explainability"""
def plot_shap_values(
        model, 
        model_name, 
        X_train, 
        y_train,
        X_valid, 
        y_valid, 
        max_display=15, 
        sample_class_label=1, 
        threshold=0.5, 
        random_state=68,
        sample_size=0.05
    ):
    try:
        print(f'Calculating SHAP values for {model_name.capitalize()} model ...')
    except TypeError as te:
        print(f'{te}: model_name parameter should be a string type but got {type(model_name)} instead.')

    if len(X_valid) != len(y_valid):
      raise ValueError('X_valid and y_valid should contain same number of observations')

    model_name = model_name.lower()
    X_val, y_val = X_valid.copy(), y_valid.copy()

    # For speeding up the process of calculating SHAP values using only sample_size * 100 % of validation data
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=sample_size,
        stratify=y_train,
        random_state=random_state
    )
    X_train_sample = X_train_sample.reset_index(drop=True)
    y_train_sample = y_train_sample.reset_index(drop=True)
    
    X_val_sample, _, y_val_sample, _ = train_test_split(
        X_val,
        y_val,
        train_size=sample_size,
        stratify=y_val,
        random_state=random_state
    )
    X_val_sample = X_val_sample.reset_index(drop=True)
    y_val_sample = y_val_sample.reset_index(drop=True)

    if model_name == 'logistic regression':
        explainer = shap.Explainer( 
            model=model, 
            masker=X_train_sample, 
            seed=random_state
        )
        shap_values = explainer(X_val_sample)   
        print((shap_values, type(shap_values), shap_values.shape))

    elif model_name in ['xgboost', 'random forest']:
        explainer = shap.TreeExplainer(
            model=model, 
            approximate=True,
        )

        shap_values = explainer(X_val_sample)
        print((shap_values, type(shap_values), shap_values.shape))
        # For Random Forest shap_values has 3 dimensions where last dim corresponds to class so have to 
        # select specific class (in this case class 1 - default)
        shap_values = shap_values[:, :, 1] if model_name == 'random forest' else shap_values

    elif model_name == 'neural network':        
        masker = shap.maskers.Independent(X_train_sample)
        explainer = shap.Explainer(
            model=model.predict_proba,
            masker=masker,
            seed=random_state
        )

        shap_values = explainer(X_val_sample)
        print((shap_values, type(shap_values), shap_values.shape))
        # shap_values for class 1 (default case), for explainer with masker it returns list of arrays for each class
        shap_values = shap_values[:, :, 1]
 
    # getting random observation index where class_label = positive class (default = 1)
    np.random.seed(random_state)
    obs_idx = np.random.choice( np.where(y_val_sample == sample_class_label)[0], size=1 )[0]
    # using sample input for waterfall plot, transpose (T) for proper shape
    sample_input = pd.DataFrame(
      data=X_val_sample.iloc[obs_idx, :]
    ).T
    sample_output = y_val_sample.iloc[obs_idx]
    # get sample probability for positive class (1)
    sample_prob = model.predict_proba(sample_input)[0, 1] 
    sample_pred = np.where( sample_prob > threshold, 1, 0)

    shap.plots.beeswarm(
        shap_values, 
        max_display=max_display, 
        show=False
    )
    plt.title(f'SHAP Values (entire set) - {model_name.capitalize()}', fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"Sample pred: {sample_pred}")

    shap.plots.waterfall(
        shap_values[obs_idx], 
        max_display=max_display, 
        show=False
    )
    plt.title(
        f'SHAP Values (sample prediction) - {model_name.capitalize()}, \
        Case = {sample_output} | Pred = {sample_pred} (Prob = {sample_prob:.3f}, Thr = {threshold:.3f}))',
        fontsize=16
    )
    plt.tight_layout()
    plt.show()

from sklearn.decomposition import PCA

"""PCA visualization for true and predicted"""
"""The function assumes that X data is scaled (with StandardScaler or MinMaxScaler etc)"""
def pca_visualization(
        X_train, 
        X_new, 
        y_new_true, 
        y_new_pred, 
        colors_dict=None, 
        class_names_dict=None, 
        model_name='', 
        figsize=(13, 5.5), 
        point_size=35, 
        alpha=0.6, 
        valid_or_test='Validation'
    ):
    
    n_classes = len( np.unique(np.concatenate([y_new_true, y_new_pred])) )

    if len(X_new) != len(y_new_true) or len(X_new) != len(y_new_pred) or len(y_new_true) != len(y_new_pred):
        raise ValueError('X, y_true and y_pred should contain the same number of observations')
    if colors_dict is None or class_names_dict is None:
        raise ValueError('Provide colors_dict and class_names_dict')
    if len(colors_dict) != n_classes or len(class_names_dict) != n_classes:
        raise ValueError('Colors dict and class names dict must be defined as class label and corresponding color/class name')

    pca = PCA(n_components=2)
    pca.fit(X_train)
    components = pca.transform(X_new)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    axes = axes.flatten() 
    decoded_classes = class_names_dict # decoded scoring categories for classes

    # Plotting PCA components for true and predicted classes
    for i, (ax, y) in enumerate( zip(axes, [y_new_true, y_new_pred]) ):
        colors = [colors_dict[label] for label in y]
        ax.scatter(
            components[:, 0], 
            components[:, 1], 
            c=colors, 
            edgecolor='w', 
            s=point_size, 
            alpha=alpha
        )
        # Add grid lines ONLY at the origin (0,0) 
        ax.axvline(x=0, color='gray', alpha=0.8, zorder=0)
        ax.axhline(y=0, color='gray', alpha=0.8, zorder=0)
        title = f'{valid_or_test} data' if i == 0 else 'Predicted classes'
        ax.set_title(title)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)

        # Legend
        for class_value, color in colors_dict.items():
            ax.scatter(
                [], [], 
                c=color, 
                label=f'{class_value} - {decoded_classes[class_value]}'
            )
        ax.legend(title='Target Class')
    fig.suptitle(f'PCA Visualization - {model_name}', fontsize=18)
    plt.tight_layout()
    plt.show()