
import xgboost as xgb 
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def xgboost_variable_importance(X, y, xgb_params=None) -> dict:
    """
    Returns a list of global variable importances, and returns a plotly figure of the 
    interaction values.

    Parameters:
    model (xgb.Booster): The trained XGBoost model.
    feature_names (list): List of feature names used in the model.

    Returns:
    dict: A dictionary with feature names as keys and their importance scores as values.
    """
    params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",          # or "gpu_hist" if available
    "max_cat_to_onehot": 20,        # tune per your cardinalities
    "seed": 42,
    # Regularization usually helps on small chemical datasets:
    "lambda": 0.2,                  # L2
    "alpha": 0.2,                   # L1
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    }

    Xy = xgb.DMatrix(X, y, enable_categorical=True)
    booster = xgb.train(params, Xy, num_boost_round=500)
    booster.predict(Xy)
    SHAP_int = booster.predict(Xy, pred_interactions=True)
    feature_names = booster.feature_names
    shap_main = np.diagonal(SHAP_int, axis1=1, axis2=2)
    shap_main = shap_main[:, :-1]  # drop bias term
    
        
    shap_df = pd.DataFrame(
        shap_main,
        columns=feature_names,
        index=X.index
    )

    global_importance = shap_df.abs().mean().sort_values(ascending=False)

    interaction_strength = np.mean(np.abs(SHAP_int), axis=0)

    # Drop bias row & column
    interaction_strength = interaction_strength[:-1, :-1]

    interaction_df = pd.DataFrame(
        interaction_strength,
        index=feature_names,
        columns=feature_names
    )


    # Assuming interaction_df is a pandas DataFrame with the correlation matrix
    # Compute the correlation matrix if not already computed
    # interaction_df = your_dataframe.corr()

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=interaction_df.values,
        x=interaction_df.columns,
        y=interaction_df.index,
        colorscale='Viridis'
    ))

    # Update layout for better visualization
    fig.update_layout(
        title='Interaction Matrix Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange='reversed'),
        height=900,
        width=900
    )
    
    return global_importance, fig
