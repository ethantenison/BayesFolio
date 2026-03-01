import plotly.express as px
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from math import log
from scipy.stats import lognorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def correlation_matrix(data: pd.DataFrame, title: str = "Correlation Heatmap"):
    
    # Select only continuous variables
    continuous_data = data.select_dtypes(include=[np.number])
    
    
    corr_matrix = continuous_data.corr()
    # Create a heatmap using Plotly
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdYlBu",
        title=title,
        zmin=-1,
        zmax=1,
    )

    fig.update_layout(width=800, height=800)
    return fig

def visualize_lognormal_distribution():
  

    samples = np.random.lognormal(mean=log(1.1), sigma=0.2, size=10000)

    plt.figure(figsize=(7,4))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color="orange")
    plt.axvline(np.median(samples), color="red", linestyle="--", label="median")
    plt.title("Histogram of LogNormal(μ=log(1), σ=0.5)")
    plt.xlabel("value")
    plt.ylabel("density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
def plot_lognormal_prior(loc, scale, label=None, color=None, x_max=20.0):
    """
    Visualize a LogNormalPrior(loc, scale) using matplotlib.

    Parameters
    ----------
    loc : float
        Mean of the underlying normal distribution (mu).
    scale : float
        Standard deviation of the underlying normal distribution (sigma).
    label : str
        Optional name for the curve.
    color : str
        Optional color for the curve.
    x_max : float
        Maximum x limit for plotting.
    """

    sigma = scale
    mu = loc

    # Generate x values (lengthscales)
    x = np.linspace(1e-6, x_max, 2000)

    # scipy's lognorm takes "s=sigma", "scale=exp(mu)"
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    plt.plot(x, pdf, label=label or f"loc={loc}, scale={scale}", color=color)
    plt.xlabel("Lengthscale ℓ")
    plt.ylabel("Density")
    plt.title("LogNormalPrior Density")
    plt.grid(True)
    plt.legend()


# SQRT2 = sqrt(2)
# SQRT3 = sqrt(3)
# plt.figure(figsize=(10, 6))

# # Your adaptive high-dim prior: loc = sqrt(2) + 0.5*log(52), scale = sqrt(3)
# from math import sqrt, log
# loc_etf = sqrt(0.5) + 0.01 * log(24) #sqrt(2) + 0.1 * log(52)
# scale_etf = 0.5

# loc_etf_sr = SQRT2 + log(24) *0.5
# scale_etf_sr = SQRT3

# loc_macro = sqrt(0.2) + 0.01 * log(30)
# scale_macro = 0.5

# loc_macro_sr = SQRT2 + log(1) *0.5
# scale_macro_sr = SQRT3

# plot_lognormal_prior(loc_etf, scale_etf, label="ETF", color="blue", x_max=60)
# plot_lognormal_prior(loc_etf_sr, scale_etf_sr, label="ETF-SR", color="lightblue", x_max=60)
# plot_lognormal_prior(loc_macro, scale_macro, label="Macro", color="orange", x_max=60)
# plot_lognormal_prior(loc_macro_sr, scale_macro_sr, label="Macro-SR", color="red", x_max=60)


def plot_pca_explained_variance(df: pd.DataFrame, cols: list):
    """
    Perform PCA on macro features and plot explained variance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the macro features.
    macro_cols : list
        List of column names corresponding to macro features.
    """
    # 1. Extract macro features
    X = df[cols].copy()

    # 2. Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Fit PCA
    pca = PCA()
    pca.fit(X_scaled)

    # 4. Explained variance
    explained_variance = pca.explained_variance_ratio_
    cum_variance = np.cumsum(explained_variance)

    # 5. Plot explained variance and cumulative variance
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, label='Individual Component Variance')
    plt.step(range(1, len(cum_variance)+1), cum_variance, where='mid', label='Cumulative Variance')
    plt.axhline(0.90, color='red', linestyle='--', label='90%')
    plt.axhline(0.95, color='green', linestyle='--', label='95%')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance - Macro Features')
    plt.legend()
    plt.show()

    # 6. Print how many components to keep
    n_comp_90 = np.argmax(cum_variance >= 0.90) + 1
    n_comp_95 = np.argmax(cum_variance >= 0.95) + 1

    print(f"Components needed for 90% variance: {n_comp_90}")
    print(f"Components needed for 95% variance: {n_comp_95}")
    
    
def apply_pca_and_replace(df, cols_to_pca, n_components, prefix="pca"):
    """
    Standardizes the selected columns, applies PCA, and replaces them 
    with the PCA component columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    cols_to_pca : list[str]
        List of column names to apply PCA to.
    n_components : int
        Number of principal components to keep.
    prefix : str, default="pca"
        Prefix for the new PCA columns.

    Returns
    -------
    df_out : pd.DataFrame
        Updated DataFrame with PCA components replacing original columns.
    scaler : StandardScaler
        The fitted scaler (useful for transforming future data).
    pca : PCA
        The fitted PCA object.
    """

    # Copy to avoid modifying original DataFrame
    df_out = df.copy()

    # Extract data to scale & PCA
    X = df_out[cols_to_pca].values

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Build PCA column names
    pca_cols = [f"{prefix}_pc{i+1}" for i in range(n_components)]

    # Insert PCA columns
    for i, col in enumerate(pca_cols):
        df_out[col] = X_pca[:, i]

    # Drop original columns
    df_out = df_out.drop(columns=cols_to_pca)

    return df_out, scaler, pca