"""Pydantic Configs used for MLFlow tracking."""
from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

######## Market Data Configs ########


class Interval(StrEnum):
    """Supported Yahoo Finance fetch intervals.

    Granularity of raw data coming from Yahoo. This is the data vendor frequency.
    """

    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"
    QUARTERLY = "3mo"


class Horizon(StrEnum):
    """Supported prediction horizons."""

    DAILY = "B"  # Business day
    WEEKLY = "W-FRI"  # Friday is often considered end of trading week
    MONTHLY = "BME"  # Business month end, aligns with last trading day of the month
    QUARTERLY = "BQ"  # Business quarter end
    YEARLY = "BA"  # Business year end


class TickerConfig(BaseModel):
    """Configuration for asset ticker data fetching.

    Attributes:
        start_date: Start date for the data range.
        end_date: End date for the data range.
        interval: Data frequency.
        tickers: List of asset tickers.
        horizon: Prediction horizon.
        lookback_date: Optional lookback start date for features.
    """

    start_date: date = Field(..., description="Start date for the data range")
    end_date: date = Field(..., description="End date for the data range")
    interval: Interval = Field(Interval.DAILY, description="Data frequency")
    tickers: list[str] = Field(..., description="List of asset tickers")
    horizon: Horizon = Field(Horizon.MONTHLY, description="Prediction horizon")
    lookback_date: date | None = Field(None, description="Optional lookback start date for features")

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


####### Priors Configs ########


class PriorType(StrEnum):
    """Supported prior distribution types."""

    LOGNORMAL = "lognormal"
    GAMMA = "gamma"
    UNIFORM = "uniform"


class PriorConfig(BaseModel):
    """Configuration for a prior distribution.

    Attributes:
        prior_type: Distribution type.
        loc: Location parameter.
        scale: Scale parameter.
        constraint_min: Minimum constraint value.
    """

    prior_type: PriorType
    loc: float
    scale: float
    constraint_min: float = 1e-6


###### Model configs ######


class KernelType(StrEnum):
    """Supported Gaussian Process kernels."""

    MATERN = "matern"
    LINEAR = "linear"
    RQ = "rq"
    RBF = "rbf"
    PERIODIC = "periodic"
    SPECTRAL_MIXTURE = "spectralmixture"


class OptimizerType(StrEnum):
    """Supported optimizer types."""

    ADAM = "adam"
    SGD = "sgd"
    LBFGS = "lbfgs"


class TrainingConfig(BaseModel):
    """Configuration for model training.

    Attributes:
        optimizer: Optimizer type.
        learning_rate: Learning rate.
        max_iter: Maximum number of iterations.
        patience: Early stopping patience.
        seed: Random seed for reproducibility.
    """

    optimizer: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.1
    max_iter: int = 500
    patience: int = 10
    seed: int = 1


class ModelType(StrEnum):
    """Supported GP model types."""

    SINGLE_TASK = "single_task_gp"
    KRON_MULTI = "kronecker_multitask_gp"
    HADAMARD_MULTI = "hadamard_multitask_gp"
    EXACT_GP = "exact_gp"


class ModelConfig(BaseModel):
    """Configuration for a GP model.

    Attributes:
        model_type: GP model variant.
        kernel: Kernel type.
        prior: Optional prior configuration.
        outcome_transform: Whether to apply outcome transform.
        input_transform: Whether to apply input transform.
    """

    model_type: ModelType
    kernel: KernelType
    prior: PriorConfig | None = None
    outcome_transform: bool = True
    input_transform: bool = False


###### Evaluation Configs ######


class MetricType(StrEnum):
    """Supported evaluation metrics."""

    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    SPEARMAN = "spearman"


class AggregateMode(StrEnum):
    """Aggregation modes for multi-asset metrics."""

    MACRO = "macro"
    WEIGHTED = "weighted"
    FLAT = "flat"


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation.

    Attributes:
        metrics: List of metrics to compute.
        aggregate: Aggregation mode.
        weights: Optional per-asset weights for weighted aggregation.
    """

    metrics: list[MetricType] = [MetricType.RMSE, MetricType.MAE, MetricType.R2]
    aggregate: AggregateMode = AggregateMode.MACRO
    weights: dict[str, float] | None = None


class CVConfig(BaseModel):
    """Configuration for cross-validation.

    Attributes:
        step: Step size between folds.
        horizon_cv: Forecast horizon used in CV.
        embargo: Embargo gap between train and test.
        training_min: Minimum number of training observations.
    """

    step: int = 1
    horizon_cv: int = 1
    embargo: int = 0
    training_min: int = 60


###### Experiment Config ######


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration.

    Attributes:
        data: Ticker data configuration.
        model: Model configuration.
        training: Training configuration.
        evaluation: Evaluation configuration.
    """

    data: TickerConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


######## Riskfolio Config ########
class RiskMeasure(StrEnum):
    """Supported risk measures for portfolio optimization."""

    MV = "MV"
    KT = "KT"
    MAD = "MAD"
    GMD = "GMD"
    MSV = "MSV"
    SKT = "SKT"
    FLPM = "FLPM"
    SLPM = "SLPM"
    CVaR = "CVaR"
    TG = "TG"
    EVaR = "EVaR"
    RLVaR = "RLVaR"
    WR = "WR"
    RG = "RG"
    CVRG = "CVRG"
    TGRG = "TGRG"
    EVRG = "EVRG"
    RVRG = "RVRG"
    MDD = "MDD"
    ADD = "ADD"
    CDaR = "CDaR"
    EDaR = "EDaR"
    RLDaR = "RLDaR"
    UCI = "UCI"


class Objective(StrEnum):
    """Supported portfolio optimization objectives."""

    MIN_RISK = "MinRisk"
    UTILITY = "Utility"
    SHARPE = "Sharpe"
    MAX_RET = "MaxRet"


class OptModel(StrEnum):
    """Supported portfolio optimization models."""

    CLASSIC = "Classic"
    BL = "BL"
    FM = "FM"
    BLFM = "BLFM"


class MuEstimator(StrEnum):
    """Supported expected return estimators."""

    HIST = "hist"
    EWMA1 = "ewma1"
    EWMA2 = "ewma2"
    JS = "JS"
    BS = "BS"
    BOP = "BOP"


class CovEstimator(StrEnum):
    """Supported covariance estimators."""

    HIST = "hist"
    EWMA1 = "ewma1"
    EWMA2 = "ewma2"
    LEDOIT = "ledoit"
    OAS = "oas"
    SHRUNK = "shrunk"
    GL = "gl"
    JLOGO = "jlogo"
    FIXED = "fixed"
    SPECTRAL = "spectral"
    SHRINK = "shrink"
    GERBER1 = "gerber1"
    GERBER2 = "gerber2"


class KurtEstimator(StrEnum):
    """Supported kurtosis square matrix estimators."""

    HIST = "hist"
    SEMI = "semi"
    FIXED = "fixed"
    SPECTRAL = "spectral"
    SHRINK = "shrink"


class RiskfolioConfig(BaseModel):
    """Configuration for Riskfolio portfolio optimization.

    Attributes:
        model: Portfolio optimization method.
        rm: Risk measure.
        rf: Risk-free rate.
        ra: Risk aversion level.
        hist: Use historical data for risk estimation.
        obj: Objective for portfolio optimization.
        method_mu: Expected return estimator.
        method_cov: Covariance estimator.
        method_kurt: Kurtosis square matrix estimator (optional).
        upperlng: Upper limit for long positions.
        nea: Number of assets in the portfolio.
    """

    model: OptModel = Field(OptModel.CLASSIC, description="Portfolio optimization method")
    rm: RiskMeasure = Field(RiskMeasure.MV, description="Risk measure")
    rf: float = Field(0.0, description="Risk-free rate")
    ra: float = Field(0.5, description="Risk aversion level")
    hist: bool = Field(True, description="Use historical data for risk estimation")
    obj: Objective = Field(Objective.SHARPE, description="Objective for portfolio optimization")
    method_mu: MuEstimator = Field(MuEstimator.HIST, description="Expected return estimator")
    method_cov: CovEstimator = Field(CovEstimator.HIST, description="Covariance estimator")
    method_kurt: KurtEstimator | None = Field(None, description="Kurtosis square matrix estimator (optional)")
    upperlng: float = Field(0.35, description="Upper limit for long positions")
    nea: int = Field(6, description="Number of assets in the portfolio")

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)
