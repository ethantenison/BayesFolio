"""
Pydantic Configs used for MLFlow tracking
"""
from pydantic import BaseModel, Field, ConfigDict
from datetime import date
from enum import StrEnum

######## Market Data Configs ########
    
class Interval(StrEnum):
    """Supported Yahoo Finance fetch intervals. Granularity of raw data coming from Yahoo. This is the data vendor frequency."""
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"
    QUARTERLY = "3mo"
    
class Horizon(StrEnum):
    """Supported prediction horizons."""
    DAILY = "B" # Business day
    WEEKLY = "W-FRI" # Friday is often considered end of trading week
    MONTHLY = "BM" # Business month end, aligns with last trading day of the month
    QUARTERLY = "BQ" # Business quarter end
    YEARLY = "BA" # Business year end

class TickerConfig(BaseModel):
    start_date: date = Field(..., description="Start date for the data range")
    end_date: date = Field(..., description="End date for the data range")
    interval: Interval = Field(Interval.DAILY, description="Data frequency")
    tickers: list[str] = Field(..., description="List of asset tickers")
    horizon: Horizon = Field(Horizon.MONTHLY, description="Prediction horizon")
    
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

####### Priors Configs ########


class PriorType(StrEnum):
    LOGNORMAL = "lognormal"
    GAMMA = "gamma"
    UNIFORM = "uniform"

class PriorConfig(BaseModel):
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
    ADAM = "adam"
    SGD = "sgd"
    LBFGS = "lbfgs"

class TrainingConfig(BaseModel):
    optimizer: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.1
    max_iter: int = 500
    patience: int = 10
    seed: int = 1
    
class ModelType(StrEnum):
    SINGLE_TASK = "single_task_gp"
    KRON_MULTI = "kronecker_multitask_gp"
    HADAMARD_MULTI = "hadamard_multitask_gp"
    EXACT_GP = "exact_gp"

class ModelConfig(BaseModel):
    model_type: ModelType
    kernel: KernelType
    prior: PriorConfig | None = None
    outcome_transform: bool = True
    input_transform: bool = False
    
    
###### Evaluation Configs ######
class MetricType(StrEnum):
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    SPEARMAN = "spearman"

class AggregateMode(StrEnum):
    MACRO = "macro"
    WEIGHTED = "weighted"
    FLAT = "flat"

class EvaluationConfig(BaseModel):
    metrics: list[MetricType] = [MetricType.RMSE, MetricType.MAE, MetricType.R2]
    aggregate: AggregateMode = AggregateMode.MACRO
    weights: dict[str, float] | None = None
    
###### Experiment Config ######
class ExperimentConfig(BaseModel):
    data: TickerConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    
#riskfolio config
class RiskMeasure(StrEnum):
    MV    = "MV"     # Standard Deviation
    KT    = "KT"     # sqrt Kurtosis
    MAD   = "MAD"    # Mean Abs Deviation
    GMD   = "GMD"    # Gini Mean Difference
    MSV   = "MSV"    # Semi Std Dev
    SKT   = "SKT"    # sqrt Semi Kurtosis
    FLPM  = "FLPM"   # 1st Lower Partial Moment
    SLPM  = "SLPM"   # 2nd Lower Partial Moment
    CVaR  = "CVaR"   # Conditional VaR
    TG    = "TG"     # Tail Gini
    EVaR  = "EVaR"   # Entropic VaR
    RLVaR = "RLVaR"  # Relativistic VaR (needs MOSEK)
    WR    = "WR"     # Worst Realization
    RG    = "RG"     # Range
    CVRG  = "CVRG"   # CVaR Range
    TGRG  = "TGRG"   # Tail Gini Range
    EVRG  = "EVRG"   # EVaR Range
    RVRG  = "RVRG"   # RLVaR Range (needs MOSEK)
    MDD   = "MDD"    # Max Drawdown
    ADD   = "ADD"    # Avg Drawdown
    CDaR  = "CDaR"   # Conditional Drawdown at Risk
    EDaR  = "EDaR"   # Entropic Drawdown at Risk
    RLDaR = "RLDaR"  # Relativistic Drawdown at Risk (MOSEK)
    UCI   = "UCI"    # Ulcer Index

class Objective(StrEnum):
    MIN_RISK = "MinRisk"
    UTILITY  = "Utility"
    SHARPE   = "Sharpe"
    MAX_RET  = "MaxRet"
    
class OptModel(StrEnum):
    CLASSIC = "Classic"
    BL      = "BL"
    FM      = "FM"
    BLFM    = "BLFM"
    
class MuEstimator(StrEnum):
    HIST   = "hist"
    EWMA1  = "ewma1"
    EWMA2  = "ewma2"
    JS     = "JS"
    BS     = "BS"
    BOP    = "BOP"
    
class CovEstimator(StrEnum):
    HIST     = "hist"
    EWMA1    = "ewma1"
    EWMA2    = "ewma2"
    LEDOIT   = "ledoit"
    OAS      = "oas"
    SHRUNK   = "shrunk"
    GL       = "gl"
    JLOGO    = "jlogo"
    FIXED    = "fixed"
    SPECTRAL = "spectral"
    SHRINK   = "shrink"
    GERBER1  = "gerber1"
    GERBER2  = "gerber2"

class KurtEstimator(StrEnum):
    HIST     = "hist"
    SEMI     = "semi"
    FIXED    = "fixed"
    SPECTRAL = "spectral"
    SHRINK   = "shrink"
    
class RiskfolioConfig(BaseModel):
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