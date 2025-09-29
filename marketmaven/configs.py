"""
Pydantic Configs used for MLFlow tracking
"""
from pydantic import BaseModel, Field
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

class TestingConfig(BaseModel):
    start_date: date = Field(..., description="Start date for the data range")
    end_date: date = Field(..., description="End date for the data range")
    interval: Interval = Field(Interval.DAILY, description="Data frequency")
    tickers: list[str] = Field(..., description="List of asset tickers")
    horizon: Horizon = Field(Horizon.MONTHLY, description="Prediction horizon")

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
    data: TestingConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig