"""Functions meant to extract relevant kernel information"""


def serialize_prior(p):
    if p is None:
        return None
    d = {"type": p.__class__.__name__}
    for attr in ["loc", "scale", "concentration", "rate", "df", "covariance_matrix"]:
        if hasattr(p, attr):
            val = getattr(p, attr)
            try:
                d[attr] = float(val)
            except Exception:
                try:
                    d[attr] = val.detach().cpu().numpy().tolist()
                except Exception:
                    d[attr] = str(val)
    return d


def serialize_constraint(c):
    if c is None:
        return None
    d = {"type": c.__class__.__name__}
    for attr in ["lower_bound", "upper_bound", "initial_value"]:
        if hasattr(c, attr):
            try:
                d[attr] = getattr(c, attr).item()
            except Exception:
                d[attr] = getattr(c, attr)
    return d


def describe_kernel_recursive(k):
    """Recursively describe any GPyTorch kernel tree."""
    out = {"type": k.__class__.__name__}

    # --- ARD dims ---
    if hasattr(k, "active_dims"):
        out["active_dims"] = list(k.active_dims)

    # --- lengthscale ---
    if hasattr(k, "raw_lengthscale"):
        out["raw_lengthscale"] = k.raw_lengthscale.detach().cpu().numpy().tolist()
    if hasattr(k, "lengthscale"):
        try:
            out["lengthscale"] = k.lengthscale.detach().cpu().numpy().tolist()
        except Exception:
            out["lengthscale"] = str(k.lengthscale)

    # priors
    if hasattr(k, "lengthscale_prior"):
        out["lengthscale_prior"] = serialize_prior(k.lengthscale_prior)
    if hasattr(k, "raw_lengthscale_constraint"):
        out["lengthscale_constraint"] = serialize_constraint(k.raw_lengthscale_constraint)

    # --- RQ kernel hyperparameters ---
    if hasattr(k, "raw_alpha"):
        out["raw_alpha"] = k.raw_alpha.detach().cpu().numpy().tolist()
    if hasattr(k, "alpha"):
        out["alpha"] = float(k.alpha.item())
    if hasattr(k, "alpha_prior"):
        out["alpha_prior"] = serialize_prior(k.alpha_prior)
    if hasattr(k, "raw_alpha_constraint"):
        out["alpha_constraint"] = serialize_constraint(k.raw_alpha_constraint)

    # --- Periodic kernel ---
    if hasattr(k, "raw_period_length"):
        out["raw_period_length"] = k.raw_period_length.detach().cpu().numpy().tolist()
    if hasattr(k, "period_length"):
        out["period_length"] = float(k.period_length.item())
    if hasattr(k, "period_length_prior"):
        out["period_length_prior"] = serialize_prior(k.period_length_prior)
    if hasattr(k, "raw_period_length_constraint"):
        out["period_length_constraint"] = serialize_constraint(k.raw_period_length_constraint)

    # --- Linear kernel variance ---
    if hasattr(k, "raw_variance"):
        out["raw_variance"] = k.raw_variance.detach().cpu().numpy().tolist()
    if hasattr(k, "variance"):
        out["variance"] = float(k.variance.item())
    if hasattr(k, "variance_prior"):
        out["variance_prior"] = serialize_prior(k.variance_prior)

    # --- Nested kernels (Additive or Product) ---
    if hasattr(k, "kernels"):
        out["sub_kernels"] = [describe_kernel_recursive(sub) for sub in k.kernels]

    return out


def describe_task_kernel(task_k):
    out = {
        "type": task_k.__class__.__name__,
        "num_tasks": int(task_k.num_tasks),
        "rank": int(task_k.rank),
        "var_raw": task_k.raw_var.detach().cpu().numpy().tolist(),
        "var": task_k.var.detach().cpu().numpy().tolist(),
        "covar_factor_raw": task_k.raw_covar_factor.detach().cpu().numpy().tolist(),
        "covar_factor": task_k.covar_factor.detach().cpu().numpy().tolist(),
    }
    if hasattr(task_k, "task_prior"):
        out["task_prior"] = serialize_prior(task_k.task_prior)
    return out


def extract_full_gp_config(model):
    """Return a complete, MLflow-friendly model description."""
    cfg = {}

    # Mean module
    cfg["mean_module"] = str(model.mean_module)

    # Feature covariance (kernel tree)
    cfg["kernel"] = describe_kernel_recursive(model.covar_module)

    # Task kernel
    cfg["task_kernel"] = describe_task_kernel(model.task_covar_module)

    # Noise
    lik = model.likelihood.noise_covar
    cfg["noise_raw"] = lik.raw_noise.detach().cpu().numpy().tolist()
    cfg["noise"] = lik.noise.detach().cpu().numpy().tolist()
    cfg["noise_prior"] = serialize_prior(getattr(lik, "noise_prior", None))
    cfg["noise_constraint"] = serialize_constraint(getattr(lik, "raw_noise_constraint", None))

    return cfg
