"""Prompt registry keyed by dataset interface."""

from __future__ import annotations

_SYSTEM_PROMPT_LINEAR = (
    "/no_think\n"
    "Write a PyMC model. Output ONLY a ```python block, no explanation.\n\n"
    "Rules:\n"
    "- Define `def model(data):` that returns a `pm.Model`\n"
    "- ALL pm.* calls MUST be inside `with pm.Model() as m:`, nothing outside\n"
    "- data['X'] is an (n, p) feature matrix, data['y'] is an (n,) response vector\n"
    "- data['n'] is the number of observations, data['p'] is the number of features\n"
    "- Use `import pymc as pm` and `import numpy as np`\n"
    "- Vary prior choices, model structure, and parameter names\n"
    "- Informative priors are fine; try different hyperparameter scales\n"
    "- Do NOT copy the examples below verbatim\n\n"
    "Available PyMC API:\n"
    "- Distributions: pm.Normal, pm.HalfNormal, pm.HalfCauchy, pm.StudentT, pm.Laplace,\n"
    "  pm.Exponential, pm.Gamma, pm.Uniform, pm.Beta, pm.Cauchy, pm.InverseGamma, etc.\n"
    "- Observation: pm.Normal('y', mu=mu, sigma=sigma, observed=data['y'])\n"
    "- Data wrapping: pm.MutableData('y_data', data['y'])\n"
    "- Auxiliary log-prob terms: pm.Potential('name', expression)\n"
    "- Custom distributions: pm.DensityDist('name', logp=fn, observed=...)\n"
    "- Log-probability: pm.logp(rv, value)\n"
    "- Math: pm.math.sigmoid, pm.math.log, pm.math.exp, pm.math.sum, pm.math.dot\n"
    "- PyTensor: import pytensor.tensor as pt\n\n"
    "Example styles:\n"
    "```python\n"
    "# Style 1: Simple linear regression\n"
    "def model(data):\n"
    "    with pm.Model() as m:\n"
    "        beta = pm.Normal('beta', 0, 10, shape=data['p'])\n"
    "        sigma = pm.HalfNormal('sigma', 5)\n"
    "        mu = pm.math.dot(data['X'], beta)\n"
    "        pm.Normal('y', mu=mu, sigma=sigma, observed=data['y'])\n"
    "    return m\n"
    "```"
)

_SYSTEM_PROMPT_BERNOULLI = (
    "/no_think\n"
    "Write a PyMC model for binary observations. Output ONLY a ```python block, no explanation.\n\n"
    "Rules:\n"
    "- Define `def model(data):` that returns a `pm.Model`\n"
    "- ALL pm.* calls MUST be inside `with pm.Model() as m:`, nothing outside\n"
    "- data['y'] contains binary outcomes (0/1)\n"
    "- data['d'] is the interface dimension and data['n'] is number of observations\n"
    "- Use `import pymc as pm` and `import numpy as np`\n"
    "- You may use pm.Bernoulli, pm.Beta, pm.Binomial, pm.Potential, pm.DensityDist\n"
    "- Do NOT output prose outside the code block\n"
)

_LINEAR_PROMPTS = [
    "Fit a linear regression model to predict y from X.",
    "Model the relationship between features and response with Bayesian inference.",
    "Estimate regression coefficients with uncertainty quantification.",
    "Build a Bayesian linear model for this dataset.",
    "Fit a regression model with appropriate priors on coefficients.",
    "Predict the response variable from the feature matrix.",
    "Model y as a linear function of X with noise estimation.",
    "Estimate feature effects on the response using Bayesian regression.",
    "Fit a probabilistic regression model to the data.",
    "Build a linear model with prior distributions on all parameters.",
]

_BERNOULLI_PROMPTS = [
    "Model binary outcomes using Bayesian inference.",
    "Fit a Bernoulli/Binomial model for observed 0/1 data.",
    "Infer uncertainty over a latent probability for binary data.",
    "Build a probabilistic model for binary observations.",
    "Fit a Bayesian model that explains the binary response vector.",
    "Model the Bernoulli data-generating process with priors on parameters.",
]

_PROMPTS_BY_DATASET = {
    "bernoulli_1d": (_SYSTEM_PROMPT_BERNOULLI, _BERNOULLI_PROMPTS),
    "bernoulli_vector": (_SYSTEM_PROMPT_BERNOULLI, _BERNOULLI_PROMPTS),
    "linear_regression": (_SYSTEM_PROMPT_LINEAR, _LINEAR_PROMPTS),
    "logistic_regression": (_SYSTEM_PROMPT_LINEAR, _LINEAR_PROMPTS),
    "gaussian_location": (_SYSTEM_PROMPT_LINEAR, _LINEAR_PROMPTS),
}


def get_prompt_spec(dataset_name: str) -> tuple[str, list[str]]:
    try:
        system_prompt, prompts = _PROMPTS_BY_DATASET[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"No prompt spec for dataset {dataset_name!r}; expected one of "
            f"{sorted(_PROMPTS_BY_DATASET)}"
        ) from exc
    return system_prompt, list(prompts)


def get_prompts(dataset_name: str, n: int) -> list[str]:
    _, pool = get_prompt_spec(dataset_name)
    if n <= 0:
        return []
    return [pool[i % len(pool)] for i in range(n)]


def format_chat_prompt(dataset_name: str, user_message: str) -> str:
    system_prompt, _ = get_prompt_spec(dataset_name)
    return (
        f"<|im_start|>system\\n{system_prompt}<|im_end|>\\n"
        f"<|im_start|>user\\n{user_message}<|im_end|>\\n"
        f"<|im_start|>assistant\\n"
    )


def get_system_prompt(dataset_name: str) -> str:
    system_prompt, _ = get_prompt_spec(dataset_name)
    return system_prompt
