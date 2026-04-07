import pymc as pm


def fit_model(
    model,
    draws=4000,
    tune=1500,
    chains=4,
    target_accept=0.95,
    max_treedepth=15,
    random_seed=42,
):
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler_kwargs={"max_treedepth": max_treedepth},
            return_inferencedata=True,
            random_seed=random_seed,
        )
    return idata