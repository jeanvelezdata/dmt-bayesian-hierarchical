Bayesian Hierarchical Model of Pre–Post Psychological Measures

==============================================================



Overview

--------

This repository contains a fully reproducible Bayesian hierarchical model analyzing changes in psychological measures before and after an intervention.



The model jointly estimates change across multiple scales while partially pooling information across:



• subjects (random intercepts and slopes)  

• measures (random intercepts and slopes)



This allows more stable inference than running separate paired tests per scale, while accounting for heterogeneity across individuals and measures.





Data Source

-----------

The dataset is publicly hosted on Zenodo:



https://zenodo.org/records/3992359



The repository automatically downloads the dataset during execution and verifies its checksum to ensure reproducibility.



No raw data are stored in this repository.





Model Structure

---------------

The model estimates standardized outcome scores using:



• a global intercept  

• a global time effect  

• subject-level random intercepts and slopes  

• measure-level random intercepts and slopes  

• residual noise  



Subject random effects are modeled with a correlated covariance structure using a manual Cholesky decomposition.



Measure effects are modeled independently using a non-centered parameterization.



The primary estimand is:



&nbsp;   β\_m = global time effect + measure-specific deviation



which represents the standardized change from pre to post for each measure.





Repository Structure

--------------------

configs/          Run configurations  

src/dmt\_bayes/    Core package code  

tests/            Smoke tests  

results/          Generated outputs (ignored by git)  



Key modules:



• data\_prep.py — reshape + standardization  

• model.py — hierarchical PyMC model  

• fit.py — sampler interface  

• report.py — tables and plots  

• cli.py — command-line interface  





Installation

------------

Create an environment and install in editable mode:



&nbsp;   python -m pip install -e .



Dependencies include:



• pymc  

• arviz  

• pandas  

• numpy  

• matplotlib  

• pyyaml  

• requests  





Reproducing Results

-------------------

Run the full workflow:



&nbsp;   dmt-bayes run --config configs/default.yaml



For a fast demonstration run:



&nbsp;   dmt-bayes run --config configs/fast.yaml



This will:



1\. Download the dataset from Zenodo  

2\. Fit the hierarchical model  

3\. Save posterior draws and summaries  





Generating Tables and Plots (no refit required)

-----------------------------------------------

From saved inference results:



&nbsp;   dmt-bayes summarize --idata results/idata.nc

&nbsp;   dmt-bayes plot --idata results/idata.nc





Outputs

-------

Posterior draws:

&nbsp;   results/idata.nc



Measure labels:

&nbsp;   results/measure\_labels.json



Tables:

&nbsp;   results/tables/posterior\_summary.csv

&nbsp;   results/tables/beta\_by\_measure.csv



Figures:

&nbsp;   results/figures/trace.png

&nbsp;   results/figures/beta\_by\_measure\_forest.png





Interpretation

--------------

Outcomes are standardized within each measure prior to modeling.



Therefore:



• estimated slopes are expressed in standard deviation units  

• a value of 0.5 indicates a half-SD increase from pre to post  

• uncertainty intervals reflect posterior credible intervals  



The hierarchical structure allows:



• shrinkage toward the global effect  

• robust estimation for noisy measures  

• simultaneous inference across scales  





Testing

-------

A smoke test verifies the full pipeline runs:



&nbsp;   pytest



GitHub Actions automatically runs tests on each commit.





Reproducibility Guarantees

--------------------------

• Dataset downloaded from a fixed Zenodo record  

• Checksum verification enforced  

• Config-driven sampling  

• Saved inference objects allow full regeneration of outputs  





License

-------

MIT License

