

\# Alzheimer's Disease Classification Using Machine Learning



A comparative machine-learning study investigating Alzheimer's disease classification using clinical, demographic, genetic, and neuroimaging biomarkers from the Alzheimer's Disease Neuroimaging Initiative (ADNI).



\*\*Authors:\*\* Adil Hasanbašić and Ajdin Širbegović



\## Project Overview



This project investigates the classification of cognitively normal (CN) individuals and individuals diagnosed with Alzheimer's disease (AD).



The primary objective is to compare machine-learning approaches while emphasizing sensitivity and the reduction of false negatives.



The research examines how feature selection, preprocessing, model choice, and decision thresholds influence classification performance.



\## Dataset



The research uses the ADNIMERGE dataset from the Alzheimer's Disease Neuroimaging Initiative (ADNI).



The analysis focuses on baseline observations and uses demographic, genetic, clinical, and neuroimaging information.



\*\*Note:\*\* The dataset is not included in this repository due to ADNI data-use restrictions. Authorized access must be obtained separately.



\## Methodology



\### Data Preprocessing



The preprocessing pipeline includes:



\- Baseline observation selection

\- Missing-value analysis and imputation

\- Categorical variable encoding

\- Feature standardization

\- Training-only preprocessing to prevent data leakage

\- Repeated stratified resampling



\### Feature Selection



Boruta was used to investigate feature relevance, with particular emphasis on identifying informative neuroimaging and genetic biomarkers.



LASSO regularization also provides embedded feature selection through coefficient shrinkage.



\### Machine Learning Models



The primary comparative study evaluates:



1\. \*\*k-Nearest Neighbors (kNN)\*\* — Distance-based classification with sensitivity-focused parameter selection.

2\. \*\*LASSO Logistic Regression\*\* — An interpretable linear classifier with L1 regularization.

3\. \*\*Random Forest\*\* — An ensemble classifier capable of modeling nonlinear feature relationships.



The repository also contains additional experimental implementations involving MLP, CNN1D, and TabNet.



\## Results



The final research report presents the following comparative results:



| Model | Sensitivity | Specificity | Balanced Accuracy | F1 | Accuracy |

|-------|-------------|-------------|-------------------|----|----------|

| kNN | 0.939 | 0.996 | 0.967 | 0.968 | 0.971 |

| LASSO | 0.638 | 0.953 | 0.795 | 0.749 | 0.817 |

| Random Forest | 0.709 | 0.928 | 0.819 | 0.786 | 0.833 |



These values are taken from the final project report.



kNN achieved the highest reported sensitivity and accuracy. The findings emphasize the importance of feature composition, threshold selection, and evaluating multiple metrics rather than accuracy alone.



\## Technologies



\- R

\- RStudio

\- Machine Learning

\- Statistical Modeling

\- Feature Selection

\- Classification and ROC Analysis



\## Repository Structure



\- `scripts/` — R scripts for preprocessing, model development, evaluation, and visualizations

\- `CS512\_ADNI.Rproj` — RStudio project configuration

\- `README.md` — Project documentation



The original local project also contains `data/` and `results/` directories, which are excluded from this public repository pending data and output review.



\## Reproducibility



The scripts document the research workflow, but the repository is not currently a fully self-contained execution environment.



Reproducing the experiments requires authorized ADNI dataset access, appropriate R packages, and the necessary preprocessing steps.



\## Research Report



\*\*A Comparative Study of Machine Learning Models for Alzheimer's Disease Classification\*\*



Authors: Adil Hasanbašić and Ajdin Širbegović



The study investigates sensitivity-focused classification, leakage-aware preprocessing, feature relevance, and comparative model evaluation.



\## Disclaimer



This project was developed for academic research purposes. Its models are not clinically validated and should not be used for medical diagnosis or clinical decision-making.



