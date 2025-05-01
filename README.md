# DSCI-441
Git repo for final project in DSCI-441 on mammalian dietary traits

The dimensionality of datasets processed and derived from studies, research projects and field analysis have vastly increased. This results in researchers needing to know skills that, in decades prior, would be considered non-essential. In research that leverages large and/or high-dimension datasets, an essential technique that is required today is the ability to conduct dimensionality reduction. Dimensions, or features, are the predictors that determine a model's output. At a glance, having more predictors than needed appears a non-issue, however it has ramifications: increased computation time, increased storage space requirements for big data, and other issues that can be constrained by a lab's financial backing. However, the biggest concern is that machine learning models trained on high-dimensional datasets often generalize poorly. In the evolutionary research domain of biological sciences, it is very common for research papers' final dataset(s) to feature non-standard formatting & nomenclature, inconsistent data structure, feature overlap, and other characteristics that interfere with the datasets ability to be used in further analysis. This project will apply dimensionality reduction techniques on datasets merged from various research papers to distill relevant features and improve robustness of data for further analysis.

Data sources

trait_data_reported.csv and trait_data_imputed.csv (reported vs. imputed trait percentages)

MamFuncDat.csv (EltonTraits 1.0)

MammalDIET_v1.0.txt.csv and mam12119-sup-0001-appendixs1.csv (Gainsbury et al. 2018)

Amniote_Database_Aug_2015.csv (Myhrvold et al. 2015)

Supplemental_data_1_diet_dataset.csv (DietaryGuild compilation)

mergedData.csv (expert-labeled validation set)

Requirements Python 3.8 or later
pandas
numpy
scikit-learn
matplotlib
seaborn

Installation

Clone the repository to your local machine.

In the project root, create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Usage

Place all raw CSV/TSV data files in the data/ directory.

Launch Jupyter Lab or Notebook in the project folder.

Open the notebook merge_pca_classification_prep.ipynb.

Execute each cell in order to:

load and merge all trait sources

clean, reorder, and impute missing values

perform manual PCA and compare to scikit-learn

run decision-tree classification with each reduction method

conduct permutation tests and bootstrap confidence intervals

visualize PCA biplots, learning curves, and feature-correlation clustermaps

To run as a standalone script, export the notebook to a .py file and execute:
jupyter nbconvert --to script merge_pca_classification_prep.ipynb
python merge_pca_classification_prep.py
