# Intro

This repository provides the solution to the tasks formulated in the README.md that can be found under
[Google Drive](https://drive.google.com/drive/folders/1KZTA_a442UXXen9sr8lbdnnoGX2Qod9p). For reference the readme has also been integrated in this repo under [INITIALREADME.md](INITIALREADME.md).

In a nutshell, the repo 

1. Provides functionality to train and serialize a GradientBoostedTree Model that is able to match talents to jobs, see [training_task.py](src/tasks/training_task.py).
2. Implements functionality to use the trained ML model to match jobs to talents, see [search_task.py](src/tasks/search_task.py).

# Install & Setup

1. Create a new virtual environment using e.g. https://github.com/pyenv/pyenv. The project was developed and tested using python version==3.11.1
2. Activate the virtual environment
3. Run ```make setup``` in the terminal to install all relevant project requirements to obtain a full dev setup
4. Place the input data of the project that can be found in [Google Drive](https://drive.google.com/drive/folders/1KZTA_a442UXXen9sr8lbdnnoGX2Qod9p) into the following location: data/data.json
5. In case you are using an IDE, you can now configure the project interpreter to be the just installed virtual environment and run [training_task.py](src/tasks/training_task.py) & [search_task.py](src/tasks/search_task.py) 

# Notes on the selected modeling approach

* Decided to perform restrictions to the universe, (see [COMMON.conf](config/COMMON.conf)), i.e. 
  * only take talent languages into consideration that are required or desired in the job universe (German & English)
  * only take roles into consideration, that are searched for in the job universe
* Features that come with a natural ranking, i.e. maturity related features & language skills have been encoded in a ranked fashion (see [COMMON.conf](config/COMMON.conf) for the chosen encoding)
* Matching/discrepancy features between talent and job (e.g. expected salary discrepancy) have been created to help the model associate the right features with each other
* A stratified split was chosen to preserve the curated 50/50 distribution of labels also during testing. Performed only a twofold split (no dev set) with 80/20 distribution, as no hyperparam tuning was required. 
* [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) was chosen, as the modeling approach is known to work very well for tabular non-normalized data.
* Accuracy was chosen as quality metric due to its very natural interpretation and the balanced nature of the target variable 
* Feature selection has been performed based on theoretical considerations (e.g. avoiding perfectly linearly dependent features), the calculated feature importance (see [Modeling Prototyping](notebooks/20240709_Prototyping.ipynb) and the model quality measurement (see [COMMON.conf](config/COMMON.conf) for the final feature selection)
* Final accuracy was measured to be 99.25%, which was considered to be very good compared to a naive baseline (e.g. always predict label=1) which is expected to have an accuracy of 50%


# Important final notes

* Under [Modeling Prototyping](notebooks/20240709_Prototyping.ipynb) a notebook has been included that was used for **early** data understanding & protoyping of the ML approach
* The functions ```match``` and ```match_bulk``` that were required to be implemented can be found here [search.py](src/search.py) 
* Docstrings have only been written for functions, that are not easily understandable via the function name only
* As per the instructions in the task, no unit tests or CI/CD components have been integrated
* If you'd like to run black and isort for code style and formatting, run ```make format-code``` in the terminal