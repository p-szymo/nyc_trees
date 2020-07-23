# Verifying Volunteer Entries to the NYC Street Tree Census

Using NYC Open Data's [2015 Street Tree Census Data Set](https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/pi5s-9p35) (click **Export** to download a non-geospatial version of the dataset), I conduct exploratory data analysis and feature engineering to find the significance of certain variables on the health of NYC street trees. I then create a classification model to gain further insights into which factors play a role in a tree's health.

Ultimately, my goal is to determine which features most help my model perform well and what information may be helpful for future censuses and policy in general. In the future, I would like to create a list of volunteer data entries for which the health status does not match that of my model's predictions. This list could be used by professionals to double-check the work of volunteers and re-determine a tree's health if necessary.

#### Answer the following questions:
- *What features are most important in determining the health of a tree?*
    - *Hypothesis:* Neighborhood and problems in trunk, root, or branches will be the most important.
- *Are there any features that may help in future censuses?*
    - *Hypothesis*: Age of the tree would be a great variable to include.
- *What do the numbers of entries for each class of tree health tell us about volunteer efforts?*
    - *Hypothesis*: Volunteers tend to rate trees in better condition.
- *Do maps shed any light on any issues with the census?*

## Findings
- Species of tree is significant, Norway maple having the lowest rate of trees in good health on average, and sawtooth oak having the highest rate.
    - I would recommend sticking to the healthier varieties (toward the right side of the graph seen below) and avoiding the least healthy varieties (toward the left of the graph).
        - *NOTE: graph sorted by proportion of trees in* Good *health, in ascending order.*
![Tree Health Across Species](images/charts/health_species_barstack.png)

- Problems with the tree itself were obviously significant.
    - Unfortunately, problems that were listed as *Other* appear to hold the highest significance.
    - In the next census, I recommend including columns with more specificity or a notes column (which one could analyze using NLP).
    - I suggest conducting more regular maintenance, as well as pushing for environmental protections, in the style of the plastic ban bag.
![Tree Health with Root, Trunk, and Branch Problems](images/charts/health_problems_3barstack.png)

- All levels of neighborhood delineation held significance.
    - I chose community board as my neighborhood variable in part because one must petition the community board to make changes to street trees.
![Tree Health Across Community Boards](images/charts/health_cb_barstack.png)

- It's difficult to choose which model to use based off of numbers alone. One must think carefully about how it is being used.
    - I went with a model that was more inaccurate but had a better spread of predictions. Not ready for primetime yet, but a step in the right direction.
    - I created a metric--precision of *Good* predictions--that serves as a fairly good proxy for the results I'm looking for, though more investigation may be necessary.
![Random Forest Confusion Matrices Comparison](images/charts/forest_cm_comparison.png)

- Many of the most important features--including, but not limited to, the number of trees on the same block, distance to the nearest tree, and presence of sidewalk damage--didn’t show much significance when looked at during EDA, but must have had solid predictive power after interacting with other branches of the Random Forest trees.

## Top features in the model (in order of average decrease in Gini impurity), out of 147 total features:
    Distance to nearest tree    (15.8%)
    Number of trees on block    (14.0%)
    Tree diameter               (13.7%)
    Species [Norway maple]      (2.1%)
    Branch problems [other]     (2.1%)
    Tree stewards [1-2]         (2.1%)
    Trunk problems [other]      (2.1%)
    Sidewalk damage             (1.9%)
    Root problems [stone]       (1.3%)
    Branch problems [light]     (1.2%)
    On curb                     (1.1%)
    Species [London planetree]  (1.1%)
    Species [Honeylocust]       (1.1%)
    Tree guards [helpful]       (0.9%)
    Root problems [other]       (0.9%)

![Top Features - Random Forest](images/charts/final_model_feature_importances.png)

# Final conclusion
I can run my final model on the data collected by volunteers and compile a list of trees for which health statuses do not match. As it stands, the NYC Street Trees Census is fairly thorough and rife with opportunities for data exploration and predictive modeling. Perhaps with a neural network, I could greatly improve my model, although computational power is a limiting factor there. In future censuses, even more data could be gathered (especially in regard to specificity of tree problems) that will increase opportunities for prediction even further.

## List of files
- **functions.py** - text file with functions for data cleaning and statistical tests.
- **eda_modeling_evaluation.ipynb** - Jupyter Notebook for data exploration and classification modeling.
- **maps_creations.ipynb** - Jupyter Notebook for making maps of NYC street trees.
- **presentation.pdf** - presentation for New York City Department of Parks and Recreation with my findings.
- **archives** folder - scrap notebooks.
- **images** folder - insightful charts and maps from the project.



## Check out my [blog post](https://medium.com/@joshua.szymanowski/new-york-forest-rangers-d11b19e386a8)
