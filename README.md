# Verifying Volunteer Entries to the NYC Trees Census

To accomplish this, answer the following questions:
* What features are most important in determining the health of a tree?
    * Hypothesis: Neighborhood and problems in trunk, root, or branches will be the most important.
* Are there any features that may help in future censuses?
    * Hypothesis: Age of the tree would be a great variable to include.
* What do the numbers of entries for each class of tree health tell us about volunteer efforts?
    * Hypothesis: Volunteers tend to rate trees in better condition.
* Do maps shed any light on any issues with the census?
    
## Approach
Using NYC Open Datas's [2015 Street Tree Census Data Set](https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/pi5s-9p35) (click **Export** to download a non-geospatial version of the dataset), I perform exploratory data analysis and feature engineering to find the significance of certain variables on the health of NYC street trees. I then create a classification model in the hopes of checking the likelihood of the veracity of the health status imputed by volunteers. Ultimately, my goal is to determine which features are most important to help my model, which information may be helpful for future censuses, and finally, creating a list of trees imputed by volunteers for which the health status does not match that of my model.

## Findings
* Species of tree is significant, Norway maple being the least good on average, and sawtooth oak being the most good.
    * I would recommend sticking to the healthier varieties in the graph created earlier, and avoiding the least healthy varieties.
* Problems with the tree itself were obviously significant.
    * Unfortunately, problems that were listed as “other” appear to hold the highest significance. In the next census, they should include columns with more specificity or a notes column (which one could feasibly process to create dummy variables).
    * I suggest conducting more regular maintenance, as well as pushing for environmental protections, in the style of the plastic ban bag.
* All levels of neighborhood delineation held significance.
    * I chose community board as my neighborhood variable in part because one must petition the community board to make changes to street trees.
* It's difficult to choose which model to use based off numbers alone. One must think carefully about how it is being used.
    * I went with a model that was more inaccurate but had a better spread of predictions. Not ready for primetime yet, but a step in the right direction.
* Many of the most prevalent features, including number of trees on block, distance to the nearest tree, and sidewalk damage, didn’t show much significance when looked at during EDA, but must have had solid predictive power after interacting with other branches of the Random Forest trees.

## Most prevalent features in the model (in order)
### Top features (out of 147):
     Tree diameter              (14.8%)
     Number of trees on block   (11.2%)
     Distance to nearest tree   (9.1%)
     Sidewalk damage            (7.6%)
     Tree stewards [1-2]        (5.1%)
     Root problems [stone]      (4.8%)
     Species [Norway maple]     (2.6%)
     Trunk problems [other]     (2.4%)
     Branch problems [light]    (2.2%)
     Species [London planetree] (2.0%)
     Branch problems [other]    (2.0%)
     Community board [503]      (1.8%)
     Species [Honeylocust]      (1.3%)
     Borough [Queens]           (1.2%)


# Final conclusion
* I can run my final model on the data collected by volunteers and compile a list of trees that whose health statuses do not line up. In the meantime, the NYC Street Trees Census is fairly thorough and rife with opportunities for data exploration and predictive modeling. Perhaps with a neural network, I could greatly improve my model. In future censuses, even more data could be gathered (especially in regard to specificity) that will increase these opportunities for prediction even further.

## List of files
* **functions.py** - text file with functions for data cleaning and statistical tests.
* **nyc_trees_final_notebook.ipynb** - Jupyter Notebook for data exploration and classification modeling.
* **nyc_trees_workbook_mapmaking.ipynb** - Jupyter Notebook for making maps of NYC street trees.
* **presentation.pdf** - presentation for New York City Department of Parks and Recreation with my findings.
* **archives** folder - Includes scratch notebooks.
* **charts** folder - PNG files of insightful charts from the project.
* **data** folder - Main CSV file, description of columns, and pickles (note: csv files too large to upload).
* **maps** folder - PNG files; screenshots of maps (note: html files are too large to upload.)



### BLOG POST FORTHCOMING
