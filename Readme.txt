Install required dependencies by opening a cmd window and navigating to the current folder using cd command as :
cd path_to_folder

Call  in cmd window:
pip install -r requirements.txt

The main program should be run by running the SearchUI.py file. Run SearchUI.py by entering in the cmd window :

python SearchUI.py

All files in the directory and sub-folders should remain unmoved as this file needs dependencies in the other folders to be run successfully.

The main UI can be found in SearchUI.py (this is also the program to be run)

The Datasets can be found in the folder BaseData and are in a csv format.
    the datasets are :
        -steam_data.csv
        -text_content.csv

All the intermediate stage results are stored in pickle dumps in the folder Stored Data, these are used in processing the query

The Indexer.py file contains all the various steps from producing the cleaned data frames, to processing the query.
    - the actual query processing function is : process_query_reduced(query, True)

the tests can be run by running Test.py, I am not sure on how to compare the results of the test, and evaluate them, so, i am only displaying the results here.

The results are shown in the UI by webscraping using Webscraper.py. This file contains the functions needed to retrieve prices, ratings and other listing information from the steam store.

The Final_Report-Implementation Project Project is the final report of the file.

Youtube link is provided in the report.