README.md

# Course competition container

This project is a machine learning model written in Python for the LING 582 course at the University of Arizona. The model reads CSV training and test data to train a logistic regression model using PyTorch, classify the test data, and write the results in a third CSV file called 'results'. The project is containerized using Docker and Docker Compose.

---

## **Task**

Given two snippets of text, determine if both snippets were produced by the same author or not.

## **Code Layout**
There are two python script files: training.py and test.py.
 - training.py - reads the training data, trains the model and returns error information and metrics
 - test.py - *also runs the training file to train the model*, then reads the test data, makes predictions, and writes the results in a file called 'results.csv'
 
**Note that test.py will run the entire program**, but training.py may be run independently for fine-tuning.


The data directory contains the training and test data files. After the code runs, the results file will appear here also.
 - training.csv has the following structure:
```plaintext
    COLUMN	DESCRIPTION

    ID	    Unique ID for this datapoint
    
    TEXT	Two snippets of text separated by [SNIPPET]
    
    LABEL	The label for this datapoint (0 = not the same author, 1 = same author)
```
 - test.csv hast the following structure:
```plaintext
    COLUMN	DESCRIPTION

    ID	    Unique ID for this datapoint
    
    TEXT	Two snippets of text separated by [SNIPPET]
```
 - results.csv has the following structure:
```plaintext
    COLUMN	DESCRIPTION

    ID	    Unique ID for this datapoint (matches test.csv IDs)
    
    LABEL	The label for this datapoint (0 = not the same author, 1 = same author)
```

---

## **Directory Structure**
```plaintext
.
├── Dockerfile                  # Docker build instructions
├── docker-compose.yml          # Docker Compose file
├── training.py                 # Code for training/fine-tuning the model
├── test.py                     # Code for training the model AND classifying test data
├── requirements.txt            # Python dependencies
├── data/                       # Directory for input CSV data
    └── best_training_data.csv  # Data for training nn
    └── test.csv                # test data
    └── (results.csv)           # output of classifier results (created after script is run)
```

## **Setup**
1. Clone the repository:

    `git clone <repository-url>`

2. Navigate to competition_container repository directory:

    `cd <repository-name>/competition_container`

(chmod +x *.sh (to make the bash scripts executable)??)

3. Container commands (inside repository-directory)

    - Build the container:

        `docker-compose -f docker-compose.yml -p contest-script up -d`

    - Start the container:

        `docker-compose -f docker-compose.yml -p contest-script start`

    - Check the container logs to see program progress:

        `docker logs contest_container`

    - To stop the container:

        `docker-compose -f docker-compose.yml -p contest-script stop`

    - To restart the container: use the stop command followed by the start command above


## **Dependencies**
The project uses the following Python packages (listed in requirements.txt):
 - pandas==2.2.3
 - torch==2.5.1
 - numpy==2.1.3
 - nltk==3.9.1
 - scikit-learn==1.5.2

## 
License
This project is licensed under the MIT License. See LICENSE for details.



**Author**\
Lynette Boos\
Email: lboos@arizona.edu



