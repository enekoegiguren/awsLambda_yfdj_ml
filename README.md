# Data Updater for the APP

This repository stores the container image files for the AWS Lambda function that updates the data on the YourFirstDataJob APP. 


![Alt text](https://github.com/enekoegiguren/lambda_jobdata/blob/main/awsLambda_francejobdata.jpg)


## Structure

This AWS Lambda function executes the following tasks:

1. Connects to the [France Travail](https://www.francetravail.fr/accueil/) API and extracts all the jobs related to the data field.
2. Performs data transformations, including:
   - Job description categorization for each job data role.
   - Calculation of minimum, maximum, and average salaries.
   - Skills identification.

3. Inserts the data into a PostgreSQL database hosted on EC2 for use in the app and also into a bucket for data storage.
