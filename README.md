# Airflow DAG for Model Evaluation and Monitoring

This repository contains an Airflow DAG (Directed Acyclic Graph) that performs model evaluation and monitoring tasks using Google Sheets and email notifications.

## Overview

The Airflow DAG consists of several tasks:

### Load Model File from Git

This task retrieves the model file from a GitLab repository and saves it locally.

### Create COCO Dataset from Supervisely

This task creates a COCO dataset by cleaning and converting data from the Supervisely platform.

### Evaluate Model

This task evaluates the model using the created COCO dataset and generates evaluation scores.

### Grafana

This task performs actions related to Grafana for monitoring purposes.

### Conditioning

This task checks if the evaluation score meets a threshold value.

### Alerting

This task sends an email alert based on the evaluation result.

## Requirements

To run this DAG, you need the following dependencies:

- Airflow: Install Airflow by following the instructions in the official documentation.
- Python 3: Make sure you have Python 3 installed on your system.
- Required Python packages: Install the required Python packages by running `pip install -r requirements.txt`.

## Scripts

The repository contains the following scripts:

### load_data.py

Contains functions to retrieve image data from the Supervisely platform.

### clean_dataset.py

Contains functions to clean the dataset.

### create_coco_dataset.py

Contains functions to create a COCO dataset.

### evaluation.py

Contains functions to evaluate the model.

### my_dag.py

The Airflow DAG script that defines the tasks and their dependencies.

## Configuration

Before running the DAG, you need to configure the following parameters:

- GitLab configuration: Set up your GitLab URL, private token, project ID, branch name, and file paths in the configuration file (`person_config.yaml`).
- Supervisely configuration: Set up your Supervisely API endpoint, access token, team ID, workspace ID, project ID, and headers in the configuration file (`person_config.yaml`).
- Threshold value: Specify the threshold value for the evaluation score in the `conditionning` task.

## Usage

1. Clone this repository to your local machine.
2. Update the configuration file (`person_config.yaml`) with the necessary parameters.
3. Set up Airflow and configure the DAG folder.
4. Copy the DAG file (`my_dag.py`) to the DAG folder in your Airflow installation.
5. Start the Airflow scheduler and webserver.
6. Access the Airflow web interface and trigger the DAG manually or set up a schedule for automatic execution.

## License

This code is provided under the [MIT License](LICENSE).
