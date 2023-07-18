from datetime import datetime
import logging
import os
import json
import requests
import gitlab
import yaml

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from load_data import get_image_supervisely
from clean_dataset import iterate_folder
from create_coco_dataset import iterate_folder_create_coco, add_meta
from evaluation import evaluation_complet


def get_model_file(path_config_file: str = '/home/hid/airflow/airflow/airflow/params/person_config.yaml',
                   **context) -> None:
    """
    Retrieves the model file from GitLab and saves it locally.

    Args:
        path_config_file (str): Path to the configuration file. Defaults to '/home/hid/airflow/airflow/airflow/params/person_config.yaml'.
        **context: Additional context provided by Airflow.

    Returns:
        None
    """
    with open(path_config_file) as fh:
        config_pipeline = yaml.load(fh, Loader=yaml.FullLoader)

    gl = gitlab.Gitlab(url='https://gitlab.com', private_token=config_pipeline['private_token'])
    project_id = config_pipeline['project_id']
    branch_name = config_pipeline['branch_name']
    old_file_name = 'model_0076999.pth'
    dir = 'models/person/'
    project = gl.projects.get(project_id)
    file_name = project.repository_tree(path=dir, ref=branch_name)[-1]['name']

    if old_file_name == file_name:
        # The same model version is running; no update
        print("The same model {} is running".format(file_name))
        with open(old_file_name, 'wb') as f:
            project.files.raw(file_path=dir + old_file_name, ref=branch_name, streamed=True, action=f.write)
    else:
        # The model has been updated
        with open(file_name, 'wb') as f:
            print("A new model {} is running".format(file_name))
            project.files.raw(file_path=dir + file_name, ref=branch_name, streamed=True, action=f.write)

    context['ti'].xcom_push(key='load_dir', value=old_file_name)


def create_coco_dataset(path_config_file: str = '/home/hid/airflow/airflow/airflow/params/person_config.yaml',
                        **context) -> None:
    """
    Creates a COCO dataset by cleaning and converting data.

    Args:
        path_config_file (str): Path to the configuration file. Defaults to '/home/hid/airflow/airflow/airflow/params/person_config.yaml'.
        **context: Additional context provided by Airflow.

    Returns:
        None
    """
    with open(path_config_file) as fh:
        config_pipeline = yaml.load(fh, Loader=yaml.FullLoader)
        data = {
            "teamId": config_pipeline['teamId'],
            "workspaceId": config_pipeline['workspaceId'],
            "projectId": config_pipeline['projectId'],
            "sort": "createdAt"
        }
    list_dir = []
    coco_dict = {}
    classes_list = config_pipeline['classes_list']
    data_path = config_pipeline['data_path']
    polygon = config_pipeline['polygon']
    response = requests.get('https://app.supervisely.com/public/api/v3/datasets.list',
                            headers=config_pipeline['headers'], data=data, stream=True)

    if response.status_code == 200:
        dataset_id = response.json()['entities'][-1]['id']

    logging.info('Data download done!')
    iterate_folder(data_path, polygon, list_dir, classes_list)  # Clean data
    logging.info("Cleaning data done!")
    print(classes_list)
    classes_list = add_meta(classes_list, coco_dict, polygon)
    coco_path = os.path.join(data_path, 'coco_dataset')
    logging.info(f"Creating coco dataset in {coco_path}")
    os.makedirs(coco_path, exist_ok=True)
    iterate_folder_create_coco(os.path.join(data_path, 'train_original'), os.path.join(data_path, 'train_original'),
                               coco_dict, classes_list, polygon)  # Create COCO
    with open(os.path.join(coco_path, 'coco_train' + '.json'), 'w') as fp:
        json.dump(coco_dict, fp)
        fp.close()

    context['ti'].xcom_push(key='dir_coco_dataset_test', value=os.path.join(coco_path, 'coco_train' + '.json'))
    context['ti'].xcom_push(key='dir_img_test', value=os.path.join(data_path, 'train_original'))
    context['ti'].xcom_push(key='save_dir', value=data_path)
    context['ti'].xcom_push(key='num_classes', value=len(classes_list) - 1)


def evaluate(**context) -> dict:
    """
    Evaluates the model using the created COCO dataset.

    Args:
        **context: Additional context provided by Airflow.

    Returns:
        dict: Evaluation scores.
    """
    dir_coco_dataset_test = context['ti'].xcom_pull(key='dir_coco_dataset_test')
    dir_img_test = context['ti'].xcom_pull(key='dir_img_test')
    save_dir = context['ti'].xcom_pull(key='save_dir')
    num_classes = context['ti'].xcom_pull(key='num_classes')
    load_dir = context['ti'].xcom_pull(key='load_dir')
    score_dict = evaluation_complet(
        dir_coco_dataset_test=dir_coco_dataset_test,
        dir_img_test=dir_img_test,
        save_dir=save_dir,
        num_classes=num_classes,
        load_dir=load_dir
    )
    context['ti'].xcom_push(key='score_dict', value=score_dict)
    return score_dict


def grafana(**context) -> int:
    """
    Performs actions related to Grafana.

    Args:
        **context: Additional context provided by Airflow.

    Returns:
        int: Return value (0).
    """
    return 0


def conditionning(thr: int = 60, **context) -> bool:
    """
    Checks if the evaluation score meets the threshold.

    Args:
        thr (int): Threshold value. Defaults to 60.
        **context: Additional context provided by Airflow.

    Returns:
        bool: True if evaluation score is above the threshold, False otherwise.
    """
    score_dict = context['ti'].xcom_pull(key='score_dict')
    return score_dict['bbox']['AP50'] > thr


def alerting(**context) -> int:
    """
    Sends an email alert based on the evaluation result.

    Args:
        **context: Additional context provided by Airflow.

    Returns:
        int: Return value (0).
    """
    if context['ti'].xcom_pull(key='return_value'):
        print('An email should be sent to some people')
    else:
        print('We are good')
    return 0


with DAG("my_dag", start_date=datetime(2021, 1, 1), schedule_interval="@daily", catchup=False) as dag:
    load_model = PythonOperator(
        task_id='Load_model_file_from_git',
        python_callable=get_model_file,
        provide_context=True,
        dag=dag,
        op_kwargs={'path_config_file': '/home/hid/airflow/airflow/airflow/params/person_config.yaml'}
    )
    load_data_set = PythonOperator(
        task_id='Create_coco_data_set_from_supervisely',
        python_callable=create_coco_dataset,
        provide_context=True,
        dag=dag,
        op_kwargs={'path_config_file': '/home/hid/airflow/airflow/airflow/params/person_config.yaml'}
    )
    evaluate = PythonOperator(
        task_id="Evaluate",
        python_callable=evaluate,
        provide_context=True,
        dag=dag,
    )
    granfana = PythonOperator(
        task_id="grafana",
        python_callable=grafana,
        provide_context=True,
        dag=dag,
    )
    conditionning = PythonOperator(
        task_id="conditionning",
        python_callable=conditionning,
        provide_context=True,
        dag=dag,
    )
    alerting = PythonOperator(
        task_id="alerting",
        python_callable=alerting,
        provide_context=True,
        dag=dag,
    )

    [load_model, load_data_set] >> evaluate >> [conditionning, granfana]
    conditionning.set_downstream(alerting)
