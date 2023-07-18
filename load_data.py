import requests
import shutil
import json
import os
from tqdm import tqdm
import random
#curl -H "x-api-key: <token-here>" https://app.supervise.ly/public/api/v2/projects.list
def get_image_supervisely(datasetId, dest_folder, teamId = 20958, workspaceId = 45960, projectId = 117956, p = .8,headers='',
                          toShuffle = False) :
    """

    :param teamId: supervisely parameter
    :param workspaceId: supervisely parameter
    :param projectId: supervisely parameter
    :param datasetId: id of dataset on supervisely
    :param dest_folder: local folder to save dataset
    :param p: percentage of train and test data (between 0 and 1)
    :param toShuffle: option to shuffle dataset before selecting train and test data
    """
    data={"teamId":teamId,
         "workspaceId":workspaceId,
         "projectId":projectId,
         "datasetId":datasetId}

    response = requests.get('https://app.supervise.ly/public/api/v3/images.list', headers=headers, data=data)
    print(response.json())
    images_id=[i['id'] for i in response.json()['entities']]
    images_name=[i['name'] for i in response.json()['entities']]
    dest_folder_train=os.path.join(dest_folder, "train_original")
    dest_folder_test=os.path.join(dest_folder, "test")
    os.makedirs(dest_folder_train, exist_ok=True)
    os.makedirs(dest_folder_test, exist_ok=True)
    if toShuffle :
        i_list = range(len(images_id))
        random.shuffle(i_list)
    for i in tqdm(range(len(images_id)), desc = "Downloading Supervisely images") :
        if (i<len(images_id)*p):
             dest_folder=dest_folder_train
        else:
             dest_folder=dest_folder_test
        if toShuffle :
            i = i_list[i]
        id = images_id[i]
        data1={"teamId":teamId,
             "workspaceId":workspaceId,
             "projectId":projectId,
             "datasetId":datasetId ,
             'imageId':id} #info to get annotations information
        data2={"teamId":teamId,
             "workspaceId":workspaceId,
             "projectId":projectId,
             "datasetId":datasetId,
             'id':id} #info to download images
        dest_file_img=os.path.join(dest_folder, "img/")
        dest_file_ann=os.path.join(dest_folder, "ann/")
        os.makedirs(dest_file_img, exist_ok=True)
        os.makedirs(dest_file_ann, exist_ok=True)
        response = requests.get('https://app.supervise.ly/public/api/v3/annotations.info', headers=headers, data=data1,stream=True)
        if response.status_code == 200:
            with open(dest_file_ann+str(images_name[i])+'.json', 'w') as fp:
                json.dump(response.json()["annotation"], fp)
                fp.close()
        response = requests.get('https://app.supervise.ly/public/api/v3/images.download', headers=headers, data=data2,stream=True)      
        if response.status_code == 200:
            with open(dest_file_img+str(images_name[i]), 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f) 
     