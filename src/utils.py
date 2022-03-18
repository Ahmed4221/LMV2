# Imports
from constants import *
from transformers import LayoutLMv2FeatureExtractor
import pandas as pd
import cv2
import numpy as np
import pickle




def unnormalize_box(bbox, width, height):
    """ 
    This function un-normalizes the bbox of words
    Param : bbox
    Param-type : list, list having coordinates of bounding box
    Param : width
    Param-type : float, width of bbox
    Param : height
    Param-type : float, height of bbox
    Returns : quad
    """
    
    data = [
         int(width * (bbox[0] / 1000)),
         int(height * (bbox[1] / 1000)),
         int(width * (bbox[2] / 1000)),
         int(height * (bbox[3] / 1000)),
    ]

    quad = convert_to_quad(data)

    return quad


def convert_to_quad(data):
    
    # converting to 8 co-ordinate system (FOR THE CORD!!!)
    quad = {
        "x2": data[2],
        "y3": data[3],
        "x3": data[2],
        "y4": data[3],
        "x1": data[0],
        "y1": data[1],
        "x4": data[0],
        "y2": data[1]
    }

    return quad


def get_row_id(line, row_id,unique_rows):
    
    if len(unique_rows) == 0:
        id = row_id + 1
        unique_rows.append(
            {
                'row': line,
                'row_id': id
            }
        )

    else:
        found_match = False
        
        for row in unique_rows:    
            
            if abs(line - row['row']) in range(0, 4):

                id = row['row_id']
                found_match = True   
            
            else:
                continue     
        
        if found_match == False:
            id = row_id + 1
            unique_rows.append(
                {
                    'row': line,
                    'row_id': id
                    }
                    )
    return id


def get_data(filename,pkl_word,pkl_box,pkl_label,unique_rows):

    img = cv2.cvtColor(np.array(filename), cv2.COLOR_BGR2RGB)
    height = img.shape[0] 
    width =  img.shape[1]

    feature_extractor = LayoutLMv2FeatureExtractor(do_resize=False)

    encoding = feature_extractor(img, return_tensors="np")
    
    # Removing one layer of wrap
    encoding['words'] = encoding['words'][0]
    encoding['boxes'] = encoding['boxes'][0]    # Normalized boxes
    
    true_boxes = [unnormalize_box(box, width, height) for box in encoding['boxes']]
    word = []
    row_id = 0

    word_examples = []
    box_examples = []
    label_examples = []
    image_data = []

    for box, text, unnorm_boxes in zip(encoding['boxes'], encoding['words'], true_boxes):
        
        quad_box = convert_to_quad(box)

        row_id = get_row_id(quad_box['y3'], row_id,unique_rows)

        img_copy = img.copy()
        
        img_copy = create_box(unnorm_boxes, text, img_copy)
        img_copy = cv2.resize(img_copy, (700, 700))
        
        cv2.imshow('img', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(ground_labels)
        
        valid = False
        while valid == False:
            try:
                category = input("Please enter the category: ")
                label = ground_labels[category]
                valid = True
            except:
                print("Invalid key, try again")
        
        label = ground_labels[category]

        row = ({'quad': quad_box,
        'text': text,
        'row_id': row_id,
        'category': label,
        'groud_id': 0
        }) # For the CORD Dataset

        image_data.append({
            'box': unnorm_boxes,
            'text': text
        }) # For processing images

        word.append(row)

        # For the pickle file(Training)
        word_examples.append(text)
        box_examples.append(box)
        label_examples.append(label)


    df = pd.DataFrame(list(zip(word)), 
    columns=['words'])
    
    pkl_word.append(word_examples)
    pkl_box.append(box_examples)
    pkl_label.append(label_examples)

    image_data = pd.DataFrame(list(zip(image_data)),
                        columns=['norm_data'])
    

    return df, img, image_data


def create_box(bbox, label, image):

    x1 = bbox['x1']
    y1 = bbox['y1']
    x2 = bbox['x3']
    y2 = bbox['y3']

    image = cv2.rectangle(image, (x1, y1), (x2, y2), (12, 12, 255), 1)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (12, 12, 255), 2)

    return image


def save_json(data_file, file_name):
    json_file_path = file_name+"_processed.json"
    data_file.to_json(os.path.join(PROCESSED_DATA_JSON_PATH,json_file_path), orient='records', lines=True)


def save_labelled_image(image, file_name):
    image_file_name = file_name+"_processed.jpg"
    cv2.imwrite(os.path.join(PROCESSED_DATA_PATH,image_file_name), image)


def get_labelled_image(file,pkl_word,pkl_box,pkl_label,unique_rows):

    data, image, image_data = get_data(file,pkl_word,pkl_box,pkl_label,unique_rows)

    for i in range(len(image_data)):
        
        box, text= image_data['norm_data'][i]['box'], image_data['norm_data'][i]['text']

        image = create_box(box, text, image)

    
    return image, data


def draw_image(image):
    
    cv2.imshow('Labelled image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getListOfFiles(dirName,allowed_extensions = ALLOWED_EXTENSIONS):    
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            extenstion = fullPath[-3:]
            if extenstion in allowed_extensions:
                allFiles.append(fullPath)
                
    return allFiles


def save_pickel(pkl_word,pkl_box,pkl_label):
    with open(TRAIN_PICKLE_NAME, 'wb') as t:
        pickle.dump([pkl_word, pkl_label, pkl_box], t)