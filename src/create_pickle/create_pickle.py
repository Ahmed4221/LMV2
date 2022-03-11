import json
import os
import pickle

def generate_annotations(path: str):
    
    annotation_files = []
    for js in (os.listdir(path)):
        with open(os.path.join(path,js)) as f:
            annotation_files.append(json.load(f))
  

    words = []
    boxes = []
    labels = []

    for js in annotation_files:
        words_example = []
        boxes_example = []
        labels_example = []

        for line in js:
            
            word = line['words']
            # get word
            txt = word['text']

            # get bounding box
            # important: each bounding box should be in (upper left, lower right) format
            # it took me some time to understand the upper left is (x1, y3)
            # and the lower right is (x3, y1)
            x1 = word['quad']['x1']
            y1 = word['quad']['y1']
            x3 = word['quad']['x3']
            y3 = word['quad']['y3']
            
            box = [x1, y1, x3, y3]

            # ADDED
            # skip empty word
            if len(txt) < 1: 
                continue
            if min(box) < 0 or max(box) > 1000: # another bug in which a box had -4
                continue
            if ((box[3] - box[1]) < 0) or ((box[2] - box[0]) < 0): # another bug in which a box difference was -12
                continue
            # ADDED

            words_example.append(txt)
            boxes_example.append(box) 
            labels_example.append(word['category'])
            
        words.append(words_example) 
        boxes.append(boxes_example)
        labels.append(labels_example)

    return words, boxes, labels


def save_pickle(words, boxes, labels, path: str):
    with open(path, 'wb') as t:
        pickle.dump([words, labels, boxes], t)