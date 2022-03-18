from collections import Counter
from statistics import mode
import pandas as pd
from transformers import LayoutLMv2ForTokenClassification, AdamW, AutoModel
import torch
from tqdm.notebook import tqdm
from constants import *
from dataset import *
from transformers import LayoutLMv2Processor
from torch.utils.data import DataLoader

def get_all_labels(train_pickle_file):
    """ 
    This function gets list of labels from the pickle file of labeled data
    Param : train_pickle_file
    Param-type : pickle file of multiple jsons with list of bbox, word and cat.
    Returns : list of labels
    """
    all_labels = [item for sublist in train_pickle_file[1] for item in sublist]
    Counter(all_labels)
    return all_labels

def prepare_dataloader(pickle_file_path,image_dir):
    """ 
    This create torch data loader object from CORDDataset Class
    Param : pickle_file_path
    Param-type : string, path of pickle file
    Param : image_dir
    Param-type : string, path of image_dir
    Returns : dataloader
    """
    pickle_file = pd.read_pickle(pickle_file_path)
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
    dataset = CORDDataset(annotations=pickle_file,
                                image_dir=image_dir, 
                                processor=processor,
                                label2id = label2id)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader,overall_labels,dataset,processor

def train_model(train_dataloader,labels):
    """ 
    This function trains the model
    Param : train_dataloader
    Param-type : torch data-loader, Dataloader object of training data
    Param : labels
    Param-type : list, list of labels
    Returns : model
    """
    
    model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                                      num_labels=len(labels))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=MODEL_CONFIG['lr'])

    global_step = MODEL_CONFIG['global_step']
    num_train_epochs = MODEL_CONFIG['num_train_epochs']

    #put the model in training mode
    model.train() 
    for epoch in tqdm(range(num_train_epochs)):  
        # print("Epoch:", epoch)
        over_all_loss = None
        for batch in (train_dataloader):
            # get the inputs;
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            image = batch['image'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(input_ids=input_ids,
                            bbox=bbox,
                            image=image,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels) 
            
            loss = outputs.loss
            over_all_loss = loss.item()
            loss.backward()
            optimizer.step()
            global_step += 1
        print(f"Loss after epoch {epoch} is: {over_all_loss}")

    # model.save_pretrained("/content/drive/MyDrive/LayoutLMv2/Tutorial notebooks/CORD/Checkpoints")
    # model.save_pretrained(MODEL_SAVE_PATH)
    model.save_pretrained(TRAINED_MODEL) #os.path.join(TRAINED_MODEL, MODEL_PATH))
    return model