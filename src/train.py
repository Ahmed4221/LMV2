from train_utils import *
def train():
    """ 
    This function acts as controller for training the model
    Param : None
    Returns : model
    """
    train_dataloader,labels,_,_ = prepare_dataloader(TRAIN_PICKLE_NAME,TRAIN_IMG_DIR)
    model = train_model(train_dataloader,labels)
    return model