
from utils import *
from train import *
from inference import *
from pdf2image import convert_from_path
def main():
    """ The main controller which runs the entire pipeline
        Params : None
        Returns : None
    """
    model = None
    if LABEL_DATA:
        files = getListOfFiles(RAW_DATA_PATH)
        unique_rows = []    # For row_id function
        pkl_word = []
        pkl_box = []
        pkl_label = []
        for file in files:   
            file_name = file.split('/')[-1].split('.')[0]
            #writing image as pdf here
            cv2.imwrite(os.path.join(TRAIN_IMG_DIR, file_name +'.jpg'), cv2.cvtColor(np.array(convert_from_path(file)[0]), cv2.COLOR_BGR2RGB))
            #get image labeled
            image, data = get_labelled_image(convert_from_path(file)[0],pkl_word,pkl_box,pkl_label,unique_rows)
            #saving pickel
            save_pickel(pkl_word,pkl_box,pkl_label)
            #savig the json file in case we need to save pickle later on
            save_json(data, file_name)
            #saving the labelled image separately
            save_labelled_image(image, file_name)

        if DRAW:
            draw_image(image)
    if TRAIN:
        model = train()

    if TEST:

        if model == None:
            # model = torch.load(os.path.join(MODEL_SAVE_PATH,'pytorch_model.bin'))
            print("LOADING MODEL")
            model = torch.load(os.path.join(MODEL_SAVE_PATH,PATH))
            try:
                model.eval()
            except:
                print("Model loading didn't work")
        run_inference(model)

    
    


if __name__ == '__main__':
    main()