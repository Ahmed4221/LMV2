
from utils import *
from pdf2image import convert_from_path
def main():
    files = getListOfFiles(RAW_DATA_PATH)
    
    for file in files:
        
        file_name = file.split('/')[-1].split('.')[0]

        cv2.imwrite(os.path.join(TRAIN_IMG_DIR, file_name), cv2.cvtColor(np.array(convert_from_path(file)[0]), cv2.COLOR_BGR2RGB))

        image, data = get_labelled_image(convert_from_path(file)[0])
        
        save_json(data, file_name)
        
        save_labelled_image(image, file_name)

    if DRAW:
        draw_image(image)
    
    save_pickel()


if __name__ == '__main__':
    main()