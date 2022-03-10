
from utils import *
from pdf2image import convert_from_path
def main():
    files = getListOfFiles(RAW_DATA_PATH)
    file_order = []
    for file in files:
        file_order.append(file)    
        file_name = file.split('/')[-1].split('.')[0]

        print(os.path.join(TRAIN_IMG_DIR, file_name))
        cv2.imwrite(os.path.join(TRAIN_IMG_DIR, file_name +'.jpg'), cv2.cvtColor(np.array(convert_from_path(file)[0]), cv2.COLOR_BGR2RGB))

        image, data = get_labelled_image(convert_from_path(file)[0])
        
        save_pickel()
        
        save_json(data, file_name)
        
        save_labelled_image(image, file_name)

    if DRAW:
        draw_image(image)
    
    df = pd.DataFrame(file_order)
    df.to_csv('file_order.csv', index=False)


if __name__ == '__main__':
    main()