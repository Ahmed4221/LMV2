
from utils import *
from pdf2image import convert_from_path
def main():
    files = getListOfFiles(RAW_DATA_PATH)
    # print(files)
    for file in files:
        
        image, data = get_labelled_image(convert_from_path(file)[0])
        
        file_name = file.split('/')[-1].split('.')[0]
        
        save_json(data, file_name)
        
        save_labelled_image(image, file_name)

    if DRAW:
        draw_image(image)
    
    save_pickel()


if __name__ == '__main__':
    main()