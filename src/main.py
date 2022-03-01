
from utils import *
from pdf2image import convert_from_path
def main():
    files = getListOfFiles(RAW_DATA_PATH)
    print(files)
    for file in files:
        
        image, data = get_labelled_image(convert_from_path(file)[0])
        file_name = file.split('/')[-1].split('.')[0]
        # image_file_name = file_name+"_processed.jpg"
        # cv2.imwrite(os.path.join(PROCESSED_DATA_PATH,image_file_name), image)
        json_file_path = file_name+"_processed.json"
        data.to_json(os.path.join(PROCESSED_DATA_JSON_PATH,json_file_path), orient='records', lines=True)
    if DRAW:
        draw_image(image)


if __name__ == '__main__':
    main()