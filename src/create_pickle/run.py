from create_pickle import *
from config import *


words_test, boxes_test, labels_test = generate_annotations(TEST_JSON)

save_pickle(words_test, boxes_test, labels_test, TEST_PICKEL)

words_train, boxes_train, labels_train = generate_annotations(TRAIN_JSON)

save_pickle(words_train, boxes_train, labels_train, TRAIN_PICKEL)
