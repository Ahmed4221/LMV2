from train_utils import *
import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def results_test(preds, out_label_ids, labels):
  """ 
    This function produces evaluation metrices for the test results
    Param : preds
    Param-type : list of prediction
    Param : out_label_ids
    Param-type : list of ids of actual outputs
    Param : labels
    Param-type :  list of actual labels
    Returns : classification report of results
  """
    
  preds = np.argmax(preds, axis=2)

  label_map = {i: label for i, label in enumerate(labels)}

  out_label_list = [[] for _ in range(out_label_ids.shape[0])]
  preds_list = [[] for _ in range(out_label_ids.shape[0])]


  for i in range(out_label_ids.shape[0]):
      for j in range(out_label_ids.shape[1]):
          if out_label_ids[i, j] != -100:
              out_label_list[i].append(label_map[out_label_ids[i][j]])
              preds_list[i].append(label_map[preds[i][j]])

  results = {
      "precision": precision_score(out_label_list, preds_list),
      "recall": recall_score(out_label_list, preds_list),
      "f1": f1_score(out_label_list, preds_list),
  }
  return results, classification_report(out_label_list, preds_list)


def run_inference(model):
    """ 
        This function Runs main inference over the test files stored in test dir
        Params : model
        Param-type : torch model
        Returns : None
    """
    test_dataloader,labels,test_dataset,processor = prepare_dataloader(TEST_PICKLE_NAME,TEST_IMG_DIR)
    preds_val = None
    out_label_ids = None

    # put model in evaluation mode
    model.eval()
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            image = batch['image'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
            if preds_val is None:
                preds_val = outputs.logits.detach().cpu().numpy()
                out_label_ids = batch["labels"].detach().cpu().numpy()
            else:
                preds_val = np.append(preds_val, outputs.logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0
                )
    labels = list(set(overall_labels))
    val_result, class_report = results_test(preds_val, out_label_ids, labels)
    print("Overall results:", val_result)
    print(class_report)





