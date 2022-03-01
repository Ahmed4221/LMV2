from train_utils import *
from transformers import LayoutLMv2ForTokenClassification, AdamW
from tqdm.notebook import tqdm


def main():
    dataloader, dataset = get_dataloader()

    model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased'
                                                                      )
                                                                      # num_labels=len(labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    global_step = 0
    num_train_epochs = 4

    #put the model in training mode
    model.train() 
    for epoch in range(num_train_epochs):  
        print("Epoch:", epoch)
        for batch in tqdm(dataloader):
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
                
                # print loss every 100 steps
                if global_step % 100 == 0:
                    print(f"Loss after {global_step} steps: {loss.item()}")

        loss.backward()
        optimizer.step()
        global_step += 1

    model.save_pretrained("data/model/Checkpoints")

    # encoding = dataset[0]
    # processor.tokenizer.decode(encoding['input_ids'])

if __name__ == "__main__":
    main()