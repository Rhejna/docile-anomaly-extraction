import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification

def train_one_epoch(model: LayoutLMv3ForTokenClassification, dataloader: DataLoader, optimizer: AdamW, device: torch.device) -> float:
    model.train()
    cumul = 0

    for load in dataloader:
        batch = {key: value.to(device) for key, value in load.items()}
        outputs = model(**batch) # call the model directly on the batch
        print(outputs.loss)
        outputs.loss.backward() # calculates the gradients
        optimizer.step() # adjusts the model’s settings
        optimizer.zero_grad() # resets the gradients to zero
        cumul += outputs.loss.item()

    return cumul/len(dataloader)



    
    pass