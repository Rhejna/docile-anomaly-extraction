import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification

def train_one_epoch(model: LayoutLMv3ForTokenClassification, train_dataloader: DataLoader, optimizer: AdamW, device: torch.device) -> float:
    """
    Input: the model, a DataLoader (the training one), the optimiser, the device.
    Output: the average loss over the entire epoch (a single number), useful for tracking progress from one epoch to the next.
    """
    model.train()
    cumul = 0

    for load in train_dataloader:
        batch = {key: value.to(device) for key, value in load.items()}
        outputs = model(**batch) # call the model directly on the batch
        # print(outputs.loss)
        outputs.loss.backward() # calculates the gradients
        optimizer.step() # adjusts the model’s settings
        optimizer.zero_grad() # resets the gradients to zero
        cumul += outputs.loss.item()

    return cumul/len(train_dataloader)


def evaluate_loss(model: LayoutLMv3ForTokenClassification, val_dataloader: DataLoader, device: torch.device) -> float:
    """
    Same signature as `train_one_epoch`, but using the validation DataLoader. 
    We simply want to know how badly the model performs on test data it hasn't been trained on, not to adjust its settings.
    """
    model.eval()
    cumul = 0

    with torch.no_grad():
        for load in val_dataloader:
            batch = {key: value.to(device) for key, value in load.items()}
            outputs = model(**batch) # call the model directly on the batch
            cumul += outputs.loss.item()

    return cumul/len(val_dataloader)