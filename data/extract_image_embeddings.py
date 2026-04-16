import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from dataloader import ConceptDataset, collate_fn
from open_clip import create_model_from_pretrained, get_tokenizer

@torch.no_grad()
def extract_image_embeddings(model, 
                             dataset, 
                             processor=None, 
                             batch_size: int = 256,
                             medical: bool = False,
                             num_workers: int = 8):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    all_embeddings, all_files = [], []

    for _, (images, _, image_files, _) in tqdm(enumerate(loader), total=len(loader), desc="Extracting embeddings"):
        images = [img[0] for img in images]  # unwrap from list

        if medical:
            images = torch.stack([processor(img) for img in images]).to(device)
            embeddings = model.encode_image(images)
        else:
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token

        all_embeddings.append(embeddings.detach().cpu().numpy())
        all_files.extend(image_files)

    return np.concatenate(all_embeddings, axis=0), all_files        

if __name__ == "__main__":
    data_name = "cub200"            # set to "oai" to use MedImageInsight
    dataset = ConceptDataset(data_name=data_name,
                             image_size=224, 
                             in_context=0, 
                             vary_context=1,
                             context_templates="llama_templates.json")
    print(len(dataset), "images in dataset")

    if data_name == "oai":
        model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, processor = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        medical = True
    else:
        model_name = "facebook/dinov2-base"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        medical = False

    embeddings, images = extract_image_embeddings(
        model=model,
        dataset=dataset,
        processor=processor,
        batch_size=256,
        medical=medical,
        num_workers=8
    )

    np.save(f"data/{data_name}/train_image_embeddings.npy", embeddings)
    with open(f"data/{data_name}/train_image_files.txt", "w") as f:
        for p in images:
            f.write(f"{p}\n")