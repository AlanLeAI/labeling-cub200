import os
import numpy as np
import pandas as pd
import json
from PIL import Image
from copy import deepcopy
from itertools import combinations
from typing import List, Optional
from collections import OrderedDict

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from transformers import AutoProcessor

class ConceptDataset(Dataset):
    def __init__(self,
                 data_name: str,
                 root: str = 'data',
                 subset: str = '',
                 concept_labels: str = 'init_concept_labels.json',
                 test_labels: str = 'test_labels.json',
                 context_templates: str = 'multi_image_templates.json',
                 in_context: int = 1,
                 vary_context: int = 1,
                 image_size: int = 532,
                 augment_image: bool = False,
                 collapse_system_prompt: bool = False,
                 training: bool = True):
        
        self.name = data_name
        self.subset = subset
        self.training = training
        
        if training:
            self.data = pd.read_csv(f'{root}/{data_name}/data_train{subset}.csv')
            self.test_labels = {}  # No test labels for training data
        else:
            self.data = pd.read_csv(f'{root}/{data_name}/data_test{subset}.csv')
            if os.path.exists(f'{root}/{data_name}/{test_labels}'):
                self.test_labels = json.load(open(f'{root}/{data_name}/{test_labels}', 'r'))
            else:
                print(f"Warning: Test labels file {root}/{data_name}/{test_labels} not found. Proceeding without test labels.")
                self.test_labels = {}

        self.data_path = f'data/{data_name}'
        self.image_size = image_size
        self.image_path = self.data['image_file'].values
        self.concept_labels = json.load(open(f'{root}/{data_name}/{concept_labels}', 'r'))
        self.concepts = list(self.concept_labels[list(self.concept_labels.keys())[0]].keys())
        print("Concepts:", self.concepts)
        print(f"Number of concept labels: {len(self.concept_labels)}")
        print(f"Number of samples: {len(self.data)}")
        self.start_indices = np.where(np.isin(self.image_path, list(self.concept_labels.keys())))[0]

        if 'llama' in context_templates:
            self._format_image = self._format_image_llama
            self._format_prompt = self._format_prompt_llama
        else:
            self._format_image = self._format_image_multi
            self._format_prompt = self._format_prompt_multi

        self.augment_image = augment_image
        self.collapse_system_prompt = collapse_system_prompt
        self.in_context = in_context
        self.vary_context = vary_context

        if vary_context > 1 and in_context > 0:
            in_context_indices = list(combinations(range(len(self.concept_labels)), in_context))
            assert len(in_context_indices) >= vary_context, f"Not enough combinations for in-context examples: {len(in_context_indices)} < {vary_context}"
            self.in_context_indices = [in_context_indices[i] for i in np.random.choice(len(in_context_indices), vary_context, replace=False)]

        self.context = json.load(open(f'{self.data_path}/{context_templates}', 'r'))
    
    def _format_prompt_multi(self, in_context_examples):
        prompt = []
        system = ',\n'.join(f'{{"name" : "{concept}", "value": "..."}}' for concept in self.concepts)
        system = f'[\n{system}\n]'

        system_prompt = deepcopy(self.context['system'])
        system_prompt['content'][0]['text'] = system_prompt['content'][0]['text'].format(system)
        prompt.append(system_prompt)

        question = ', '.join(self.concepts[:-1]) + f', and {self.concepts[-1]}'

        for example in in_context_examples:
            question_prompt = deepcopy(self.context['question'])
            question_prompt['content'][1]['text'] = question_prompt['content'][1]['text'].format(len(self.concepts), question)
            prompt.append(question_prompt)
            answer = ',\n'.join([f'{{"name": "{concept}", "value": "{self.concept_labels[example][concept]}"}}' for concept in self.concepts])
            answer = f'[\n{answer}\n]'
            answer_prompt = deepcopy(self.context['answer'])
            answer_prompt['content'][0]['text'] = answer_prompt['content'][0]['text'].format(answer)
            prompt.append(answer_prompt)

        final_question_prompt  = deepcopy(self.context['question'])
        final_question_prompt['content'][1]['text'] = final_question_prompt['content'][1]['text'].format(len(self.concepts), question)
        prompt.append(final_question_prompt )
        assert len(prompt) == 2 + len(in_context_examples) * 2

        if self.collapse_system_prompt:
            system_prompt = prompt.pop(0) if prompt[0]['role'] == 'system' else None
            if system_prompt:
                prompt[0]['content'] = [{"type": "text", "text": system_prompt['content'][0]['text'] + "\n\n"}] + prompt[0]['content']

        return prompt
    
    def _format_prompt_llama(self, in_context_examples):
        prompt = []
        system = ',\n'.join(f'{{"name" : "{concept}", "value": "..."}}' for concept in self.concepts)
        system = f'[\n{system}\n]'

        system_prompt = deepcopy(self.context[f"system{len(in_context_examples)}"])
        system_prompt['content'] = system_prompt['content'].format(system)
        prompt.append(system_prompt)

        question = ', '.join(self.concepts[:-1]) + f', and {self.concepts[-1]}'
        positions = {0: [], 
                     1: ['LEFT panel', 'RIGHT panel'], 
                     3: ['TOP-LEFT panel', 'TOP-RIGHT panel', 'BOTTOM-LEFT panel', 'BOTTOM-RIGHT panel']}

        for example, pos in zip(in_context_examples, positions[len(in_context_examples)][:-1]):
            question_prompt = deepcopy(self.context['question'])
            question_prompt['content'][0]['text'] = question_prompt['content'][0]['text'].format(pos, len(self.concepts), question)
            if len(prompt) == 1:
                question_prompt['content'] = [{"type": "image"}] + question_prompt['content']
            prompt.append(question_prompt)

            answer = ',\n'.join([f'{{"name": "{concept}", "value": "{self.concept_labels[example][concept]}"}}' for concept in self.concepts])
            answer = f'[\n{answer}\n]'
            answer_prompt = deepcopy(self.context['answer'])
            answer_prompt['content'][0]['text'] = answer_prompt['content'][0]['text'].format(answer)
            prompt.append(answer_prompt)

        final_question_prompt  = deepcopy(self.context['question'])
        if len(in_context_examples) > 0:
            final_question_prompt['content'][0]['text'] = final_question_prompt['content'][0]['text'].format(positions[len(in_context_examples)][-1], len(self.concepts), question)
        else:
            final_question_prompt['content'][0]['text'] = final_question_prompt['content'][0]['text'].format('given image', len(self.concepts), question)
            final_question_prompt['content'] = [{"type": "image"}] + final_question_prompt['content']

        prompt.append(final_question_prompt)
        assert len(prompt) == 2 + len(in_context_examples) * 2

        if self.collapse_system_prompt:
            system_prompt = prompt.pop(0) if prompt[0]['role'] == 'system' else None
            if system_prompt:
                prompt[0]['content'] = [{"type": "text", "text": system_prompt['content'][0]['text'] + "\n\n"}] + prompt[0]['content']

        return prompt

    def _format_label(self, image_file):
        if image_file in self.concept_labels.keys():
            label = ',\n'.join([f'{{"name": "{concept}", "value": "{self.concept_labels[image_file][concept]}"}}' for concept in self.concepts])
            label = f'[\n{label}\n]'
        elif not self.training and image_file in self.test_labels.keys():
            label = ',\n'.join([f'{{"name": "{concept}", "value": "{self.test_labels[image_file][concept]}"}}' for concept in self.concepts])
            label = f'[\n{label}\n]'
        else:
            label = ''
            
        return label
    
    # add augmentation to the image
    def _transform_image(self, image, augment=False):
        if augment and self.name != 'oai':
            print("Applying augmentation to image")
            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.RandomRotation(degrees=35, expand=True), # Add rotation up to ±20 degrees
                T.RandomResizedCrop(self.image_size, scale=(0.8, 0.95), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(degrees=35),  # Add rotation up to ±35 degrees
                T.ColorJitter(brightness=0.05, contrast=0.2, saturation=0.2, hue=0.1)  # Very slight brightness (0.05)
            ])
        else:
            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
            ])
        return transform(image)
    
    def _load_image(self, path):
        if self.name == 'oai':
            # Load float32 knee crop: (H,W,3) or (3,H,W)
            arr = np.load(path).astype(np.float32, copy=False)

            # CHW -> HWC if needed
            if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                arr = np.transpose(arr, (1, 2, 0))  # -> (H,W,3)

            # Sanity check
            if not (arr.ndim == 3 and arr.shape[2] == 3):
                raise ValueError(f"Expected 3-channel array, got shape {arr.shape}")

            # Map to [0,1] (your npys are float32, either [0,1] or [-1,1])
            if np.nanmin(arr) < 0.0:               # handle [-1,1]
                arr = (arr + 1.0) / 2.0
            arr = np.clip(arr, 0.0, 1.0)

            # Convert to uint8 for PIL; let the model's processor apply mean/std later
            arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
            return Image.fromarray(arr8, mode='RGB')

        # Non-OAI: treat as a regular image file
        return Image.open(path).convert('RGB')

    def _format_image_multi(self, in_context_examples, image_path, bounding_box=None):
        images = []
        for example in in_context_examples:
            img = self._load_image(f'{self.data_path}/images/{example}')
            img = self._transform_image(img, augment=self.augment_image)
            images.append(img)

        final_image = self._load_image(f'{self.data_path}/images/{image_path}')
        final_image = self._transform_image(final_image, augment=self.augment_image)
        images.append(final_image)
                
        return images
    
    def _format_image_llama(self, in_context_examples, image_path):
        image = self._load_image(f'{self.data_path}/images/{image_path}')
        image = self._transform_image(image, augment=self.augment_image)

        if len(in_context_examples) == 1:
            new_width = self.image_size * 2
            new_height = self.image_size
            new_image = Image.new('RGB', (new_width, new_height))

            example = self._load_image(f'{self.data_path}/images/{in_context_examples[0]}')
            example = self._transform_image(example, augment=self.augment_image)
            new_image.paste(example, (0, 0))
            new_image.paste(image, (new_width // 2, 0))

        elif len(in_context_examples) == 3:
            new_width = self.image_size * 2
            new_height = self.image_size * 2
            new_image = Image.new('RGB', (new_width, new_height))

            img1 = self._load_image(f'{self.data_path}/images/{in_context_examples[0]}')
            img1 = self._transform_image(img1, augment=self.augment_image)
            img2 = self._load_image(f'{self.data_path}/images/{in_context_examples[1]}')
            img2 = self._transform_image(img2, augment=self.augment_image)
            img3 = self._load_image(f'{self.data_path}/images/{in_context_examples[2]}')
            img3 = self._transform_image(img3, augment=self.augment_image)

            new_image.paste(img1, (0, 0))
            new_image.paste(img2, (new_width // 2, 0))
            new_image.paste(img3, (0, new_height // 2))
            new_image.paste(image, (new_width // 2, new_height // 2))
        else:
            new_image = image

        # new_image.save(os.path.join('llama_images', image_path.split('/')[-1].replace('.npy', '.jpg')))
        return [new_image]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = []
        prompt = []
        image_file = []
        label = []

        if self.vary_context > 1 and self.in_context > 0:
            for indices in self.in_context_indices:
                in_context_examples = [list(self.concept_labels.keys())[i] for i in indices]
                image.append(self._format_image(in_context_examples, self.image_path[idx]))
                prompt.append(self._format_prompt(in_context_examples))
                image_file.append(self.image_path[idx])
                label.append(self._format_label(self.image_path[idx]))
        else:
            for i in range(self.vary_context):
                in_context_examples = np.random.choice(list(self.concept_labels.keys()), self.in_context, replace=False)
                # image list (nhat)
                image.append(self._format_image(in_context_examples, self.image_path[idx]))
                # end image list
                prompt.append(self._format_prompt(in_context_examples))
                image_file.append(self.image_path[idx])
                label.append(self._format_label(self.image_path[idx]))

        return {
            "image": image,
            "prompt": prompt,
            "image_file": image_file,
            "label": label
        }
    
def collate_fn(batch):
    images = [image for item in batch for image in item['image']]
    prompts = [prompt for item in batch for prompt in item['prompt']]
    image_files = [image_file for item in batch for image_file in item['image_file']]
    labels = [label for item in batch for label in item['label']]
    return images, prompts, image_files, labels

class ActiveLearningDataset(object):
    def __init__(self,
                 data: Dataset):
        
        self.data = data
        self.train_mask = np.full(len(self.data,), False)
        self.pool_mask = np.full(len(self.data,), True)

        if self.data.concept_labels is not None:
            start_indices = np.where(np.isin(self.data.image_path, list(self.data.concept_labels.keys())))[0]
            self.train_mask[start_indices] = True
            self.pool_mask[start_indices] = False

        self.train_data = Subset(self.data, None)
        self.pool_data = Subset(self.data, None)

        self._update_indices()

    def _update_indices(self):
        self.train_data.indices = np.nonzero(self.train_mask)[0]
        self.pool_data.indices = np.nonzero(self.pool_mask)[0]

    def _save_data(self, indices, output_dir="data_train.csv"):
        indices = np.asarray(indices)

    @property
    def acquired_indices(self):
        return self.train_data.indices

    def is_empty(self):
        return len(self.pool_data) == 0

    def get_random_pool_indices(self, size):
        assert 0 <= size 
        if size <= len(self.pool_data):
            pool_indices = torch.randperm(len(self.pool_data))[:size]
        else:
            pool_indices = torch.randperm(len(self.pool_data))

        return pool_indices

    def get_pool_indices(self, pool_indices):
        """Transform indices (in `pool_data`) to indices in the original `data`."""
        dataset_indices = self.pool_data.indices[pool_indices]
        return dataset_indices
         
    def extract_dataset_from_pool(self, size):
        """Extract a dataset randomly from the pool dataset and make those indices unavailable.
        Useful for extracting a validation set."""
        
        pool_indices = self.get_random_pool_indices(size)
        dataset_indices = self.get_pool_indices(pool_indices)
        self.pool_mask[dataset_indices] = False
        self._update_indices()
        return Subset(self.data, dataset_indices)
    
    def remove(self, indices, pool=True):
        """Remove elements from the pool data."""
        if pool:
            indices = self.get_pool_indices(indices)

        self.pool_mask[indices] = False
        self._update_indices()
    
    def acquire(self, indices, pool=True):
        """Acquire elements from the pool data into the training data."""

        if pool:
            indices = self.get_pool_indices(indices)

        self.train_mask[indices] = True
        self.pool_mask[indices] = False

        self._update_indices()

class ClassificationDataset(Dataset):
    def __init__(
        self,
        data_name: str,
        root: str,
        training: bool = True,
        preprocess = None,
        crop = False,
        concept_order: Optional[List[str]] = None
    ):
        self.data_path = f"{root}/{data_name}"
        self.name = data_name
        csv_path = f"{self.data_path}/train_clusters.csv" if training else f"{self.data_path}/test_clusters.csv"
        self.data = pd.read_csv(csv_path)
        self.preprocess = preprocess
        self.crop = crop
        self.skip_out_of_range = True  # silently ignore out-of-range concept indices
        self.training = training

        # Ensure label column exists and is named 'label'
        if "label" not in self.data.columns:
            raise ValueError(f"'label' column not found in {csv_path}.")
        # (keep as 'label'—no renaming—so your downstream code stays consistent)

        # Load concept exemplars to define concept names and per-concept cardinalities
        with open(f"{root}/{data_name}/concept_exemplars.json", "r") as f:
            exemplars = OrderedDict(json.load(f).items())

        if concept_order is not None:
            missing = [c for c in concept_order if c not in exemplars]
            extra = [c for c in exemplars if c not in concept_order]
            if missing or extra:
                raise ValueError(
                    "concept_order mismatch.\n"
                    f"Missing in exemplars: {missing}\n"
                    f"Extra not listed in concept_order: {extra}"
                )
            exemplars = OrderedDict((c, exemplars[c]) for c in concept_order)

        self.concept_names: List[str] = list(exemplars.keys())

        # Verify CSV has a column for each concept
        missing_cols = [c for c in self.concept_names if c not in self.data.columns]
        if missing_cols:
            raise ValueError(
                "Missing concept columns in clusters CSV: " + ", ".join(missing_cols)
            )

        # Per-concept sizes + global offsets
        self.concept_sizes: List[int] = []
        for c in self.concept_names:
            size = len(exemplars[c])
            if size <= 0:
                raise ValueError(f"Concept '{c}' has empty index space.")
            self.concept_sizes.append(size)

        self.offsets: List[int] = []
        curr = 0
        for sz in self.concept_sizes:
            self.offsets.append(curr)
            curr += sz
        self.total_dim = curr

    def __len__(self):
        return len(self.data)

    def _load_image(self, path):
        if self.name == 'oai':
            # Load float32 knee crop: (H,W,3) or (3,H,W)
            arr = np.load(path).astype(np.float32, copy=False)

            # CHW -> HWC if needed
            if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                arr = np.transpose(arr, (1, 2, 0))  # -> (H,W,3)

            # Sanity check
            if not (arr.ndim == 3 and arr.shape[2] == 3):
                raise ValueError(f"Expected 3-channel array, got shape {arr.shape}")

            # Map to [0,1] (your npys are float32, either [0,1] or [-1,1])
            if np.nanmin(arr) < 0.0:               # handle [-1,1]
                arr = (arr + 1.0) / 2.0
            arr = np.clip(arr, 0.0, 1.0)

            # Convert to uint8 for PIL; let the model's processor apply mean/std later
            arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
            return Image.fromarray(arr8, mode='RGB')

        # Non-OAI: treat as a regular image file
        return Image.open(path).convert('RGB')
    
    def _transform_image(self, image, augment=False):
        if augment:
            transform = T.Compose([
                # T.RandomHorizontalFlip(),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
                # T.RandomRotation(degrees=10),
                # T.RandomAffine(degrees=0, translate=(0.05, 0.05))
            ])
        else:
            transform = T.Compose([])  # Identity transform - returns image unchanged
        return transform(image)

    # def __getitem__(self, idx: int):
    #     row = self.data.iloc[idx]

    #     image = self._load_image(f'data/{self.name}/images/{row["image"]}')
    #     image = self._transform_image(image, augment=self.training)
        
    #     if self.preprocess:
    #         image = self.preprocess(image)
            
    #     # Build flattened multi-hot vector
    #     concept_one_hot = torch.zeros(self.total_dim, dtype=torch.float)
    #     for ci, cname in enumerate(self.concept_names):
    #         concept_idx = int(row[cname])
    #         if 0 <= concept_idx < self.concept_sizes[ci]:
    #             pos = self.offsets[ci] + concept_idx
    #             concept_one_hot[pos] = 1.0
    #         else:
    #             if self.skip_out_of_range:
    #                 # silently ignore bad indices
    #                 pass
    #             else:
    #                 raise IndexError(
    #                     f"{cname} index {concept_idx} out of range [0, {self.concept_sizes[ci]}) at row {idx}"
    #                 )
    #     target = torch.tensor(int(row["label"]), dtype=torch.long)
    #     return image, concept_one_hot, target

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        image = self._load_image(f'data/{self.name}/images/{row["image"]}')
        image = self._transform_image(image, augment=self.training)

        if self.preprocess:
            image = self.preprocess(image)

        # per-attribute concept indices (LongTensor [num_attrs])
        concept_indices = torch.empty(len(self.concept_names), dtype=torch.long)
        for ci, cname in enumerate(self.concept_names):
            concept_idx = int(row[cname])
            if 0 <= concept_idx < self.concept_sizes[ci]:
                concept_indices[ci] = concept_idx
            else:
                if self.skip_out_of_range:
                    # replace invalid with a safe default (commonly 0)
                    concept_indices[ci] = int(self.default_oob_value)
                else:
                    raise IndexError(
                        f"{cname} index {concept_idx} out of range [0, {self.concept_sizes[ci]}) at row {idx}"
                    )

        target = torch.tensor(int(row["label"]), dtype=torch.long)

        # Return format that matches the ConceptAlignment training:
        # (image, label, concept_indices)
        return image, target, concept_indices

if __name__ == '__main__':
    dataset = ConceptDataset(data_name='cub200',
                             root='data',
                             concept_labels='init_concept_labels.json',
                             context_templates='default_templates.json',
                             in_context=3, 
                             vary_context=2,
                             image_size=280,
                             collapse_system_prompt=False)

    print(f"Number of samples in dataset: {len(dataset)}")
    active_learning_dataset = ActiveLearningDataset(dataset)
    active_learning_subset = active_learning_dataset.extract_dataset_from_pool(2)
    print(f"Number of samples in pool dataset: {len(active_learning_dataset.pool_data)}")
    print(f"Number of samples in training dataset: {len(active_learning_dataset.train_data)}")
    train_loader = DataLoader(active_learning_subset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    # processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", padding_side="left")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", padding_side="left")
    # processor = AutoProcessor.from_pretrained("AdaptLLM/biomed-Llama-3.2-11B-Vision-Instruct", padding_side="left")
    # ids = [processor.tokenizer.convert_tokens_to_ids(s) for s in ["0", "1", "2", "3"]]
    # print(f"Token IDs: {ids}")

    for batch_idx, (images, prompts, image_files, labels) in enumerate(train_loader):
        # print(len(images))
        # print(len(prompts))
        prompts = processor.apply_chat_template(prompts, add_generation_prompt=True)
        with open('test.txt', 'w') as f:
            f.write(f"Prompt: {prompts[0]}\n")
        inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True, truncation=True)
