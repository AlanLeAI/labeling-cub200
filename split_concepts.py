import json
import os
import re

train_samples = json.load(open('train_samples.json', 'r'))
test_samples = json.load(open('test_samples.json', 'r'))

concept_names = ["body shape",
                "crest shape",
                "beak shape",
                "neck shape",
                "wing shape",
                "tail shape",
                "flight feather shape",
                "feet shape",
                "crown color and marking",
                "beak color and marking",
                "eye color and marking",
                "face color and marking",
                "throat color and marking",
                "upper chest color and marking",
                "lower belly color and marking",
                "upside tail color and marking",
                "upside wing color and marking",
                "feet color"]

marking_names = ['stripe', 'stripes', 'striping',
                 'spot', 'spots', 'spotting',
                 'patch', 'patches',
                 'bar', 'bars',
                 'ring', 'rings',
                 'line', 'lines',
                 'band', 'bands',
                 'side', 'sides',
                 'flank', 'flanks',
                 'sheen', 'hue',
                 'tinge', 'tinges',
                 'wash',
                 'mask',
                 'base',
                 'mottling',
                 'pouch',
                 'speckle', 'speckles',
                 'horn', 'horns',
                 'tip', 'tips',
                 'edge', 'edges',
                 'streak', 'streaks', 'streaking']

def split_concepts(samples):
    new_samples = {}

    for image in samples:
        new_sample = {}
        for concept in concept_names:
            if 'color and marking' in concept:
                # color and marking are separated by "with"
                color_key = concept.replace('color and marking', 'color')
                marking_key = concept.replace('color and marking', 'marking')

                if 'with' in samples[image][concept]:
                    color, marking = samples[image][concept].split('with')
                    # remove any leading/trailing whitespace and commas
                    color = color.strip().strip(',')
                    marking = marking.strip().strip(',')
                    part = concept.replace('color and marking', '').strip()
                    # remove 'upper', 'lower', 'upside' and any trailing spaces
                    part = re.sub(r'\b(upper|lower|upside)\b', '', part).strip()
                    for m in marking_names:
                        marking = re.sub(r'\b' + re.escape(m) + r'\b', f'{part} {m}', marking)
                    new_sample[color_key] = color
                    new_sample[marking_key] = marking
                elif 'not clearly visible' in samples[image][concept]:
                    new_sample[color_key] = f'not clearly visible {color_key}'
                    new_sample[marking_key] = f'not clearly visible {marking_key}'
                else:
                    color = samples[image][concept].strip().strip(',')
                    new_sample[color_key] = color
                    new_sample[marking_key] = f'no {marking_key}'
            else:
                if 'not clearly visible' in samples[image][concept]:
                    new_sample[concept] = f'not clearly visible {concept}'
                else:
                    new_sample[concept] = samples[image][concept]

        new_samples[image] = new_sample
    
    return new_samples

new_train_samples = split_concepts(train_samples)
new_test_samples = split_concepts(test_samples)

os.rename('train_samples.json', 'train_samples_old.json')
json.dump(new_train_samples, open('train_samples.json', 'w'), indent=2)
os.rename('test_samples.json', 'test_samples_old.json')
json.dump(new_test_samples, open('test_samples.json', 'w'), indent=2)

concept_labels = json.load(open('concept_labels.json', 'r'))
new_concept_labels = split_concepts(concept_labels)
os.rename('concept_labels.json', 'concept_labels_old.json')
json.dump(new_concept_labels, open('concept_labels.json', 'w'), indent=2)