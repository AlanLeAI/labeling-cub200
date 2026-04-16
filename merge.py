import json

merged = {}

for k in range(1, 5):
    with open(f'test_samples{k}.json', 'r') as f:
        data = json.load(f)
        merged.update(data)

print(len(merged))
with open('test_samples.json', 'w') as f:
    json.dump(merged, f, indent=2)