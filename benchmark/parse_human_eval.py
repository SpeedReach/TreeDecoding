import os
import json

in_folder = "out/tree"
out_folder = "out/human_eval"

all_ids = set()
for i in range(164):
    all_ids.add(f"HumanEval/{i}")

for filename in os.listdir(in_folder):
    file_path = os.path.join(in_folder, filename)
        
    # Check if it's a file (not a directory)
    if not os.path.isfile(file_path):
        continue
    with open(file_path, 'r', encoding='utf-8') as file:
        with open(os.path.join(out_folder, filename), 'w') as out_file:
            all_id_copy = all_ids.copy()
            for line in file:
                data = json.loads(line)
                out_file.write(json.dumps({
                    "task_id": data['id'],
                    "completion": data['output']
                }) + "\n")
                all_id_copy.remove(data['id'])
            for id in all_id_copy:
                out_file.write(json.dumps({
                    "task_id": id,
                    "completion": ""
                }) + "\n")
        
