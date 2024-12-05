import os
import json

in_folder = "out/tree"
out_folder = "out/human_eval"

for filename in os.listdir(in_folder):
    file_path = os.path.join(in_folder, filename)
        
    # Check if it's a file (not a directory)
    if not os.path.isfile(file_path):
        continue
    with open(file_path, 'r', encoding='utf-8') as file:
        with open(os.path.join(out_folder, filename), 'w') as out_file:
            for line in file:
                data = json.loads(line)
                print(data.keys())
                out_file.write(json.dumps({
                    "task_id": data['id'],
                    "completion": data['output']
                }) + "\n")
        
