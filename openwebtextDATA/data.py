# import os 
# from tqdm import tqdm
# from datasets import load_from_disk


# dataset = load_from_disk("")

# folder_path = "C:\\Users\\Glbap\\OneDrive\\Documents\\visualStudio\\Projects\\LLM\\openwebtextD"
# output_file_train = "output_train.txt"
# output_file_val = "output_val.txt"
# vocab_file = "vocab.txt"

# files = load_from_disk(folder_path)
# total_files = len(files)

# split_index = int(total_files * 0.9)
# files_train = files[:split_index]
# files_val = files[split_index:]

# vocab = set()

# with open(output_file_train, "w", encoding="utf-8") as outfile:
#     for filename in tqdm(files_train, total=len(files_train)):
#         file_path = os.path.join(folder_path, filename)
#         with lzma.open(file_path, "rt", encoding="utf-8") as infile:
#             text = infile.read()
#             outfile.write(text)
#             characters = set(text)
#             vocab.update(characters)

# with open(output_file_val, "w", encoding="utf-8") as outfile:
#     for filename in tqdm(files_val, total=len(files_val)):
#         file_path = os.path.join(folder_path, filename)
#         with lzma.open(file_path, "rt", encoding="utf-8") as infile:
#             text = infile.read()
#             outfile.write(text)
#             characters = set(text)
#             vocab.update(characters)

# with open(vocab_file, "w", encoding="utf-8") as vfile:
#     for char in vocab:
#         vfile.write(char + '\n')


from datasets import load_from_disk
from tqdm import tqdm
import os

# Paths
dataset_path = "C:\\Users\\Glbap\\OneDrive\\Documents\\visualStudio\\Projects\\LLM\\openwebtextD"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

# Load dataset (reads only the .arrow files)
dataset = load_from_disk(dataset_path)

# Shuffle and split into train/val
dataset = dataset.shuffle(seed=42)
split_index = int(len(dataset) * 0.9)
train_data = dataset.select(range(split_index))
val_data = dataset.select(range(split_index, len(dataset)))

# Build vocab and write train split
vocab = set()
with open(output_file_train, "w", encoding="utf-8") as f:
    for example in tqdm(train_data, desc="Writing train data"):
        text = example["text"].strip()
        f.write(text + "\n")
        vocab.update(text)

# Write val split
with open(output_file_val, "w", encoding="utf-8") as f:
    for example in tqdm(val_data, desc="Writing val data"):
        text = example["text"].strip()
        f.write(text + "\n")
        vocab.update(text)

# Write character-level vocab
with open(vocab_file, "w", encoding="utf-8") as f:
    for char in sorted(vocab):
        f.write(char + "\n")






# print("starting")
# custom_dir = "C:\\Users\\Glbap\\OneDrive\\Documents\\visualStudio\\Projects\\LLM\\openwebtextD"
# dataset = load_dataset("openwebtext", split = "train", trust_remote_code=True)
# dataset.save_to_disk(custom_dir)
# print(f"Openwebtext save at: {custom_dir}")
# print(len(dataset))
