#vocab size = 32k

import sentencepiece as spm

#dataset_path = "C:\\Users\\Glbap\\OneDrive\\Documents\\visualStudio\\Projects\\MultilingualGPT\\openwebtext"
dataset_path = "train_split.txt"




spm.SentencePieceTrainer.train(
    input=dataset_path,
    model_prefix='spm_model',
    vocab_size=32000,  # or any size like 32k, 16k
    model_type='bpe',  # or unigram, word, char
    character_coverage = 1,
    input_sentence_size = 100000, 
    shuffle_input_sentence=True,
    # normalization_rule_name='identity',
    byte_fallback=True,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3
)
print('Traninig finished')