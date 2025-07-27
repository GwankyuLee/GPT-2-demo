
# import sentencepiece as spm



# sp = spm.SentencePieceProcessor()
# sp.load("openwebtextDATA\\spm_model.model")

# vocab_size = sp.get_piece_size()


# encode = lambda s: sp.encode(s, out_type=int)
# decode = lambda ids: sp.decode(ids)


# text = "This is a sample sentence."
# tokens = sp.encode(text, out_type=str)
# token_ids = sp.encode(text, out_type=int)
# print("Tokens:", tokens)
# print("Token IDs:", token_ids)
# print("Decoded:", sp.decode(token_ids))

import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())