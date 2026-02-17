#intro to transformer based LLM

Initialy, wanted to make encoder-decoder model for multilingualGPT(ability to translate languages)
but due to time constraints, changed it to decoder only model.

### encoding/decoding
used BPE instead of unigram LM tokenization algorithm due to my personal choices, although unigram is better for multuiilingual works, BPE is faster, and is more suitable in my current condition of my personal computer, Chat gpt also used BPE algorithm.
Unigram can also cause inaccuracy for predicting of what comes next.


## data

downloaded openwebtextcorpus data from hugginface, to allow more efficient and faster speed training. changed file type to .arrow files



## the strucutre of files

data.py processed the data

demo folder contains prototype code experimented with jupyter notebook

tokenizer_train.py contains code for traniing settigns

my model:
current train loss is at 1.6054, and val loss of 1.6489
and has gone through 3500 iterations of trainig
loss.item has value of 1.509182333946228
output: given a prompt, generates nonsense. but correctly has a structure of a sentence/paragraph.




# instructions

before doing any of these, you require a dataset.
after training the model by running `python training.py`
each time you run python .\training.py , it will train the model 1000 iterations saving every 500 iterations into checkpoint.pth.
run `python chatbot.py`
you can change prompt as you like.