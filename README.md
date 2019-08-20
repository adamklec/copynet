# CopyNet

This is an implementation of CopyNet https://arxiv.org/abs/1603.06393
CopyNet extends the functionality of encoder-decoder models to allow the generation
of output sequences that contain "out of vocabulary" tokens that appeared in the input sequence.

Dependencies:
pytorch
numpy
tensorboardX (for logging)
tqmd (for logging)
spacy (for tokenization)

The model is trained on sequence pairs. Create a directory to hold training files. Each file should have 2 lines of text.
The first is the input sequence, the second is the target output sequnce.
The tokens in each sequence should be seperated by spaces.
I used spacy to tokenize the training data so the SequencePairDataset class as well as the evaluation methods assume that spacy will be used.
If you want to use a different tokenizer be sure to update those files accordingly.

Train the model using the train.py script. Most hyperparameters can be tuned with command line arguments documented in the training script.

Unfortunately I cannot share the training data that I used to develop this model implementation. I did it using a proprietary data set that I no longer have access to. As it is now, you must provide your own training data in order for this repo to function. I would welcome PRs to enable a demo on an open source data set. 
