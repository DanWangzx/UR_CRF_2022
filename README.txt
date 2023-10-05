This is an implemention of linear-chain CRF on csug dataset. The best accuracy achieved is 96.11 with Adam optimizer and an efficient batch of 1024. This is slightly better than the result of 95.48 by gold weights from train.weights. 

To access our best result (Adam + 1024 batch), simply do:
	python main.py 

To custom train, please do:
	python main.py --SGD (if not provided, the default optimizer is Adam) --lr [float] --epoch [int] --batch [int]

General code overview:
This implementation used the pytorch DataLoader class to build and access the batches. The general input is a parallel tensor with dim of [batch_id, sequence_max_len], with the values mapping to either word_id or tag_id (which could be later retrieved with encoder/decoder dictionary). We initialize our weights as two tensors corresponding to the transition and emission matrices. To increase robustness, we did not learn all possible features from the training example, as there will be always a new word that could not be recognized by the model. We ended up simply using the features from the train.weights file (but of course we didn't plug in the number from the file lol).

The loss function composed of 1) the feature_score, which is basically summing up the weights of all applicable features within a batch, 2) the expected output, which is computed via building up an alpha-table and taking the log_sum_exp over the last position. Because we are doing gradient descent, the Loss function is set to "expected output - feature_score" 

To evaluate the data, we used a similar viterbo-decoder that takes the max over the alpha table and backtrack the indices as the result. Since we used padding to uniform the input tensors, during evaluation step all ending-zeros were dropped. We end up doing a token-wise accuracy for all sequences. 

Experiments:
Both adam and SGD optimizer has reached good result on training even just one epoch. With a batch of 1024, Adam would converge to 95-ish within just a few batches (30% of the sequences). The SGD, with a default learning rate of 1, would gradually converge to 88 with the same setup. The result shows that this is a valid implementation of linear-chain CRF.
