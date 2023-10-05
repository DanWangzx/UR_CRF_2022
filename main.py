import time
import torch
import math
from torch.utils.data import DataLoader, Dataset
from torch import logsumexp, optim
DATA_PATH =  r"C:\Users\danny\Desktop\Delivery"
dec_word, enc_word, word_count = {}, {}, 1   #encoding_dict: {index:word}, etc. 
dec_label, enc_label, label_count = {}, {}, 0  #decoding_dict : {word:index}, etc. 
weights_prev = None #
gold_trans, gold_emis = None, None

# load features into a tensor of [feature_emis, feature_trans]
class CorpusSet(Dataset): # initializing the dataset corpus with forms compatible with a Dataloader in PyTorch
    def __init__(self, filename, max_string_len):
        word_tensors, tag_tensors = self.dataPrep(filename, max_string_len)
        word_tensors, tag_tensors = word_tensors.long(), tag_tensors.long()
        self.samples = [[word_tensors[i], tag_tensors[i]] for i in range(len(word_tensors))]
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    
    def dataPrep(self, filename, max_string_len): # preprocessing the data to retrive into tensor space
        with open(filename, 'r') as f: 
            corpus = f.readlines()
            corp_len = len(corpus)
            word_tensor = torch.zeros(corp_len, max_string_len)
            tag_tensor = torch.zeros(corp_len, max_string_len)
            for i in range(len(corpus)):
                tokens = corpus[i].split()
                seq, strings, tags = tokens[0], tokens[1::2], tokens[2::2]
                word_tensor[i], tag_tensor[i] = self.feature_fill(strings, tags, max_string_len)
            return word_tensor, tag_tensor

    def feature_fill(self, strings, tags, max_string_len):
        w, t = torch.zeros(max_string_len), torch.zeros(max_string_len)
        for i in range(min(len(strings), max_string_len)):
            w[i] = enc_word.get(strings[i], 0)
            t[i] = enc_label[tags[i]]
        return w, t


def parse_weight_line(line): #the weights-parsing module that enables the golden weights
    tokens = line.split()
    string, score = tokens[0], tokens[1]
    keys = string.split('_')
    if keys[0] == 'E':
        return keys[2], keys[1], score, 'emission'
    else:
        return keys[1], keys[2], score, 'transition'

def parse_weights(filename): # function to read in the weights file into two hashmaps of E and T, this used for validaion only
    global gold_emis
    global gold_trans
    with open(filename, 'r') as f:
        # table=defaultdict({})
        # table[x][y] = 1

        gold_emis, gold_trans = torch.zeros(len(enc_word)+1, len(enc_label)), torch.zeros(len(enc_label), len(enc_label))
        for line in f:
            x, y, score, type = parse_weight_line(line)
            if type == 'emission':
                gold_emis[enc_word[x]][enc_label[y]] = float(score)
            else:
                gold_trans[enc_label[y]][enc_label[x]] = float(score)

        return gold_emis, gold_trans #dict, tensor


class CRF(torch.nn.Module):
    def __init__(self, weights_filename):
        self.feature_space_engineering(weights_filename) # gaining the feature representations from the weights file
        emis, trans = torch.nn.Parameter(torch.zeros(len(enc_word)+1, len(enc_label))), torch.nn.Parameter(torch.zeros(len(enc_label), len(enc_label)))

        self.weights = [emis, trans]

    def feature_space_engineering(self, filename): # should take in the train.weights file and produce the features
        global dec_word, enc_word, word_count
        global dec_label, enc_label, label_count
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.split() 
                string = tokens[0]
                keys = string.split('_')
                if keys[0] == 'E':
                    if keys[1] not in enc_label:
                        enc_label[keys[1]] = label_count
                        label_count += 1
                    if keys[2] not in enc_word:
                        enc_word[keys[2]] = word_count
                        word_count += 1
                else:
                    dec_word = {v:k for k, v in enc_word.items()}
                    dec_label = {v:k for k, v in enc_label.items()}
                    return
    
    def batch_train(self, batched_corpus, args): # major trainer function, enabled batch train
        for i in range(args.epoch):
            if args.SGD:
                optimizer = optim.SGD(self.weights, lr=args.lr)
            else:
                optimizer = optim.Adam(self.weights, lr=args.lr)
            count = 0

            for words, tags in batched_corpus:
                optimizer.zero_grad()
                output = self.forward(words)
                loss = self.loss_fn(output, words, tags)
                loss.backward()
                optimizer.step()
                count += args.batch
                print(f'epoch: {i}, batch size of {len(words)} is processed, total {count} sequences done')

                score = self.testing([words, tags], key=False)
                print(f'accuracy:{score}')
                score_gold = self.testing([words, tags], key=True)
                print(f'gold accuracy:{score_gold}')
        
            print(f'epoch: {i} complete')
        print('training procedure finished')
        return 

    def forward(self, words): # the log_Z computed by alpha table 
        alphas_h = torch.zeros(len(words), len(words[0]), len(enc_label))
        alphas_h[:, 0] = self.weights[0][words[:, 0], :]
        for i in range(1, len(words[0])):
            alphas_h[:, i] = torch.logsumexp(self.weights[0][words[:,i], :, None]+self.weights[1][None, None, :, :]+alphas_h[:, i-1, None, :], dim=3)
        
        log_Z = torch.logsumexp(alphas_h[:, -1, :], dim=1).sum(0)
        return log_Z

    def loss_fn(self, output, words, tags_tensor): # loss function by output - W^T*Phi 
        w_phi = self.feature_score(words,tags_tensor)
        return output - w_phi

    def feature_score(self, words, tags): # the W^T*phi score, we passed a two lists of indices (word_indices and tag_indices) to retrive a list of numbers, with gather function.
        w = torch.zeros(len(words))
        w[0] = (self.weights[0][words[:, 0]].clone().gather(1, tags[:,0].clone().unsqueeze(1))).sum(0) 
        for i in range(1, len(words[0])):
            w[i] = (self.weights[0][words[:, i]].clone().gather(1, tags[:,i].clone().unsqueeze(1)) + self.weights[1][tags[:, i]].clone().gather(1, tags[:, i-1].clone().unsqueeze(1))).sum(0)

        return w.sum(0)

# testing modules:
    # if key, the module would use the golden weights to predict the tokens
    def testing(self, full_corpus, key=False):
        correct, total = 0, 0
        word_samples, tag_samples = full_corpus
        for j in range(len(word_samples)):
            words, tags = word_samples[j], tag_samples[j]
            # remove zeros
            for i in range(len(words)-1, -1, -1):
                if words[i] == 0:
                    continue
                else:
                    words = words[:i+1]
                    tags = tags[:i+1]
                    break

            vit_seq = self.vit_seq(words, mode=key)
            for i in range(len(vit_seq[0])):
                if vit_seq[0][i] == tags[i]:
                    correct += 1
            total += len(words)
        #print(f'batch size of {len(word_samples)} processed, score is {correct/total}')
        return correct/total
    
    def vit_seq(self, words, mode=False): # unbatched viterbi
        emis = self.weights[0]
        trans = self.weights[1]
        if mode:
            emis, trans = gold_emis, gold_trans
        alpha, tracking = torch.zeros(len(words), len(enc_label)), torch.zeros(len(words), len(enc_label))
        alpha[0] = emis[words[0]]
        for i in range(1, len(words)):
            alpha[i], tracking[i] = (emis[words[i]][:, None]+trans[None, :, :]+alpha[i-1, None, :]).max(2)

        score, index = alpha[-1].max(0)

        index = int(index)
        res = []
        res.append(index)
        #print(index)
        for i in range(len(words)-1, 0, -1):
            back_by_one = int(tracking[i][index])
            res.append(back_by_one)
            index = back_by_one
        res.reverse()

        return res, score

def main():
    import argparse
    import os
    
    x = time.time()
    parser = argparse.ArgumentParser(description='Basic CRF algorithm on HMM-like data')
    parser.add_argument('--lr', type=float, default=1, help='Learning rate to use for update in training loop.')
    parser.add_argument('--epoch', type=int, default=1, help='the number of epochs in training.')
    parser.add_argument('--train', type=str, default=os.path.join(DATA_PATH,'train'), help='Training data file.')
    parser.add_argument('--dev', type=str, default=os.path.join(DATA_PATH,'dev'), help='Dev data file.')
    parser.add_argument('--test', type=str, default=os.path.join(DATA_PATH,'test'), help='Test data file.')
    parser.add_argument('--weights', type=str, default=os.path.join(DATA_PATH,'train.weights'), help='weights data file.')
    parser.add_argument('--SGD', action='store_true', default=False, help='If provided, it will use the sgd optimizer, otherwise would use Adam')
    parser.add_argument('--batch', type=int, default=1024, help='the number of epochs in training.')

    args = parser.parse_args()

    crf = CRF(args.weights)
    data = CorpusSet(args.train, 50)
    #print(f'initializtaion complete within {time.time()-x} seconds')
    dataloader = DataLoader(data, batch_size=args.batch, shuffle=True)
    parse_weights(args.weights)    
    crf.batch_train(dataloader, args)

    print('algorithm complete')
if __name__ == '__main__':
    main()
