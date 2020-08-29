import torch
import os
import numpy as np
import torch.nn.functional as F
import pickle
import tensorflow as tf
from google.protobuf import text_format
from . import data_utils
from multiprocessing import Pool

class CalcSentProbsModel:
    def __init__(self,model_name):
        self.model_name = model_name

    def load_model(self):
        if self.model_name == 'lstm':
            with open('hidden650_batch128_dropout0.2_lr20.0.pt', 'rb') as f:
                ##Our current code only works with CPU##
                self.model = torch.load(f, map_location=torch.device('cpu'))
            self.model.eval()
            with open('word2id.pkl', 'rb') as f:
                self.word2id = pickle.load(f)
            with open('id2word.pkl', 'rb') as f:
                self.id2word = pickle.load(f)
        elif self.model_name == 'lstm-large':
            FLAGS = tf.compat.v1.flags.FLAGS
            # General flags.
            tf.compat.v1.flags.DEFINE_string('pbtxt', '',
                                         'GraphDef proto text file used to construct model '
                                         'structure.')
            tf.compat.v1.flags.DEFINE_string('ckpt', '','Checkpoint directory used to fill model values.')
            tf.compat.v1.flags.DEFINE_string('vocab_file', '', 'Vocabulary file.')
            
            self.gd_file = FLAGS.pbtxt
            self.ckpt_file = FLAGS.ckpt
            self.vocab_file = FLAGS.vocab_file

        elif self.model_name == 'gpt2' or self.model_name == 'gpt2-large':
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            if torch.cuda.is_available():
                self.model.to('cuda')
            self.model.eval()
        elif self.model_name == 'bert':
            from transformers import BertTokenizer, BertModel, BertForMaskedLM
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.model.eval()
            self.mask_id = self.tokenizer.encode("[MASK]")[1:-1][0]
        elif self.model_name == 'ngram':
            import kenlm
            self.model = kenlm.Model('ngram.arpa')
                
        else:
            print("Invalid model name")
            exit()

    def calculate_sent_probs(self,sent_list,max_sent_len=20,batch_size=25):
        if self.model_name == 'lstm':
            sent_prob = np.zeros(len(sent_list))
            for i,sentence in enumerate(sent_list):
                sent_data = torch.tensor([self.word2id[word] if word in self.word2id else self.word2id['<unk>'] for word in (['<eos>']+sentence.split(" ")+['.'])])
                init_state = self.model.init_hidden(1)
                hid_state = init_state
                prob_list = np.zeros((len(sent_data)-1))
                for j, word in enumerate(sent_data[:-1]):
                    output,hid_state =  self.model(word.unsqueeze(0).unsqueeze(0),hid_state)
                    log_probs = torch.log(F.softmax(output[0][0],dim=0))
                    prob_list[j] = log_probs[sent_data[j+1].item()]
                sent_prob[i] = np.sum(prob_list)
            return sent_prob

        elif self.model_name == 'lstm-large':
            self.BATCH_SIZE = 1
            self.NUM_TIMESTEPS = 1
            self.MAX_WORD_LEN = 50
            sent_prob = np.zeros(len(sent_list))
            arg = [(i,sentence) for i,sentence in enumerate(sent_list)]
            ##This is a memory-consuming process, so the thread number should be moderate##
            with Pool(processes=10) as p:
                result_list = p.starmap(self.calc_sent_prob_LSTM_large,arg)
            for i in range(len(sent_list)):
                sent_prob[i] = - result_list[i]
            return sent_prob

        elif self.model_name == 'gpt2' or self.model_name == 'gpt2-large':
            sent_prob = np.zeros(len(sent_list))
            if torch.cuda.is_available():
                eos_list = ['.' for sentence in sent_list]
                padded_sent_list,eos_pos_list = self.padding_GPT2(sent_list,eos_list,max_sent_len)
                padded_sent_list = self.batchify(padded_sent_list,batch_size)
                eos_pos_list = self.batchify(eos_pos_list,batch_size)
                for i,sentence_batch in enumerate(padded_sent_list):
                    outputs = self.model(sentence_batch, labels=sentence_batch)
                    predictions = outputs[1]
                    prob_array = np.array([np.sum(np.array([torch.log(F.softmax(prediction[k],dim=0))[sentence_batch[j][k+1]].item() for k in range(int(eos_pos_list[i][j]))])) for j, prediction in enumerate(predictions)])
                    sent_prob[batch_size*i:batch_size*(i+1)] = prob_array

            else:
                for i, sentence in enumerate(sent_list):
                    words = sentence.split(" ")
                    words = ["<|endoftext|>"] + words + ["."]
                    tokenized_sentence = self.tokenizer.encode(" ".join(words), add_special_tokens=True, add_prefix_space=True)
                    input_ids = torch.tensor(tokenized_sentence).unsqueeze(0)
                    outputs = self.model(input_ids, labels=input_ids)
                    predictions = outputs[1][0]
                    prob_list = np.array([torch.log(F.softmax(predictions[j], dim=0))[tokenized_sentence[j+1]].item() for j in range(len(tokenized_sentence)-1)])
                    sent_prob[i] = np.sum(prob_list)
            return sent_prob

        elif self.model_name == 'bert':
            ##We used CPU for BERT because it cannot be easily "batchified"##
            sent_prob = np.zeros(len(sent_list))
            for i,sentence in enumerate(sent_list):
                words = sentence.split(" ")
                words.append(".")
                tokenized_sentence = self.tokenizer.encode(" ".join(words))
                prob_list = [self.calc_prob_BERT(tokenized_sentence,masked_index) for masked_index in range(1,(len(tokenized_sentence)-1))]
                sent_prob[i] = np.sum(np.array(prob_list))
            return sent_prob
        
        elif self.model_name == 'ngram':
            sent_prob = np.zeros(len(sent_list))
            for i,sentence in enumerate(sent_list):
                sent_prob[i] = self.model.score(sentence,bos = True, eos = True)
            return sent_prob
        
        else:
            print("Invalid model name")
            exit()

    def calc_prob_BERT(self,tokenized_sentence,masked_index):
        masked_sentence = tokenized_sentence.copy()
        masked_sentence[masked_index] = self.mask_id
        input_tensor = torch.tensor([masked_sentence])
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predictions = outputs[0]
            probs = predictions[0, masked_index]
            log_probs = torch.log(F.softmax(probs,dim=0))
            prob = log_probs[tokenized_sentence[masked_index]].item()
        return prob

    def calc_sent_prob_LSTM_large(self,process_id,sentence):
        words = sentence.split()
        words = words + ['.'] + ['<eos>']
        sentence = " ".join(words)
        sess, t, vocab = self.load_model_LSTM_large()
        targets = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        weights = np.ones([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.float32)
        inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros( [self.BATCH_SIZE, self.NUM_TIMESTEPS, vocab.max_word_length], np.int32)
        
        sent = [vocab.word_to_id(w) for w in sentence.split()]
        sent_char_ids = [vocab.word_to_char_ids(w) for w in sentence.split()]
        
        samples = sent[:]
        char_ids_samples = sent_char_ids[:]
        
        total_surprisal = 0
        sess.run(t['states_init'])
        
        for n in range(len(sentence.split(" "))-2):
            inputs[0, 0] = samples[0]
            char_ids_inputs[0, 0, :] = char_ids_samples[0]
            samples = samples[1:]
            char_ids_samples = char_ids_samples[1:]
            softmax = sess.run(t['softmax_out'],
                               feed_dict={t['char_inputs_in']: char_ids_inputs,
                               t['inputs_in']: inputs,
                               t['targets_in']: targets,
                               t['target_weights_in']: weights})
                
            surprisal = -1 * np.log2(softmax[0][sent[n+1]])
            total_surprisal += surprisal
        print(str(process_id) + ": " + str(total_surprisal))
        return total_surprisal

    def padding_GPT2(self,sent_list,eos_list,max_sent_len):
        padded_sent = torch.zeros((len(sent_list),max_sent_len))
        eos_pos_list = np.zeros(len(sent_list))
        for i,sentence in enumerate(sent_list):
            words = sentence.split(" ")
            words = ["<|endoftext|>"] + words + ["."]
            tokenized_sentence = self.tokenizer.encode(" ".join(words), add_special_tokens=True, add_prefix_space=True)
            eos_pos_list[i] = tokenized_sentence.index(self.tokenizer.encode(eos_list[i],add_prefix_space=True)[-1])
            if len(tokenized_sentence) > max_sent_len:
                print("Need to increase the max_sent_len")
                exit()
            padding = " ".join(["0" for i in range(max_sent_len-len(tokenized_sentence))])
            tokenized_sentence.extend(self.tokenizer.encode(padding))
            padded_sent[i] = torch.tensor(tokenized_sentence)
        return padded_sent, eos_pos_list

    def padding_LSTM(self,sent_list,eos_list,max_sent_len):
        padded_sent = torch.zeros((len(sent_list),max_sent_len))
        eos_pos_list = np.zeros(len(sent_list))
        for i,sentence in enumerate(sent_list):
            words = sentence.split(" ")
            words = ["<eos>"] + words + ["."]
            tokenized_sentence = [self.word2id[word] if word in self.word2id else self.word2id['<unk>'] for word in (['<eos>']+sentence.split(" ")+['.'])]
            eos_pos_list[i] = tokenized_sentence.index(self.word2id[eos_list[i]])
            if len(tokenized_sentence) > max_sent_len:
                print("Need to increase the max_sent_len")
                exit()
            padding = [self.word2id["0"] for i in range(max_sent_len-len(tokenized_sentence))]
            tokenized_sentence.extend(padding)
            padded_sent[i] = torch.tensor(tokenized_sentence)
        return padded_sent, eos_pos_list



    def batchify(self,data_list,batch_size):
        batch_num = int(len(data_list)/batch_size)
        if torch.is_tensor(data_list):
            return torch.cuda.LongTensor([list(data_list[batch_size*i:batch_size*(i+1)].numpy()) for i in range(batch_num)])
        else:
            return np.array([list(data_list[batch_size*i:batch_size*(i+1)]) for i in range(batch_num)])


    def load_model_LSTM_large(self):
        with tf.Graph().as_default():
            with tf.compat.v1.gfile.FastGFile(self.gd_file, 'r') as f:
                s = f.read()
                gd = tf.compat.v1.GraphDef()
                text_format.Merge(s, gd)
        
            t = {}
            [t['states_init'], t['lstm/lstm_0/control_dependency'],
             t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
             t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
             t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
             t['all_embs'], t['softmax_weights'], t['global_step']
             ] = tf.import_graph_def(gd, {}, ['states_init',
                                              'lstm/lstm_0/control_dependency:0',
                                              'lstm/lstm_1/control_dependency:0',
                                              'softmax_out:0',
                                              'class_ids_out:0',
                                              'class_weights_out:0',
                                              'log_perplexity_out:0',
                                              'inputs_in:0',
                                              'targets_in:0',
                                              'target_weights_in:0',
                                              'char_inputs_in:0',
                                              'all_embs_out:0',
                                              'Reshape_3:0',
                                              'global_step:0'], name='')
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
            sess.run('save/restore_all', {'save/Const:0': self.ckpt_file})
            sess.run(t['states_init'])
        vocab = lm_1b_utils.CharsVocabulary(self.vocab_file, self.MAX_WORD_LEN)
        return sess, t, vocab

