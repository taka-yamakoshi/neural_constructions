import torch
import os
import numpy as np
import torch.nn.functional as F
import pickle
import tensorflow as tf
from google.protobuf import text_format
from . import data_utils
from multiprocessing import Pool

class ExtractHiddenModel:
    def __init__(self,model_name,pos):
        self.model_name = model_name
        self.pos = pos
    
    def load_model(self):
        if self.model_name == 'lstm':
            with open('datafile/hidden650_batch128_dropout0.2_lr20.0.pt', 'rb') as f:
                ##Our current code only works with CPU##
                self.model = torch.load(f, map_location=torch.device('cpu'))
            self.model.eval()
            with open('datafile/word2id.pkl', 'rb') as f:
                self.word2id = pickle.load(f)
            with open('datafile/id2word.pkl', 'rb') as f:
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

        else:
            print("Invalid model name")
            exit()
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

    def extract_hidden_states(self,sent_list,verb_list,first_obj_list,eos_list,max_sent_len=20,batch_size=25):
        if self.model_name == 'lstm':
            W_in = self.model.state_dict()['encoder.weight']
            W_x0 = self.model.state_dict()['rnn.weight_ih_l0']
            W_h0 = self.model.state_dict()['rnn.weight_hh_l0']
            b_x0 = self.model.state_dict()['rnn.bias_ih_l0']
            b_h0 = self.model.state_dict()['rnn.bias_hh_l0']
            W_x1 = self.model.state_dict()['rnn.weight_ih_l1']
            W_h1 = self.model.state_dict()['rnn.weight_hh_l1']
            b_x1 = self.model.state_dict()['rnn.bias_ih_l1']
            b_h1 = self.model.state_dict()['rnn.bias_hh_l1']
            W_out = self.model.state_dict()['decoder.weight']
            b_out = self.model.state_dict()['decoder.bias']
            my_model = skeleton_LSTM(W_in, W_x0, W_h0, b_x0, b_h0, W_x1, W_h1, b_x1, b_h1, W_out, b_out, self.word2id, self.id2word)
            hidden = np.zeros((len(sent_list),4,650))
            for i,sentence in enumerate(sent_list):
                sent_data = torch.tensor([self.word2id[word] if word in self.word2id else self.word2id['<unk>'] for word in (['<eos>']+sentence.split(" ")+['.'])])
                init_c0 = torch.zeros(650)
                init_h0 = torch.zeros(650)
                init_c1 = torch.zeros(650)
                init_h1 = torch.zeros(650)
                init_state = [init_c0,init_h0,init_c1,init_h1]
                if self.pos == 'verb':
                    hidden[i],_,_ = my_model.forward_with_hidden_output(sent_data,init_state,verb_list[i],first_obj_list[i],eos_list[i])
                elif self.pos == 'first_obj':
                    _,hidden[i],_ = my_model.forward_with_hidden_output(sent_data,init_state,verb_list[i],first_obj_list[i],eos_list[i])
                elif self.pos == 'eos':
                    _,_,hidden[i] = my_model.forward_with_hidden_output(sent_data,init_state,verb_list[i],first_obj_list[i],eos_list[i])
            return hidden
        if self.model_name == 'lstm-large':
            self.BATCH_SIZE = 1
            self.NUM_TIMESTEPS = 1
            self.MAX_WORD_LEN = 50
            hidden = np.zeros((len(sent_list),1,1024))
            arg = [(i,sentence,verb_list[i],first_obj_list[i],eos_list[i]) for i,sentence in enumerate(sent_list)]
            ##This is a memory-consuming process, so the thread number should be moderate##
            with Pool(processes=10) as p:
                result_list = p.starmap(self.extract_hidden_LSTM_large,arg)
            for i in range(len(sent_list)):
                hidden[i][0] = result_list[i]
            return hidden
        if self.model_name == 'gpt2' or self.model_name == 'gpt2-large':
            if self.model_name == 'gpt2':
                hidden = np.zeros((len(sent_list),13,768))
            elif self.model_name == 'gpt2-large':
                hidden = np.zeros((len(sent_list),37,1280))
            if torch.cuda.is_available():
                padded_sent_list,verb_pos_list,first_obj_pos_list, eos_pos_list = self.padding_GPT2(sent_list,verb_list,first_obj_list,eos_list,max_sent_len)
                padded_sent_list = self.batchify(padded_sent_list,batch_size)
                verb_pos_list = self.batchify(verb_pos_list,batch_size)
                first_obj_pos_list = self.batchify(first_obj_pos_list,batch_size)
                eos_pos_list = self.batchify(eos_pos_list,batch_size)
                for i,sentence_batch in enumerate(padded_sent_list):
                    outputs = self.model(sentence_batch, labels=sentence_batch)
                    for layer_num, layer in enumerate(outputs[3]):
                        for j in range(batch_size):
                            if self.pos == 'verb':
                                hidden[i][j][layer_num] = layer[j][int(verb_pos_list[i][j])].cpu().detach().numpy()
                            elif self.pos == 'first_obj':
                                hidden[i][j][layer_num] = layer[j][int(first_obj_pos_list[i][j])].cpu().detach().numpy()
                            elif self.pos == 'eos':
                                hidden[i][j][layer_num] = layer[j][int(eos_pos_list[i][j])].cpu().detach().numpy()
            else:
                for i, sentence in enumerate(sent_list):
                    words = sentence.split(" ")
                    words = ["<|endoftext|>"] + words + ["."]
                    tokenized_sentence = self.tokenizer.encode(" ".join(words), add_special_tokens=True, add_prefix_space=True)
                    verb_pos = tokenized_sentence.index(self.tokenizer.encode(verb_list[i],add_prefix_space=True)[-1])
                    first_obj_pos  = tokenized_sentence.index(self.tokenizer.encode(first_obj_list[i],add_prefix_space=True)[-1])
                    eos_pos = tokenized_sentence.index(self.tokenizer.encode(eos_list[i],add_prefix_space=True)[-1])
                    input_ids = torch.tensor(tokenized_sentence).unsqueeze(0)
                    outputs = self.model(input_ids, labels=input_ids)
                    for j, layer in enumerate(outputs[3]):
                        if self.pos == 'verb':
                            hidden[i][j] = layer[0][verb_pos].detach().numpy()
                        elif self.pos == 'first_obj':
                            hidden[i][j] = layer[0][first_obj_pos].detach().numpy()
                        elif self.pos == 'eos':
                            hidden[i][j] = layer[0][eos_pos].detach().numpy()
            return hidden

    def padding_GPT2(self,sent_list,verb_list,first_obj_list,eos_list,max_sent_len):
        padded_sent = torch.zeros((len(sent_list),max_sent_len))
        verb_pos_list = np.zeros(len(sent_list))
        first_obj_pos_list = np.zeros(len(sent_list))
        eos_pos_list = np.zeros(len(sent_lists))
        for i,sentence in enumerate(sent_list):
            words = sentence.split(" ")
            words = ["<|endoftext|>"] + words + ['.']
            tokenized_sentence = tokenizer.encode(" ".join(words), add_special_tokens=True, add_prefix_space=True)
            verb_pos_list[i] = tokenized_sentence.index(tokenizer.encode(verb_list[i],add_prefix_space=True)[-1])
            first_obj_pos_list[i]  = tokenized_sentence.index(tokenizer.encode(first_obj_list[i],add_prefix_space=True)[-1])
            eos_pos_list[i] = tokenized_sentence.index(tokenizer.encode(eos_list[i],add_prefix_space=True)[-1])
            if len(tokenized_sentence) > max_sent_len:
                print("Need to increase the max_sent_len")
                exit()
            padding = " ".join(["0" for i in range(max_sent_len-len(tokenized_sentence))])
            tokenized_sentence.extend(tokenizer.encode(padding))
            padded_sent[i] = torch.tensor(tokenized_sentence)
        return padded_sent, verb_pos_list, first_obj_pos_list, eos_pos_list

    def batchify(self,data_list,batch_size):
        batch_num = int(len(data_list)/batch_size)
        if torch.is_tensor(data_list):
            return torch.cuda.LongTensor([list(data_list[batch_size*i:batch_size*(i+1)].numpy()) for i in range(batch_num)])
        else:
            return np.array([list(data_list[batch_size*i:batch_size*(i+1)]) for i in range(batch_num)])

    def extract_hidden_LSTM_large(self,process_id,sentence,verb,first_obj,eos):
        print(process_id)
        words = sentence.split()
        words = words + ['.'] + ['<eos>']
        sentence = " ".join(words)
        sess, t, vocab = self.load_model_LSTM_large()
        targets = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        weights = np.ones([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.float32)
        inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros( [self.BATCH_SIZE, self.NUM_TIMESTEPS, vocab.max_word_length], np.int32)

        if sentence.find('<S>') != 0:
            sentence = '<S> ' + sentence
        
        word_ids = [vocab.word_to_id(w) for w in sentence.split()]
        char_ids = [vocab.word_to_char_ids(w) for w in sentence.split()]

        verb_id = vocab.word_to_id(verb)
        first_obj_id = vocab.word_to_id(first_obj)
        eos_id = vocab.word_to_id(eos)
        
        inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS, vocab.max_word_length], np.int32)
        verb_flag = True
        first_obj_flag = True
        for i in range(len(word_ids)):
            inputs[0, 0] = word_ids[i]
            char_ids_inputs[0, 0, :] = char_ids[i]

        # Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
        # LSTM.
            lstm_emb = sess.run(t['lstm/lstm_1/control_dependency'],
                            feed_dict={t['char_inputs_in']: char_ids_inputs,
                            t['inputs_in']: inputs,
                            t['targets_in']: targets,
                            t['target_weights_in']: weights})
            if self.pos == 'verb':
                if word_ids[i] == verb_id and verb_flag:
                    return lstm_emb
            if self.pos == 'first_obj':
                if word_ids[i] == first_obj_id and first_obj_flag:
                    return lstm_emb
            if self.pos == 'eos':
                if word_ids[i] == eos_id:
                    return lstm_emb


class skeleton_LSTM:
    def __init__(self,W_in, W_x0, W_h0, b_x0, b_h0, W_x1, W_h1, b_x1, b_h1, W_out, b_out, word2id,id2word):
        self.W_in = W_in
        self.W_x0 = W_x0
        self.W_h0 = W_h0
        self.b_x0 = b_x0
        self.b_h0 = b_h0
        self.W_x1 = W_x1
        self.W_h1 = W_h1
        self.b_x1 = b_x1
        self.b_h1 = b_h1
        self.W_out = W_out
        self.b_out = b_out
        self.word2id = word2id
        self.id2word = id2word
    def embed(self,in_word,W_in):
        return W_in[in_word]
    def myLSTM(self,in_word,hid_state):
        c0 = hid_state[0]
        h0 = hid_state[1]
        c1 = hid_state[2]
        h1 = hid_state[3]
        
        Xm0 = torch.mv(self.W_x0,self.embed(in_word,self.W_in)) + self.b_x0
        Hm0 = torch.mv(self.W_h0,h0) + self.b_h0
        
        i0 = torch.sigmoid(Xm0[:650] + Hm0[:650])
        f0 = torch.sigmoid(Xm0[650:1300] + Hm0[650:1300])
        g0 = torch.tanh(Xm0[1300:1950] + Hm0[1300:1950])
        o0 = torch.sigmoid(Xm0[1950:] + Hm0[1950:])
        new_c0 = c0*f0 + g0*i0
        mid_c0 = c0*f0
        add_c0 = g0*i0
        new_h0 = torch.tanh(new_c0)*o0
        
        Xm1 = torch.mv(self.W_x1,new_h0) + self.b_x1
        Hm1 = torch.mv(self.W_h1,h1) + self.b_h1
        
        i1 = torch.sigmoid(Xm1[:650] + Hm1[:650])
        f1 = torch.sigmoid(Xm1[650:1300] + Hm1[650:1300])
        g1 = torch.tanh(Xm1[1300:1950] + Hm1[1300:1950])
        o1 = torch.sigmoid(Xm1[1950:] + Hm1[1950:])
        new_c1 = c1*f1 + g1*i1
        mid_c1 = c1*f1
        add_c1 = g1*i1
        new_h1 = torch.tanh(new_c1)*o1
        
        outvec = torch.mv(self.W_out,new_h1) + self.b_out
        return outvec, [new_c0,new_h0,new_c1,new_h1]
    
    def forward_with_hidden_output(self,sentence,init_hid_state,verb,first_obj,eos):
        if verb in self.word2id:
            verb_id = self.word2id[verb]
        else:
            verb_id  = self.word2id['<unk>']
        if first_obj in self.word2id:
            first_obj_id = self.word2id[first_obj]
        else:
            first_obj_id  = self.word2id['<unk>']
        if eos in self.word2id:
            eos_id = self.word2id[eos]
        else:
            eos_id  = self.word2id['<unk>']
        hid_state = init_hid_state
        verb_flag = True
        first_obj_flag = True
        for i, word in enumerate(sentence):
            outvec, hid_state = self.myLSTM(word,hid_state)
            if word == verb_id and verb_flag:
                verb_flag = False
                hidden_verb = np.array([list(state) for state in hid_state])
            if word == first_obj_id and first_obj_flag:
                first_obj_flag = False
                hidden_first_obj = np.array([list(state) for state in hid_state])
            if word == eos_id:
                hidden_eos = np.array([list(state) for state in hid_state])
        return hidden_verb,hidden_first_obj,hidden_eos
