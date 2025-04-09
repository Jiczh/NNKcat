#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
torch.manual_seed(1)



class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,max_len):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim,device=torch.device('cuda'))

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim).to(torch.device('cuda'))
        #self.dropout = nn.Dropout(0.25)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tagfeat = nn.Linear(hidden_dim, 30,device=torch.device('cuda'))  ######output feature
        self.hidden2tag = nn.Linear(30, tagset_size,device=torch.device('cuda'))  ########output feature 
        self.hidden2tag2 = nn.Linear(max_len, 1,device=torch.device('cuda'))
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        
        tag_space = self.hidden2tag2(lstm_out.view(len(Totaltensor), -1).permute(1,0))
        
        tag_spacefeat = self.hidden2tagfeat(tag_space[:,0])
        tag_space2 = self.hidden2tag(tag_spacefeat)
        #tag_scores = F.log_softmax(tag_space, dim=1)
        #tag_space2 = tag_space[-1,:]
        return tag_space2 , tag_spacefeat


def featgen(vector,max_len,hiddendim=256,outdim=10):
    hiddentofeatnum = nn.Linear(hiddendim, outdim,device=torch.device('cuda'))
    lastdimremove = nn.Linear(max_len, 1,device=torch.device('cuda'))
    featgen = hiddentofeatnum(vector.view(max_len, -1))
    featgen2 = lastdimremove(featgen.permute(1,0))
    return featgen2.view(-1)


def prepare_sequence(seq, to_ix):
    sequenceslice=[ seq[(i):(i+1)] +"SEQ" for i in range(0, len(seq)) ]
    idxs = [to_ix[w] for w in sequenceslice]
    tensorseq = torch.tensor(idxs, dtype=torch.long)
    return tensorseq


def prepare_ligand(seq, to_ix):
    sequenceslice=[ seq[(i):(i+1)] for i in range(0, len(seq)) ]
    idxs = [to_ix[w] for w in sequenceslice]
    tensorseq = torch.tensor(idxs, dtype=torch.long)
    return tensorseq



datatrainraw=pd.read_csv("uniq/uniq_train_seqlist.csv") ###use sequence-characteristic_value train dataset

datatestraw=pd.read_csv("uniq/uniq_test_seqlist.csv") ###use sequence-characteristic_value test dataset

totaldata=pd.concat([datatrainraw, datatestraw])

totaldata.iloc[:, 1]

word_to_ix = {}
word_to_ix["None"] = 0
max_len1 = 0
for sent in totaldata.iloc[:,0]:
    sequenceslice=[ sent[(i):(i+1)]+"SEQ" for i in range(0, len(sent)) ]
    
    if(len(sent)>max_len1):
        max_len1=len(sent)
    for word in sequenceslice:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index


max_len2 = 0 


traindata=[]


for xrow in range(datatrainraw.shape[0]):
    sequence=datatrainraw.iloc[xrow,0]
    seqtensor = prepare_sequence(sequence, word_to_ix)
    pad1d=torch.nn.ConstantPad1d((max_len1-len(seqtensor),0), 0)
    seqtensor = pad1d(seqtensor)
    Totaltensor = seqtensor
    traindata.append((Totaltensor,torch.tensor([datatrainraw.iloc[xrow,1]], dtype=torch.float32)))


testdata=[]


for xrow in range(datatestraw.shape[0]):
    sequence=datatestraw.iloc[xrow,0]
    seqtensor = prepare_sequence(sequence, word_to_ix)
    pad1d=torch.nn.ConstantPad1d((max_len1-len(seqtensor),0), 0)
    seqtensor = pad1d(seqtensor)
    Totaltensor = seqtensor
    testdata.append((Totaltensor,torch.tensor([datatestraw.iloc[xrow,1]], dtype=torch.float32)))


EMBEDDING_DIM = 128
HIDDEN_DIM = 128


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), 1,max_len1+max_len2)  
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.011)    ######optimizer SGD have other options


torch.cuda.get_device_name(0)    
len(word_to_ix)


def train(model, traindata,loss_function=loss_function,optimizer=optimizer):
    losstotal=[]
    model.train()
    tag_scoreslist=[]
    for sentence, tags in traindata:
        model.zero_grad()
        sentence_in = sentence.to(torch.device('cuda'))
        targets = tags.to(torch.device('cuda'))
        tag_scores,hidden10 = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        losstotal.append(float(loss))
        loss.backward()
        optimizer.step()
        tag_scoreslist.append(tag_scores.cpu())
    A = np.array([traindata[i][1].numpy() for i in range(len(traindata))])
    B = np.array([tag_scoreslist[i].detach().numpy()for i in range(len(tag_scoreslist))])
    result=np.c_[A, B]
    return np.sqrt(np.mean(losstotal)), result


def test(model, testdata,loss_function=loss_function,optimizer=optimizer):
    losstotalva=[]
    model.eval()
    tag_scoreslist=[]
    with torch.no_grad():
        for sentence, tags in testdata:
            sentence_in = sentence.to(torch.device('cuda'))
            targets = tags.to(torch.device('cuda'))
            tag_scores,hidden10 = model(sentence_in)
            tag_scoreslist.append(tag_scores.cpu())
            loss = loss_function(tag_scores, targets)
            losstotalva.append(float(loss))
    A = np.array([testdata[i][1].numpy() for i in range(len(testdata))])
    B = np.array([tag_scoreslist[i].detach().numpy()for i in range(len(tag_scoreslist))])
    result=np.c_[A, B]
    return np.sqrt(np.mean(losstotalva)), result


def vaildate(model, testdata,loss_function=loss_function,optimizer=optimizer):
    losstotalva=[]
    model.eval()
    tag_scoreslist=[]
    countnum = 0
    with torch.no_grad():
        for sentence, tags in testdata:
            sentence_in = sentence.to(torch.device('cuda'))
            targets = tags.to(torch.device('cuda'))
            tag_scores,hidden10 = model(sentence_in)
            tag_scoreslist.append(tag_scores.cpu())
            loss = loss_function(tag_scores, targets)
            losstotalva.append(float(loss))
            countnum = countnum+1
            if countnum % 1000 ==0:
                print(countnum)
    A = np.array([testdata[i][1].numpy() for i in range(len(testdata))])
    B = np.array([tag_scoreslist[i].numpy()for i in range(len(tag_scoreslist))])
    result=np.c_[A, B]
    return np.sqrt(np.mean(losstotalva)) , result


def vaildate10featuregen(model, testdata,max_len,loss_function=loss_function,optimizer=optimizer):
    losstotalva=[]
    model.eval()
    tag_scoreslist=[]
    countnum = 0
    with torch.no_grad():
        for sentence, tags in testdata:
            sentence_in = sentence.to(torch.device('cuda'))
            targets = tags.to(torch.device('cuda'))
            tag_scores,hidden10 = model(sentence_in)
            #hidden10 = featgen(hidden10,outdim=outdim,max_len=max_len)
            tag_scoreslist.append(hidden10.cpu())
            countnum = countnum+1
            if countnum % 100 ==0:
                print(countnum)
    B = np.array([tag_scoreslist[i].numpy()for i in range(len(tag_scoreslist))])
    
    return B

import time
start_time = str(time.ctime()).replace(':','-').replace(' ','_')
#start_time


RMSELIST=[]
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy datacc
    train_RMSE,resulttrain = train(model,traindata)
    test_RMSE,resulttest = test(model,testdata)
    print(epoch+1, train_RMSE, test_RMSE)
    torch.save(model.state_dict(), "uniq/LSTM_epoch"+'_'+start_time+'_'+str(epoch + 1)+".pth")  # save spatial_encoder
    torch.save(optimizer.state_dict(), "uniq/LSTM_optimizer_epoch"+'_'+start_time+'_'+str(epoch + 1)+".pth") 
    temp=np.c_[epoch+1,train_RMSE,test_RMSE]
    RMSELIST.append(temp)
    np.savetxt('uniq/train'+'_'+start_time+str(epoch+1)+'.csv',  resulttrain, delimiter=',')
    np.savetxt('uniq/test'+'_'+start_time+str(epoch+1)+'.csv',  resulttest, delimiter=',')
    np.savetxt('uniq/'+'log_'+start_time+'.csv',[RMSELIST[i][0] for i in range(len(RMSELIST))],delimiter=',')


######### feature generation 
model.load_state_dict(torch.load("uniq/LSTM_epoch_Mon_May__6_12-57-25_2024_27.pth"))

feature_multi = vaildate10featuregen(model,testdata,max_len1+max_len2)    ####testdata here is dataframe that contains sequence for translation
np.savetxt('uniq/uniq_seq_test_30fea.csv',  feature_multi, delimiter=',')





