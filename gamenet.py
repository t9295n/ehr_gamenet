import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 (original GCN paper)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



'''
Graph Convolution Network
'''
class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cuda:0')):
        '''
        Graph Convolution network to learn improved embeddings on drug-combos and DDIs
        Inputs:
            - vocabulary size (for medications/drugs)
            - size of embedding vector
            - adjacency matrix (ehr or ddi)
        '''
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        
        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim).to(device)
        self.dropout = nn.Dropout(p=0.4).to(device)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim).to(device)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding


class GAMENet(nn.Module):
    def __init__(self, vocab_size, ehr_adj_norm, ddi_adj, ddi_adj_norm, emb_dim=64, device=torch.device('cuda:0'), ddi_in_memory=True):
        '''
        GAMENet: medical embedding module, patient representation module, graph augmented memory module
        '''
        super(GAMENet, self).__init__()
        self.vocab_size = vocab_size ## 0 = diag vocab size, 1 = proc vocab size
        self.device = device
        self.ddi_in_memory = ddi_in_memory

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.dropout = nn.Dropout(p=0.4)

        ## medical embeddings, create diagnosis and procedure embeddings, initialize weights
        self.diag_embed = nn.Embedding(vocab_size[0], emb_dim)
        self.proc_embed = nn.Embedding(vocab_size[1], emb_dim)
        self.diag_embed.weight.data.uniform_(-0.1, 0.1)
        self.proc_embed.weight.data.uniform_(-0.1, 0.1)

        ## patient representation module (query), projects hidden states from dual-rnn to query using fully connected NN
        self.diag_rnn = nn.GRU(emb_dim, emb_dim*2, batch_first=True)
        self.proc_rnn = nn.GRU(emb_dim, emb_dim*2, batch_first=True)
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        ## graph augmented memory module

        ## memory bank, stores ehr graph (and ddi if argument is set to true)
        ## beta is a weighting var to fuse different knowledge graphs
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj_norm, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj_norm, device=device)
        self.beta = nn.Parameter(torch.FloatTensor(1))
        self.beta.data.uniform_(-0.1, 0.1)

        ## convert output to predict final multi-label medication
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim*3, emb_dim*2),
            nn.ReLU(),
            nn.Linear(emb_dim*2, vocab_size[2])
        )

    def forward(self, input):
        # medical embeddings module (diagnosis and procedures per visit) and queries
        mean_embedding = lambda embedding: embedding.mean(dim=1).unsqueeze(dim=0).to(self.device)
        diag_seq = torch.cat([mean_embedding(self.dropout(self.diag_embed(torch.LongTensor(visit[0]).unsqueeze(dim=0).to(self.device)))) for visit in input], dim=1)
        proc_seq = torch.cat([mean_embedding(self.dropout(self.proc_embed(torch.LongTensor(visit[1]).unsqueeze(dim=0).to(self.device)))) for visit in input], dim=1)

        ## patient representation module (query), projects hidden states from dual-rnn to query using fully connected NN
        patient_reps = torch.cat([self.diag_rnn(diag_seq)[0], self.proc_rnn(proc_seq)[0]], dim=-1).squeeze(dim=0)
        queries = self.query(patient_reps)

        ### Graph Augmented Memory Module ###
        ## embeds and stores EHR and DDI graph as facts in memory bank
        ## inserts patient history to dynamic memory

        '''I: input memory representation'''
        query = queries[-1:]
        history_size = len(input) - 1

        '''G: Generalization (generating and updating memory representation)'''
        ## memory bank, stores ehr graph (and ddi if argument is set to true)
        ## beta is a weighting var to fuse different knowledge graphs
        memory_bank = self.ehr_gcn() - self.ddi_gcn() * self.beta if self.ddi_in_memory else self.ehr_gcn()
        content_attention = F.softmax(torch.mm(query, memory_bank.t()), dim=-1)

        ## dynamic memory
        if history_size > 0:
            memory_keys = queries[:history_size] 
            memory_vals = np.zeros((history_size, self.vocab_size[2]))
            for i in range(history_size):
                visit  = input[i]
                memory_vals[i, visit[2]] = 1
            memory_vals = torch.FloatTensor(memory_vals).to(self.device)

            ## temporal attention and history medication for later output
            temporal_attention = F.softmax(torch.mm(query, memory_keys.t()))
            history_medication = torch.mm(temporal_attention, memory_vals)

        '''O: Output memory representation'''
        ## memory bank output + dynamic memory output (same as memory bank if no history)
        memory_bank_output = torch.mm(content_attention, memory_bank) 
        dynamic_memory_output = torch.mm(history_medication, memory_bank) if history_size > 0 else memory_bank_output

        '''R: Response, use patient representation and memory output to predict multi-label medication'''
        output = self.output(torch.cat([query, memory_bank_output, dynamic_memory_output], dim=-1))
        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = torch.mm(neg_pred_prob.t(), neg_pred_prob)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()
            return output, batch_neg
        else:
            return output
