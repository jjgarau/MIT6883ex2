from copy import copy
import itertools
import os
from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GraphConv

from util import setup, check_match, sub_nP, evaluate_prefix_expression

from transformers import T5Model

import argparse


# CONVERTING INPUT TO TENSORS
def tensorize_data(data):
    """
    Collect tensors to build the input data for the model
    """
    for d in data:
        # Indices of the in_tokens in the in_vocab
        d['in_idxs'] = torch.tensor([in_vocab.token2idx.get(x, in_vocab.unk) for x in d['in_tokens']])
        d['n_in'] = n_in = len(d['in_idxs'])
        d['n_nP'] = n_nP = len(d['nP'])
        # True if the position in the input has a quantity
        d['nP_in_mask'] = mask = torch.zeros(n_in, dtype=torch.bool)
        mask[d['nP_positions']] = True
        if 'out_tokens' in d:
            # Indices of the out_tokens in the out_vocab
            d['out_idxs'] = torch.tensor([out_vocab.token2idx.get(x, out_vocab.unk) for x in d['out_tokens']])
            d['n_out'] = len(d['out_idxs'])
            # A mask where the first n_nP elements are True
            d['nP_out_mask'] = mask = torch.zeros(n_max_nP, dtype=torch.bool)
            mask[:n_nP] = True
        # Graph edges for constructing the DGL graph later
        d['qcomp_edges'] = get_quantity_comparison_edges(d)
        d['qcell_edges'] = get_quantity_cell_edges(d)
        d['qcomp_add_edges'] = get_operation_edges(d, lambda x, y: x + y)
        d['qcomp_sub_edges'] = get_operation_edges(d, lambda x, y: x - y)
        d['qcomp_mult_edges'] = get_operation_edges(d, lambda x, y: x * y)
        d['qcomp_div_edges'] = get_operation_edges(d, lambda x, y: x / (y + 0.0001))


def get_quantity_comparison_edges(d):
    """
    Fill out an adjacency matrix representing quantity comparisons, then convert to list of edges
    """
    quants = [float(x) for x in d['nP']]
    quant_positions = d['nP_positions']
    assert max(quant_positions) < d['n_in']
    adj_matrix = torch.eye(d['n_in'], dtype=np.bool)
    for x, x_pos in zip(quants, quant_positions):
        for y, y_pos in zip(quants, quant_positions):
            adj_matrix[x_pos, y_pos] |= x > y
    """
    Convert the adjacency matrix of the directed graph into a tuple of (src_edges, dst_edges), which
    is the input format of dgl.graph (see https://docs.dgl.ai/generated/dgl.graph.html).
    Hint: check out the 'nonzero' function
    """
    ### Your code here ###
    ids = torch.nonzero(adj_matrix)
    return ids[:, 0], ids[:, 1]


def get_quantity_cell_edges(d):
    """
    Fill out an adjacency matrix representing the quantity cell graph, then convert to list of edges
    """
    in_idxs = d['in_idxs']
    quant_positions = d['nP_positions']
    quant_cell_positions = d['quant_cell_positions']
    assert max(quant_cell_positions) < d['n_in']
    word_cells = set(quant_cell_positions) - set(quant_positions)
    adj_matrix = torch.eye(d['n_in'], dtype=torch.bool)
    for w_pos in word_cells:
        for q_pos in quant_positions:
            if abs(w_pos - q_pos) < 4:
                adj_matrix[w_pos, q_pos] = adj_matrix[q_pos, w_pos] = True
    pos_idxs = in_idxs[quant_cell_positions]
    for idx1, pos1 in zip(pos_idxs, quant_cell_positions):
        for idx2, pos2 in zip(pos_idxs, quant_cell_positions):
            if idx1 == idx2:
                adj_matrix[pos1, pos2] = adj_matrix[pos2, pos1] = True
    """
    Convert the adjacency matrix of the directed graph into a tuple of (src_edges, dst_edges), which
    is the input format of dgl.graph (see https://docs.dgl.ai/generated/dgl.graph.html).
    Hint: check out the 'nonzero' function
    """
    ### Your code here ###
    ids = torch.nonzero(adj_matrix)
    return ids[:, 0], ids[:, 1]


def get_operation_edges(d, fun):
    quants = [float(x) for x in d['nP']]
    quant_positions = d['nP_positions']
    assert max(quant_positions) < d['n_in']
    adj_matrix = 2 * torch.eye(d['n_in'], dtype=int)
    for x, x_pos in zip(quants, quant_positions):
        for y, y_pos in zip(quants, quant_positions):
            for z, z_pos in zip(quants, quant_positions):
                if fun(x, y) > z and x_pos != y_pos and x_pos != z_pos and y_pos != z_pos:
                    adj_matrix[x_pos, z_pos] += 1
                    adj_matrix[y_pos, z_pos] += 1
    adj_matrix = (adj_matrix / 2).type(torch.int)
    sources, ends = [], []
    for i in range(d['n_in']):
        for j in range(d['n_in']):
            for _ in range(adj_matrix[i, j].item()):
                sources.append(i)
                ends.append(j)

    return torch.Tensor(sources).type(torch.int), torch.Tensor(ends).type(torch.int)


# MODEL
class TransformerAttention(nn.Module):
    """
    Used in Transformer Block, implements the dot-product attention
    """
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_hid, n_head * (n_k * 2 + n_v))
        self.out = nn.Linear(n_head * n_v, n_hid)

    def forward(self, x, mask=None):
        n_batch, n_batch_max_in, n_hid = x.shape
        q_k_v = self.qkv(x).view(n_batch, n_batch_max_in, n_head, 2 * n_k + n_v).transpose(1, 2)
        q, k, v = q_k_v.split([n_k, n_k, n_v], dim=-1)

        q = q.reshape(n_batch * n_head, n_batch_max_in, n_k)
        k = k.reshape_as(q).transpose(1, 2)
        qk = q.bmm(k) / np.sqrt(n_k)

        if mask is not None:
            qk = qk.view(n_batch, n_head, n_batch_max_in, n_batch_max_in).transpose(1, 2)
            qk[~mask] = -np.inf
            qk = qk.transpose(1, 2).view(n_batch * n_head, n_batch_max_in, n_batch_max_in)
        qk = qk.softmax(dim=-1)
        v = v.reshape(n_batch * n_head, n_batch_max_in, n_v)
        qkv = qk.bmm(v).view(n_batch, n_head, n_batch_max_in, n_v).transpose(1, 2).reshape(n_batch, n_batch_max_in, n_head * n_v)
        out = self.out(qkv)
        return x + out


class TransformerBlock(nn.Module):
    """
    Custom Transformer
    """
    def __init__(self):
        super().__init__()
        self.attn = TransformerAttention()
        n_inner = n_hid * 4
        self.inner = nn.Sequential(
            nn.Linear(n_hid, n_inner),
            nn.ReLU(inplace=True),
            nn.Linear(n_inner, n_hid)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(x, mask=mask)
        return x + self.inner(x)


class GCNBranch(nn.Module):
    def __init__(self, n_hid_in, n_hid_out, dropout=0.3):
        super().__init__()
        """
        Define a branch of the graph convolution with
        1. GraphConv from n_hid_in to n_hid_in
        2. ReLU
        3. Dropout
        4. GraphConv from n_hid_in to n_hid_out

        Note: your should call dgl.nn.GraphConv with allow_zero_in_degree=True
        """
        ### Your code here ###
        # self.gc1 = GraphConv(n_hid_in, n_hid_in, allow_zero_in_degree=True)
        # self.act = nn.ReLU()
        # self.drop = nn.Dropout(p=dropout)
        # self.gc2 = GraphConv(n_hid_in, n_hid_out, allow_zero_in_degree=True)
        n_hid_med = n_hid_in
        self.gc1 = dgl.nn.SAGEConv(n_hid_in, n_hid_med, aggregator_type='gcn', feat_drop=dropout)
        self.gc2 = dgl.nn.SAGEConv(n_hid_med, n_hid_med, aggregator_type='gcn', feat_drop=dropout)
        self.gc3 = GraphConv(n_hid_med, n_hid_out, allow_zero_in_degree=True)


    def forward(self, x, graph):
        """
        Forward pass of your defined branch above
        """
        ### Your code here ###
        # out = self.gc1(graph, x)
        # out = self.act(out)
        # out = self.drop(out)
        # out = self.gc2(graph, out)
        out = self.gc1(graph, x)
        out = self.gc2(graph, out)
        out = self.gc3(graph, out)
        return out



class GCN(nn.Module):
    """
    A graph convolution network with multiple graph convolution branches
    """

    def __init__(self, n_head=8, dropout=0.3):
        super().__init__()
        self.n_head = n_head
        self.branches = nn.ModuleList(GCNBranch(n_hid, n_hid // n_head, dropout) for _ in range(n_head))

        self.feed_forward = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid)
        )
        self.layer_norm = nn.LayerNorm(n_hid)

    def forward(self, h, gt_graph, attr_graph, qcomp_operation_graphs):
        a, b, c, d = qcomp_operation_graphs
        x = h.reshape(-1, n_hid)
        if self.n_head == 16:
            # graphs = [gt_graph, gt_graph, attr_graph, attr_graph, a, a, b, b, gt_graph, gt_graph, attr_graph, attr_graph, a, a, b, b]
            graphs = [gt_graph, gt_graph, attr_graph, attr_graph, gt_graph, gt_graph, attr_graph, attr_graph, gt_graph, gt_graph, attr_graph, attr_graph, gt_graph, gt_graph, attr_graph, attr_graph]
        elif self.n_head == 8:
            # graphs = [gt_graph, gt_graph, attr_graph, attr_graph, a, a, b, b]
            graphs = [gt_graph, gt_graph, attr_graph, attr_graph, gt_graph, gt_graph, attr_graph, attr_graph]
        elif self.n_head == 4:
            graphs = [gt_graph, gt_graph, attr_graph, attr_graph]
        else:
            graphs = [gt_graph, attr_graph]
        x = torch.cat([branch(x, g) for branch, g in zip(self.branches, graphs)], dim=-1).view_as(h)
        x = h + self.layer_norm(x)
        # return x + self.feed_forward(x)
        return x + self.layer_norm(self.feed_forward(x))


class Gate(nn.Module):
    """
    Activation gate used a few times in the TreeDecoder
    """

    def __init__(self, n_in, n_out):
        super(Gate, self).__init__()
        self.t = nn.Linear(n_in, n_out)
        self.s = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.t(x).tanh() * self.s(x).sigmoid()


class TreeDecoder(nn.Module):
    """
    Defines parameters and methods for decoding into an expression. Used in train and predict
    """

    def __init__(self, dropout=0.5):
        super().__init__()
        drop = nn.Dropout(dropout)
        self.constant_embedding = nn.Parameter(torch.randn(1, out_vocab.n_constants, n_hid))

        self.qp_gate = nn.Sequential(drop, Gate(n_hid, n_hid))
        self.gts_right = nn.Sequential(drop, Gate(2 * n_hid, n_hid))

        self.attn_fc = nn.Sequential(drop,
                                     nn.Linear(2 * n_hid, n_hid),
                                     nn.Tanh(),
                                     nn.Linear(n_hid, 1)
                                     )
        self.quant_fc = nn.Sequential(drop,
                                      nn.Linear(n_hid * 3, n_hid),
                                      nn.Tanh(),
                                      nn.Linear(n_hid, 1, bias=False)
                                      )
        self.op_fc = nn.Sequential(drop, nn.Linear(n_hid * 2, out_vocab.n_ops))

        self.op_embedding = nn.Embedding(out_vocab.n_ops + 1, n_hid, padding_idx=out_vocab.n_ops)
        self.gts_left = nn.Sequential(drop, Gate(n_hid * 2 + n_hid, n_hid))
        self.gts_left_qp = nn.Sequential(drop, Gate(n_hid * 2 + n_hid, n_hid), self.qp_gate)

        self.subtree_gate = nn.Sequential(drop, Gate(n_hid * 2 + n_hid, n_hid))

    def gts_attention(self, q, zbar, in_mask=None):
        """
        Corresponds roughly to the GTS-Attention function defined by the paper
        """
        attn_score = self.attn_fc(
            torch.cat([q.unsqueeze(1).expand_as(zbar), zbar], dim=2)
        ).squeeze(2)
        if in_mask is not None:
            attn_score[~in_mask] = -np.inf
        attn = attn_score.softmax(dim=1)
        return (attn.unsqueeze(1) @ zbar).squeeze(1)  # (n_batch, n_hid)

    def gts_predict(self, qp_Gc, quant_embed, nP_out_mask=None):
        """
        Corresponds roughly to the GTS-Predict functions defined by the paper
        """
        quant_score = self.quant_fc(
            torch.cat([qp_Gc.unsqueeze(1).expand(-1, quant_embed.size(1), -1), quant_embed], dim=2)
        ).squeeze(2)
        op_score = self.op_fc(qp_Gc)
        pred_score = torch.cat((op_score, quant_score), dim=1)
        if nP_out_mask is not None:
            pred_score[:, out_vocab.base_nP:][~nP_out_mask] = -np.inf
        return pred_score

    def merge_subtree(self, op, tl, yr):
        """
        Corresponds to part of the GTS-Subtree function defined by the paper
        """
        return self.subtree_gate(torch.cat((op, tl, yr), dim=-1))


class Model(nn.Module):
    """
    Overall model containing all the neural network parameters and methods
    1. The base seq2seq model is in self.transformer_layers if use_t5=None else self.t5_encoder
    2. The graph convolution network is in self.gcn
    3. The tree decoder is in self.decoder
    """

    def __init__(self, dropout=0.5):
        super().__init__()
        drop = nn.Dropout(dropout)

        if use_t5:
            """
            Use t5_model.encoder as the encoder for this model. Note that unlike the custom transformer, you don't
            need to use an external input or positional embedding for the T5 transformer 
            (i.e. don't define self.in_embed or self.pos_emb) since it already defines them internally

            You may specify layer weights to freeze during finetuning by modifying the freeze_layers global variable
            """
            ### Your code here ###
            self.t5_model = T5Model.from_pretrained(f't5-{use_t5}')
            self.t5_encoder = self.t5_model.encoder

            for i_layer, block in enumerate(self.t5_encoder.block):
                if i_layer in freeze_layers:
                    for param in block.parameters():
                        param.requires_grad = False
        else:
            # Input embedding for custom transformer
            self.in_embed = nn.Sequential(nn.Embedding(in_vocab.n, n_hid, padding_idx=in_vocab.pad), drop)
            # Positional embedding for custom transformer
            self.pos_embed = nn.Embedding(1 + n_max_in, n_hid)  # Use the first position as global vector
            self.transformer_layers = nn.ModuleList(TransformerBlock() for _ in range(n_layers))

        self.gcn = GCN(n_head=args.n_head)

        self.decoder = TreeDecoder()

        if not use_t5:
            self.apply(self.init_weight)

    def init_weight(self, m):
        if type(m) in [nn.Embedding]:
            nn.init.normal_(m.weight, 0, 0.1)

    def encode(self, in_idxs, n_in, gt_graph, attr_graph, qcomp_operation_graphs, in_mask=None):
        in_idxs_pad = F.pad(in_idxs, (1, 0), value=in_vocab.pad)
        if use_t5:
            """
            Use your T5 encoder to encoder the input indices. Note that you do NOT need to use an input embedding or
            positional embedding (e.g. self.in_embed or self.pos_embed) for T5, since it already defines
            the embeddings internally
            """
            ### Your code here ###
            h = self.t5_encoder(in_idxs_pad)[0]
        else:
            x = self.in_embed(in_idxs_pad)  # (n_batch, n_batch_max_in, n_hid)
            h = x + self.pos_embed(torch.arange(x.size(1), device=x.device))
            for layer in self.transformer_layers:
                h = layer(h, mask=in_mask)
        zg, h = h[:, 0], h[:, 1:]
        zbar = self.gcn(h, gt_graph, attr_graph, qcomp_operation_graphs)
        return zbar, zg


# TRAINING A BATCH
class Node:
    """
    Node for tree traversal during training
    """

    def __init__(self, up):
        self.up = up
        self.is_root = up is None
        self.left = self.right = None
        self.ql = self.tl = self.op = None


def train(batch, model, opt):
    """
    Compute the loss on a batch of inputs, and take a step with the optimizer
    """
    n_batch = len(batch)

    n_in = [d['n_in'] for d in batch]
    pad = lambda x, value: nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=value)
    in_idxs = pad([d['in_idxs'] for d in batch], in_vocab.pad).to(device)
    in_mask = pad([torch.ones(n, dtype=torch.bool) for n in n_in], False).to(device)
    nP_in_mask = pad([d['nP_in_mask'] for d in batch], False).to(device)
    nP_out_mask = torch.stack([d['nP_out_mask'] for d in batch]).to(device)

    qcomp_graph, qcell_graph = [], []
    qcomp_add_graph, qcomp_sub_graph, qcomp_mult_graph, qcomp_div_graph = [], [], [], []
    for d in batch:
        """
        Create qcomp_graph and qcell_graph from d['qcomp_edges'] and d['qcell_edges'] by calling dgl.graph
        (see https://docs.dgl.ai/generated/dgl.graph.html)

        Note that num_nodes needs to be set to the maximum input length in this batch
        """
        ### Your code here ###
        qcomp_graph_i = dgl.graph(d['qcomp_edges'], num_nodes=max(n_in), device=device)
        qcell_graph_i = dgl.graph(d['qcell_edges'], num_nodes=max(n_in), device=device)

        qcomp_graph.append(qcomp_graph_i)
        qcell_graph.append(qcell_graph_i)

        qcomp_add_graph_i = dgl.graph(d['qcomp_add_edges'], num_nodes=max(n_in), device=device)
        qcomp_sub_graph_i = dgl.graph(d['qcomp_sub_edges'], num_nodes=max(n_in), device=device)
        qcomp_mult_graph_i = dgl.graph(d['qcomp_mult_edges'], num_nodes=max(n_in), device=device)
        qcomp_div_graph_i = dgl.graph(d['qcomp_div_edges'], num_nodes=max(n_in), device=device)

        qcomp_add_graph.append(qcomp_add_graph_i)
        qcomp_sub_graph.append(qcomp_sub_graph_i)
        qcomp_mult_graph.append(qcomp_mult_graph_i)
        qcomp_div_graph.append(qcomp_div_graph_i)

    qcomp_graph = dgl.batch(qcomp_graph)
    qcell_graph = dgl.batch(qcell_graph)

    qcomp_add_graph = dgl.batch(qcomp_add_graph)
    qcomp_sub_graph = dgl.batch(qcomp_sub_graph)
    qcomp_mult_graph = dgl.batch(qcomp_mult_graph)
    qcomp_div_graph = dgl.batch(qcomp_div_graph)
    qcomp_operation_graphs = (qcomp_add_graph, qcomp_sub_graph, qcomp_mult_graph, qcomp_div_graph)

    label = pad([d['out_idxs'] for d in batch], out_vocab.pad)
    nP_candidates = [d['nP_candidates'] for d in batch]

    zbar, qroot = model.encode(in_idxs, n_in, qcomp_graph, qcell_graph, qcomp_operation_graphs, in_mask=None)
    z_nP = zbar.new_zeros((n_batch, n_max_nP, n_hid))
    z_nP[nP_out_mask] = zbar[nP_in_mask]

    decoder = model.decoder

    n_quant = out_vocab.n_constants + n_max_nP
    quant_embed = torch.cat([decoder.constant_embedding.expand(n_batch, -1, -1), z_nP],
                            dim=1)  # (n_batch, n_quant, n_hid)

    nodes = np.array([Node(None) for _ in range(n_batch)])
    op_min, op_max = out_vocab.base_op, out_vocab.base_op + out_vocab.n_ops
    quant_min, quant_max = out_vocab.base_quant, out_vocab.base_quant + n_quant

    # Initialize root node vector according to zg (the global context)
    qp = decoder.qp_gate(qroot)
    scores = []
    for i, label_i in enumerate(label.T):  # Iterate over the output positions
        Gc = decoder.gts_attention(qp, zbar, in_mask)
        qp_Gc = torch.cat([qp, Gc], dim=1)  # (n_batch, 2 * n_hid)

        score = decoder.gts_predict(qp_Gc, quant_embed, nP_out_mask)
        scores.append(score)

        # Whether the label is an operator
        is_op = (op_min <= label_i) & (label_i < op_max)
        # Whether the label is a quantity
        is_quant = ((quant_min <= label_i) & (label_i < quant_max)) | (label_i == out_vocab.unk)

        op_embed = decoder.op_embedding((label_i[is_op] - out_vocab.base_op).to(device))
        qp_Gc_op = torch.cat([qp_Gc[is_op], op_embed], dim=1)

        is_left = np.zeros(n_batch, dtype=np.bool)
        qleft_qp = decoder.gts_left_qp(qp_Gc_op)
        qleft = decoder.gts_left(qp_Gc_op)
        for j, ql, op in zip(is_op.nonzero(as_tuple=True)[0], qleft, op_embed):
            node = nodes[j]
            nodes[j] = node.left = Node(node)
            node.op = op
            node.ql = ql
            is_left[j] = True

        is_right = np.zeros(n_batch, dtype=np.bool)
        nP_score = score[:, out_vocab.base_nP:].detach().cpu()
        ql_tl = []
        for j in is_quant.nonzero(as_tuple=True)[0]:
            if label_i[j] == out_vocab.unk:
                candidates = nP_candidates[j][i]
                label_i[j] = out_vocab.base_nP + candidates[nP_score[j, candidates].argmax()]

            node = nodes[j]
            pnode = node.up
            t = quant_embed[j, label_i[j] - out_vocab.base_quant]
            while pnode and pnode.right is node:
                t = decoder.merge_subtree(pnode.op, pnode.tl, t)  # merge operator, left subtree, and right child
                node, pnode = pnode, pnode.up  # backtrack to parent node
            if pnode is None:  # Finished traversing tree of j
                continue
            # Now pnode.left is node. t is the tl representing the left subtree of pnode
            pnode.tl = t
            ql_tl.append(torch.cat([pnode.ql, pnode.tl]))  # For computing qright
            nodes[j] = pnode.right = Node(pnode)
            is_right[j] = True

        qp = torch.zeros((n_batch, n_hid), device=device)
        qp[is_left] = qleft_qp
        if ql_tl:
            qp[is_right] = decoder.gts_right(torch.stack(ql_tl))

    label = label.to(device).view(-1)
    scores = torch.stack(scores, dim=1).view(-1, out_vocab.n_ops + n_quant)
    loss = F.cross_entropy(scores, label, ignore_index=out_vocab.pad)

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


# PREDICTION (FOR EVALUATION)
class BeamNode(Node):
    """
    Node for beam search during evaluation
    """
    def __init__(self, up, prev, qp, token=None):
        super().__init__(up)
        self.prev = prev
        self.qp = qp
        self.token = token

    def trace_tokens(self, *last_token):
        if self.prev is None:
            return list(last_token)
        tokens = self.prev.trace_tokens()
        tokens.append(self.token)
        tokens.extend(last_token)
        return tokens

def predict(d, model, beam_size=5, n_max_out=45):
    """
    Predict the idxs corresponding to an expression given the inputs. Leverages beam search to maximize
    prediction probability
    """
    in_idxs = d['in_idxs'].unsqueeze(0).to(device=device)
    """
    Create qcomp_graph and qcell_graph from d['qcomp_edges'] and d['qcell_edges'] by calling dgl.graph
    (see https://docs.dgl.ai/generated/dgl.graph.html)
    """
    ### Your code here ###
    qcomp_graph = dgl.graph(d['qcomp_edges'], device=device)
    qcell_graph = dgl.graph(d['qcell_edges'], device=device)

    qcomp_add_graph = dgl.graph(d['qcomp_add_edges'], device=device)
    qcomp_sub_graph = dgl.graph(d['qcomp_sub_edges'], device=device)
    qcomp_mult_graph = dgl.graph(d['qcomp_mult_edges'], device=device)
    qcomp_div_graph = dgl.graph(d['qcomp_div_edges'], device=device)
    qcomp_operation_graphs = (qcomp_add_graph, qcomp_sub_graph, qcomp_mult_graph, qcomp_div_graph)

    zbar, qroot = model.encode(in_idxs, [d['n_in']], qcomp_graph, qcell_graph, qcomp_operation_graphs)
    z_nP = zbar[:, d['nP_positions']]

    decoder = model.decoder

    quant_embed = torch.cat([decoder.constant_embedding, z_nP], dim=1) # (1, n_quant, n_hid)
    op_min, op_max = out_vocab.base_op, out_vocab.base_op + out_vocab.n_ops

    best_done_beam = (-np.inf, None, None)
    beams = [(0, BeamNode(up=None, prev=None, qp=decoder.qp_gate(qroot)))]
    for _ in range(n_max_out):
        new_beams = []
        for logp_prev, node in beams:
            Gc = decoder.gts_attention(node.qp, zbar)
            qp_Gc = torch.cat([node.qp, Gc], dim=1) # (2 * n_hid,)

            log_prob = decoder.gts_predict(qp_Gc, quant_embed).log_softmax(dim=1)
            top_logps, top_tokens = log_prob.topk(beam_size, dim=1)
            for logp_token_, out_token_ in zip(top_logps.unbind(dim=1), top_tokens.unbind(dim=1)):
                out_token = out_token_.item()
                logp = logp_prev + logp_token_.item()
                if op_min <= out_token < op_max:
                    op_embed = decoder.op_embedding(out_token_)
                    qp_Gc_op = torch.cat([qp_Gc, op_embed], dim=1)
                    prev_node = copy(node)
                    next_node = prev_node.left = BeamNode(
                        up=prev_node, prev=prev_node,
                        qp=decoder.gts_left_qp(qp_Gc_op),
                        token=out_token
                    )
                    prev_node.op = op_embed
                    prev_node.ql = decoder.gts_left(qp_Gc_op)
                else:
                    pnode, prev_node = node.up, node
                    t = quant_embed[:, out_token - out_vocab.base_quant]
                    while pnode and pnode.tl is not None:
                        t = decoder.merge_subtree(pnode.op, pnode.tl, t)
                        node, pnode = pnode, pnode.up
                    if pnode is None:
                        best_done_beam = max(best_done_beam, (logp, prev_node, out_token))
                        continue
                    pnode = copy(pnode)
                    pnode.tl = t
                    next_node = pnode.right = BeamNode(
                        up=pnode, prev=prev_node,
                        qp=decoder.gts_right(torch.cat([pnode.ql, pnode.tl], dim=1)),
                        token=out_token
                    )
                new_beams.append((logp, next_node))
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        done_logp, done_node, done_last_token = best_done_beam
        if not len(beams) or done_logp >= beams[0][0]:
            break
    return done_node.trace_tokens(done_last_token)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Main script for running LPG")
    parser.add_argument('--use_t5', type=str, default='small')
    parser.add_argument('--n_hid', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_k', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--n_batch', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args()

    # use_t5 = 'small'  # Value should be None, 'small', or 'base', or 'large', or '3B'

    use_t5 = args.use_t5 if args.use_t5 in ['small', 'base', 'large', '3b'] else None
    model_save_dir = f'models/{use_t5 or "custom"}'
    os.makedirs(model_save_dir, exist_ok=True)

    # IMPORTANT NOTE: if you change some of these hyperparameters during training,
    # you will also need to change them during prediction (see next section)
    n_max_in = 100
    n_epochs = args.n_epochs
    n_batch = args.n_batch
    learning_rate = args.learning_rate
    if use_t5:
        # T5 hyperparameters
        freeze_layers = []
        weight_decay = 1e-5
        if use_t5 == '3b':
            n_hid = 1024
        else:
            n_hid = dict(small=512, base=768, large=1024)[use_t5]  # Do not modify unless you want to try t5-large
    else:
        # Custom transformer hyperparameters
        # n_layers = 3
        # n_hid = 512
        # n_k = n_v = 64
        # n_head = 8
        # weight_decay = 0
        n_layers = args.n_layers
        n_hid = args.n_hid
        n_k = n_v = args.n_k
        n_head = args.n_head
        weight_decay = args.weight_decay

    device = 'cuda:' + str(args.cuda_device) if torch.cuda.is_available() and not args.use_cpu else 'cpu'

    train_data, val_data, in_vocab, out_vocab, n_max_nP, t5_model = setup(use_t5)
    tensorize_data(itertools.chain(train_data, val_data))

    model = Model()
    if args.load_model:
        eval_epoch = args.eval_epoch
        model.load_state_dict(torch.load(f'models/{use_t5 or "custom"}/model-{eval_epoch}.pth'))
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    model.to(device)

    epoch = 0
    while epoch < n_epochs:
        print('Epoch:', epoch + 1)
        model.train()
        losses = []
        for start in trange(0, len(train_data), n_batch):
            batch = sorted(train_data[start: start + n_batch], key=lambda d: -d['n_in'])
            loss = train(batch, model, opt)
            losses.append(loss)
        scheduler.step()

        print(f'Training loss: {np.mean(losses):.3g}')

        epoch += 1
        if epoch % 5 == 0:
            model.eval()
            value_match, equation_match = [], []
            with torch.no_grad():
                for d in tqdm(val_data):
                    if d['is_quadratic']:  # This method is not equiped to handle equations with quadratics
                        val_match = eq_match = False
                    else:
                        pred = predict(d, model)
                        d['pred_tokens'] = [out_vocab.idx2token[idx] for idx in pred]
                        val_match, eq_match = check_match(pred, d)
                    value_match.append(val_match)
                    equation_match.append(eq_match)
            print(f'Validation expression accuracy: {np.mean(equation_match):.3g}')
            print(f'Validation value accuracy: {np.mean(value_match):.3g}')
            # We save the model every 10 epochs, feel free to load in a trained model with
            # model.load_state_dict(torch.load(f'models/model-{epoch}.pth'))
            # Note: if you want to restart training from a saved model, you must also save and load the optimizer with
            # torch.save(opt.state_dict(), os.path.join(model_save_dir, f'opt-{epoch}.pth'))
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'model-{epoch}.pth'))
        print()

    if True:
        eval_epoch = args.eval_epoch

        # Make sure your parameter here is the exact same as the parameters you trained with,
        # else the model will not load correctly

        test_data, in_vocab, out_vocab, n_max_nP, t5_model = setup(use_t5, do_eval=True)
        model = Model()
        model.load_state_dict(torch.load(f'models/{use_t5 or "custom"}/model-{eval_epoch}.pth'))
        tensorize_data(test_data)
        model.to(device)

        with torch.no_grad():
            for d in tqdm(test_data):  # There's no quadratics in the test_data, fortunately
                pred = predict(d, model)
                d['pred_tokens'] = pred_tokens = [out_vocab.idx2token[idx] for idx in pred]
                d['subbed_tokens'] = subbed_tokens = sub_nP(pred_tokens, d['nP'])
                d['Predicted'] = round(evaluate_prefix_expression(subbed_tokens), 3)  # Make sure to round to 3 decimals

        import pandas as pd

        predictions = pd.DataFrame(test_data).set_index('Id')
        predictions[['Predicted']].replace([np.inf, -np.inf, np.nan], 0).to_csv('prediction.csv')
