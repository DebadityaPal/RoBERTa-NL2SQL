
import os
import json
import random as rd
from copy import deepcopy

from matplotlib.pylab import *

import math
import torch
import torchvision.datasets as dsets
import torch.nn as nn
import torch.nn.functional as F
# import torch_xla
# import torch_xla.core.xla_model as xm

device = torch.device("cuda")




def encode(lstm, wemb_l, l, return_hidden=False, hc0=None, last_only=False):
    """ [batch_size, max token length, dim_emb]
    """
    bS, mL, eS = wemb_l.shape

    # sort before packking
    l = array(l)
    perm_idx = argsort(-l)
    perm_idx_inv = generate_perm_inv(perm_idx)

    # pack sequence

    packed_wemb_l = nn.utils.rnn.pack_padded_sequence(wemb_l[perm_idx, :, :],
                                                      l[perm_idx],
                                                      batch_first=True)

    # Time to encode
    if hc0 is not None:
        hc0 = (hc0[0][:, perm_idx], hc0[1][:, perm_idx])

    # ipdb.set_trace()
    packed_wemb_l = packed_wemb_l.float()  # I don't know why..
    packed_wenc, hc_out = lstm(packed_wemb_l, hc0)
    hout, cout = hc_out

    # unpack
    wenc, _l = nn.utils.rnn.pad_packed_sequence(packed_wenc, batch_first=True)

    if last_only:
        # Take only final outputs for each columns.
        wenc = wenc[tuple(range(bS)), l[perm_idx] - 1]  # [batch_size, dim_emb]
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]

    wenc = wenc[perm_idx_inv]

    if return_hidden:
        # hout.shape = [number_of_directoin * num_of_layer, seq_len(=batch size), dim * number_of_direction ] w/ batch_first.. w/o batch_first? I need to see.
        hout = hout[:, perm_idx_inv].to(device)
        cout = cout[:, perm_idx_inv].to(device)  # Is this correct operation?

        return wenc, hout, cout
    else:
        return wenc


def encode_hpu(lstm, wemb_hpu, l_hpu, l_hs):
    wenc_hpu, hout, cout = encode(lstm,
                                  wemb_hpu,
                                  l_hpu,
                                  return_hidden=True,
                                  hc0=None,
                                  last_only=True)

    wenc_hpu = wenc_hpu.squeeze(1)
    bS_hpu, mL_hpu, eS = wemb_hpu.shape
    hS = wenc_hpu.size(-1)

    wenc_hs = wenc_hpu.new_zeros(len(l_hs), max(l_hs), hS)
    wenc_hs = wenc_hs.to(device)

    # Re-pack according to batch.
    # ret = [B_NLq, max_len_headers_all, dim_lstm]
    st = 0
    for i, l_hs1 in enumerate(l_hs):
        wenc_hs[i, :l_hs1] = wenc_hpu[st:(st + l_hs1)]
        st += l_hs1

    return wenc_hs


def generate_perm_inv(perm):
    # Definitly correct.
    perm_inv = zeros(len(perm), dtype=int)  # Was an undefine int32 variable
    for i, p in enumerate(perm):
        perm_inv[int(p)] = i

    return perm_inv


def pred_sc(s_sc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc = []
    for s_sc1 in s_sc:
        pr_sc.append(s_sc1.argmax().item())

    return pr_sc


def pred_sc_beam(s_sc, beam_size):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc_beam = []

    for s_sc1 in s_sc:
        val, idxes = s_sc1.topk(k=beam_size)
        pr_sc_beam.append(idxes.tolist())

    return pr_sc_beam


def pred_sa(s_sa):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sa = []
    for s_sa1 in s_sa:
        pr_sa.append(s_sa1.argmax().item())

    return pr_sa


def pred_wn(s_wn):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_wn = []
    for s_wn1 in s_wn:
        pr_wn.append(s_wn1.argmax().item())
        # print(pr_wn, s_wn1)
        # if s_wn1.argmax().item() == 3:
        #     input('')

    return pr_wn


def pred_wc(wn, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted!
    """
    # get g_num
    pr_wc = []
    for b, wn1 in enumerate(wn):
        s_wc1 = s_wc[b]

        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())[:wn1]
        pr_wc1.sort()

        pr_wc.append(list(pr_wc1))
    return pr_wc


def pred_wo(wn, s_wo):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # s_wo = [B, 4, n_op]
    pr_wo_a = s_wo.argmax(dim=2)  # [B, 4]
    # get g_num
    pr_wo = []
    for b, pr_wo_a1 in enumerate(pr_wo_a):
        wn1 = wn[b]
        pr_wo.append(list(pr_wo_a1.data.cpu().numpy()[:wn1]))

    return pr_wo


def topk_multi_dim(tensor, n_topk=1, batch_exist=True):

    if batch_exist:
        idxs = []
        for b, tensor1 in enumerate(tensor):
            idxs1 = []
            tensor1_1d = tensor1.reshape(-1)
            values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
            idxs_list = unravel_index(idxs_1d.cpu().numpy(), tensor1.shape)
            # (dim0, dim1, dim2, ...)

            # reconstruct
            for i_beam in range(n_topk):
                idxs11 = []
                for idxs_list1 in idxs_list:
                    idxs11.append(idxs_list1[i_beam])
                idxs1.append(idxs11)
            idxs.append(idxs1)

    else:
        tensor1 = tensor
        idxs1 = []
        tensor1_1d = tensor1.reshape(-1)
        values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
        idxs_list = unravel_index(idxs_1d.numpy(), tensor1.shape)
        # (dim0, dim1, dim2, ...)

        # reconstruct
        for i_beam in range(n_topk):
            idxs11 = []
            for idxs_list1 in idxs_list:
                idxs11.append(idxs_list1[i_beam])
            idxs1.append(idxs11)
        idxs = idxs1
    return idxs


def remap_sc_idx(idxs, pr_sc_beam):
    for b, idxs1 in enumerate(idxs):
        for i_beam, idxs11 in enumerate(idxs1):
            sc_beam_idx = idxs[b][i_beam][0]
            sc_idx = pr_sc_beam[b][sc_beam_idx]
            idxs[b][i_beam][0] = sc_idx

    return idxs


def check_sc_sa_pairs(tb, pr_sc, pr_sa, ):
    """
    Check whether pr_sc, pr_sa are allowed pairs or not.
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    """
    bS = len(pr_sc)
    check = [False] * bS
    for b, pr_sc1 in enumerate(pr_sc):
        pr_sa1 = pr_sa[b]
        hd_types1 = tb[b]['types']
        hd_types11 = hd_types1[pr_sc1]
        if hd_types11 == 'text':
            if pr_sa1 == 0 or pr_sa1 == 3:  # ''
                check[b] = True
            else:
                check[b] = False

        elif hd_types11 == 'real':
            check[b] = True
        else:
            raise Exception("New TYPE!!")

    return check


def pred_wvi_se_beam(max_wn, s_wv, beam_size):
    """
    s_wv: [B, 4, mL, 2]
    - predict best st-idx & ed-idx
    output:
    pr_wvi_beam = [B, max_wn, n_pairs, 2]. 2 means [st, ed].
    prob_wvi_beam = [B, max_wn, n_pairs]
    """
    bS = s_wv.shape[0]

    # [B, 4, mL, 2] -> [B, 4, mL, 1], [B, 4, mL, 1]
    s_wv_st, s_wv_ed = s_wv.split(1, dim=3)

    s_wv_st = s_wv_st.squeeze(3)  # [B, 4, mL, 1] -> [B, 4, mL]
    s_wv_ed = s_wv_ed.squeeze(3)

    prob_wv_st = F.softmax(s_wv_st, dim=-1).detach().to('cpu').numpy()
    prob_wv_ed = F.softmax(s_wv_ed, dim=-1).detach().to('cpu').numpy()

    k_logit = int(ceil(sqrt(beam_size)))
    n_pairs = k_logit**2
    assert n_pairs >= beam_size
    values_st, idxs_st = s_wv_st.topk(k_logit)  # [B, 4, mL] -> [B, 4, k_logit]
    values_ed, idxs_ed = s_wv_ed.topk(k_logit)  # [B, 4, mL] -> [B, 4, k_logit]

    # idxs = [B, k_logit, 2]
    # Generate all possible combination of st, ed indices & prob
    pr_wvi_beam = []  # [B, max_wn, k_logit**2 [st, ed] paris]
    prob_wvi_beam = zeros([bS, max_wn, n_pairs])
    for b in range(bS):
        pr_wvi_beam1 = []

        idxs_st1 = idxs_st[b]
        idxs_ed1 = idxs_ed[b]
        for i_wn in range(max_wn):
            idxs_st11 = idxs_st1[i_wn]
            idxs_ed11 = idxs_ed1[i_wn]

            pr_wvi_beam11 = []
            pair_idx = -1
            for i_k in range(k_logit):
                for j_k in range(k_logit):
                    pair_idx += 1
                    st = idxs_st11[i_k].item()
                    ed = idxs_ed11[j_k].item()
                    pr_wvi_beam11.append([st, ed])

                    p1 = prob_wv_st[b, i_wn, st]
                    p2 = prob_wv_ed[b, i_wn, ed]
                    prob_wvi_beam[b, i_wn, pair_idx] = p1*p2
            pr_wvi_beam1.append(pr_wvi_beam11)
        pr_wvi_beam.append(pr_wvi_beam1)

    # prob

    return pr_wvi_beam, prob_wvi_beam


def convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_wp_t, wp_to_wh_index, nlu):
    """
    - Convert to the string in whilte-space-separated tokens
    - Add-hoc addition.
    """
    pr_wv_str_wp = []  # word-piece version
    pr_wv_str = []
    for b, pr_wvi1 in enumerate(pr_wvi):
        pr_wv_str_wp1 = []
        pr_wv_str1 = []
        wp_to_wh_index1 = wp_to_wh_index[b]
        nlu_wp_t1 = nlu_wp_t[b]
        nlu_t1 = nlu_t[b]

        for i_wn, pr_wvi11 in enumerate(pr_wvi1):
            st_idx, ed_idx = pr_wvi11

            # Ad-hoc modification of ed_idx to deal with wp-tokenization effect.
            # e.g.) to convert "butler cc (" ->"butler cc (ks)" (dev set 1st question).
            pr_wv_str_wp11 = nlu_wp_t1[st_idx:ed_idx+1]
            pr_wv_str_wp1.append(pr_wv_str_wp11)

            st_wh_idx = wp_to_wh_index1[st_idx]
            ed_wh_idx = wp_to_wh_index1[ed_idx]
            pr_wv_str11 = nlu_t1[st_wh_idx:ed_wh_idx+1]

            pr_wv_str1.append(pr_wv_str11)

        pr_wv_str_wp.append(pr_wv_str_wp1)
        pr_wv_str.append(pr_wv_str1)

    return pr_wv_str, pr_wv_str_wp


def merge_wv_t1_eng(where_str_tokens, NLq):
    """
    Almost copied of SQLNet.
    The main purpose is pad blank line while combining tokens.
    """
    nlq = NLq.lower()
    where_str_tokens = [tok.lower() for tok in where_str_tokens]
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$'
    special = {'-LRB-': '(',
               '-RRB-': ')',
               '-LSB-': '[',
               '-RSB-': ']',
               '``': '"',
               '\'\'': '"',
               }
    # '--': '\u2013'} # this generate error for test 5661 case.
    ret = ''
    double_quote_appear = 0
    for raw_w_token in where_str_tokens:
        # if '' (empty string) of None, continue
        if not raw_w_token:
            continue

        # Change the special characters
        # maybe necessary for some case?
        w_token = special.get(raw_w_token, raw_w_token)

        # check the double quote
        if w_token == '"':
            double_quote_appear = 1 - double_quote_appear

        # Check whether ret is empty. ret is selected where condition.
        if len(ret) == 0:
            pass
        # Check blank character.
        elif len(ret) > 0 and ret + ' ' + w_token in nlq:
            # Pad ' ' if ret + ' ' is part of nlq.
            ret = ret + ' '

        elif len(ret) > 0 and ret + w_token in nlq:
            pass  # already in good form. Later, ret + w_token will performed.

        # Below for unnatural question I guess. Is it likely to appear?
        elif w_token == '"':
            if double_quote_appear:
                ret = ret + ' '  # pad blank line between next token when " because in this case, it is of closing apperas
                # for the case of opening, no blank line.

        elif w_token[0] not in alphabet:
            pass  # non alphabet one does not pad blank line.

        # when previous character is the special case.
        elif (ret[-1] not in ['(', '/', '\u2013', '#', '$', '&']) and (ret[-1] != '"' or not double_quote_appear):
            ret = ret + ' '
        ret = ret + w_token

    return ret.strip()
