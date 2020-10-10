import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pylab import *
from copy import deepcopy
#import torch_xla
#import torch_xla.core.xla_model as xm

device = torch.device("cuda")


def Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi):
    """
    :param s_wv: score  [ B, n_conds, T, score]
    :param g_wn: [ B ]
    :param g_wvi: [B, conds, pnt], e.g. [[[0, 6, 7, 8, 15], [0, 1, 2, 3, 4, 15]], [[0, 1, 2, 3, 16], [0, 7, 8, 9, 16]]]
    :return:
    """
    loss = 0
    loss += F.cross_entropy(s_sc, torch.tensor(g_sc).to(device))
    loss += F.cross_entropy(s_sa, torch.tensor(g_sa).to(device))
    loss += F.cross_entropy(s_wn, torch.tensor(g_wn).to(device))
    loss += Loss_wc(s_wc, g_wc)
    loss += Loss_wo(s_wo, g_wn, g_wo)
    loss += Loss_wv_se(s_wv, g_wn, g_wvi)

    return loss


def Loss_wc(s_wc, g_wc):

    # Construct index matrix
    bS, max_h_len = s_wc.shape
    im = torch.zeros([bS, max_h_len]).to(device)
    for b, g_wc1 in enumerate(g_wc):
        for g_wc11 in g_wc1:
            im[b, g_wc11] = 1.0
    # Construct prob.
    p = F.sigmoid(s_wc)
    loss = F.binary_cross_entropy(p, im)

    return loss


def Loss_wo(s_wo, g_wn, g_wo):

    # Construct index matrix
    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        if g_wn1 == 0:
            continue
        g_wo1 = g_wo[b]
        s_wo1 = s_wo[b]
        loss += F.cross_entropy(s_wo1[:g_wn1], torch.tensor(g_wo1).to(device))

    return loss

def Loss_wv_se(s_wv, g_wn, g_wvi):
    """
    s_wv:   [bS, 4, mL, 2], 4 stands for maximum # of condition, 2 tands for start & end logits.
    g_wvi:  [ [1, 3, 2], [4,3] ] (when B=2, wn(b=1) = 3, wn(b=2) = 2).
    """
    loss = 0
    # g_wvi = torch.tensor(g_wvi).to(device)
    for b, g_wvi1 in enumerate(g_wvi):
        # for i_wn, g_wvi11 in enumerate(g_wvi1):

        g_wn1 = g_wn[b]
        if g_wn1 == 0:
            continue
        g_wvi1 = torch.tensor(g_wvi1).to(device)
        g_st1 = g_wvi1[:,0]
        g_ed1 = g_wvi1[:,1]
        # loss from the start position
        loss += F.cross_entropy(s_wv[b,:g_wn1,:,0], g_st1)

        # print("st_login: ", s_wv[b,:g_wn1,:,0], g_st1, loss)
        # loss from the end position
        loss += F.cross_entropy(s_wv[b,:g_wn1,:,1], g_ed1)
        # print("ed_login: ", s_wv[b,:g_wn1,:,1], g_ed1, loss)

    return loss

def pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv):
    pr_sc = pred_sc(s_sc)
    pr_sa = pred_sa(s_sa)
    pr_wn = pred_wn(s_wn)
    pr_wc = pred_wc(pr_wn, s_wc)
    pr_wo = pred_wo(pr_wn, s_wo)
    pr_wvi = pred_wvi_se(pr_wn, s_wv)

    return pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi

def pred_sc(s_sc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc = []
    for s_sc1 in s_sc:
        pr_sc.append(s_sc1.argmax().item())

    return pr_sc

def pred_sc(s_sc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc = []
    for s_sc1 in s_sc:
        pr_sc.append(s_sc1.argmax().item())

    return pr_sc

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

def pred_wvi_se(wn, s_wv):
    """
    s_wv: [B, 4, mL, 2]
    - predict best st-idx & ed-idx
    """

    s_wv_st, s_wv_ed = s_wv.split(1, dim=3)  # [B, 4, mL, 2] -> [B, 4, mL, 1], [B, 4, mL, 1]

    s_wv_st = s_wv_st.squeeze(3) # [B, 4, mL, 1] -> [B, 4, mL]
    s_wv_ed = s_wv_ed.squeeze(3)

    pr_wvi_st_idx = s_wv_st.argmax(dim=2) # [B, 4, mL] -> [B, 4, 1]
    pr_wvi_ed_idx = s_wv_ed.argmax(dim=2)

    pr_wvi = []
    for b, wn1 in enumerate(wn):
        pr_wvi1 = []
        for i_wn in range(wn1):
            pr_wvi_st_idx11 = pr_wvi_st_idx[b][i_wn]
            pr_wvi_ed_idx11 = pr_wvi_ed_idx[b][i_wn]
            pr_wvi1.append([pr_wvi_st_idx11.item(), pr_wvi_ed_idx11.item()])
        pr_wvi.append(pr_wvi1)

    return pr_wvi

def convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_wp_t, wp_to_wh_index, nlu):
    """
    - Convert to the string in whilte-space-separated tokens
    - Add-hoc addition.
    """
    pr_wv_str_wp = [] # word-piece version
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

def sort_pr_wc(pr_wc, g_wc):
    """
    Input: list
    pr_wc = [B, n_conds]
    g_wc = [B, n_conds]
    Return: list
    pr_wc_sorted = [B, n_conds]
    """
    pr_wc_sorted = []
    for b, pr_wc1 in enumerate(pr_wc):
        g_wc1 = g_wc[b]
        pr_wc1_sorted = []

        if set(g_wc1) == set(pr_wc1):
            pr_wc1_sorted = deepcopy(g_wc1)
        else:
            # no sorting when g_wc1 and pr_wc1 are different.
            pr_wc1_sorted = deepcopy(pr_wc1)

        pr_wc_sorted.append(pr_wc1_sorted)
    return pr_wc_sorted

def generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu):
    pr_sql_i = []
    for b, nlu1 in enumerate(nlu):
        conds = []
        for i_wn in range(pr_wn[b]):
            conds1 = []
            conds1.append(pr_wc[b][i_wn])
            conds1.append(pr_wo[b][i_wn])
            merged_wv11 = merge_wv_t1_eng(pr_wv_str[b][i_wn], nlu[b])
            conds1.append(merged_wv11)
            conds.append(conds1)

        pr_sql_i1 = {'agg': pr_sa[b], 'sel': pr_sc[b], 'conds': conds}
        pr_sql_i.append(pr_sql_i1)
    return pr_sql_i

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
        w_token = special.get(raw_w_token, raw_w_token)  # maybe necessary for some case?

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

def get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                    pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                    g_sql_i, pr_sql_i,
                    mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_sc = get_cnt_sc_list(g_sc, pr_sc)
    cnt_sa = get_cnt_sc_list(g_sa, pr_sa)
    cnt_wn = get_cnt_sc_list(g_wn, pr_wn)
    cnt_wc = get_cnt_wc_list(g_wc, pr_wc)
    cnt_wo = get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode)
    if pr_wvi:
        cnt_wvi = get_cnt_wvi_list(g_wn, g_wc, g_wvi, pr_wvi, mode)
    else:
        cnt_wvi = [0]*len(cnt_sc)
    cnt_wv = get_cnt_wv_list(g_wn, g_wc, g_sql_i, pr_sql_i, mode) # compare using wv-str which presented in original data.


    return cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wvi, cnt_wv

def get_cnt_sc_list(g_sc, pr_sc):
    cnt_list = []
    for b, g_sc1 in enumerate(g_sc):
        pr_sc1 = pr_sc[b]
        if pr_sc1 == g_sc1:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list

def get_cnt_wc_list(g_wc, pr_wc):
    cnt_list= []
    for b, g_wc1 in enumerate(g_wc):

        pr_wc1 = pr_wc[b]
        pr_wn1 = len(pr_wc1)
        g_wn1 = len(g_wc1)

        if pr_wn1 != g_wn1:
            cnt_list.append(0)
            continue
        else:
            wc1 = array(g_wc1)
            wc1.sort()

            if array_equal(pr_wc1, wc1):
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list

def get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode):
    """ pr's are all sorted as pr_wc are sorted in increasing order (in column idx)
        However, g's are not sorted.
        Sort g's in increasing order (in column idx)
    """
    cnt_list=[]
    for b, g_wo1 in enumerate(g_wo):
        g_wc1 = g_wc[b]
        pr_wc1 = pr_wc[b]
        pr_wo1 = pr_wo[b]
        pr_wn1 = len(pr_wo1)
        g_wn1 = g_wn[b]

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            # Sort based wc sequence.
            if mode == 'test':
                idx = argsort(array(g_wc1))

                g_wo1_s = array(g_wo1)[idx]
                g_wo1_s = list(g_wo1_s)
            elif mode == 'train':
                # due to tearch forcing, no need to sort.
                g_wo1_s = g_wo1
            else:
                raise ValueError

            if type(pr_wo1) != list:
                raise TypeError
            if g_wo1_s == pr_wo1:
                cnt_list.append(1)
            else:
                cnt_list.append(0)
    return cnt_list

def get_cnt_wvi_list(g_wn, g_wc, g_wvi, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_list =[]
    for b, g_wvi1 in enumerate(g_wvi):
        g_wc1 = g_wc[b]
        pr_wvi1 = pr_wvi[b]
        pr_wn1 = len(pr_wvi1)
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = argsort(array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi11 = g_wvi1[idx11]
                pr_wvi11 = pr_wvi1[i_wn]
                if g_wvi11 != pr_wvi11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list

def get_cnt_wv_list(g_wn, g_wc, g_sql_i, pr_sql_i, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_list =[]
    for b, g_wc1 in enumerate(g_wc):
        pr_wn1 = len(pr_sql_i[b]["conds"])
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = argsort(array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi_str11 = str(g_sql_i[b]["conds"][idx11][2]).lower()
                pr_wvi_str11 = str(pr_sql_i[b]["conds"][i_wn][2]).lower()
                # print(g_wvi_str11)
                # print(pr_wvi_str11)
                # print(g_wvi_str11==pr_wvi_str11)
                if g_wvi_str11 != pr_wvi_str11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list

def get_cnt_lx_list(cnt_sc1, cnt_sa1, cnt_wn1, cnt_wc1, cnt_wo1, cnt_wv1):
    # all cnt are list here.
    cnt_list = []
    cnt_lx = 0
    for csc, csa, cwn, cwc, cwo, cwv in zip(cnt_sc1, cnt_sa1, cnt_wn1, cnt_wc1, cnt_wo1, cnt_wv1):
        if csc and csa and cwn and cwc and cwo and cwv:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list

def get_cnt_x_list(engine, tb, g_sc, g_sa, g_sql_i, pr_sc, pr_sa, pr_sql_i):
    cnt_x1_list = []
    g_ans = []
    pr_ans = []
    for b in range(len(g_sc)):
        g_ans1 = engine.execute(tb[b]['id'], g_sc[b], g_sa[b], g_sql_i[b]['conds'])
        # print(f'cnt: {cnt}')
        # print(f"pr_sql_i: {pr_sql_i[b]['conds']}")
        try:
            pr_ans1 = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], pr_sql_i[b]['conds'])

            if bool(pr_ans1):  # not empty due to lack of the data from incorretly generated sql
                if g_ans1 == pr_ans1:
                    cnt_x1 = 1
                else:
                    cnt_x1 = 0
            else:
                cnt_x1 = 0
        except:
            # type error etc... Execution-guided decoding may be used here.
            pr_ans1 = None
            cnt_x1 = 0
        cnt_x1_list.append(cnt_x1)
        g_ans.append(g_ans1)
        pr_ans.append(pr_ans1)

    return cnt_x1_list, g_ans, pr_ans
