import torch
#import torch_xla
#import torch_xla.core.xla_model as xm

device = torch.device("cuda")

def generate_sql_q(sql_i, tb):
    sql_q = []
    for b, sql_i1 in enumerate(sql_i):
        tb1 = tb[b]
        sql_q1 = generate_sql_q1(sql_i1, tb1)
        sql_q.append(sql_q1)

    return sql_q

def generate_sql_q1(sql_i1, tb1):
    """
        sql = {'sel': 5, 'agg': 4, 'conds': [[3, 0, '59']]}
        agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
        cond_ops = ['=', '>', '<', 'OP']
        Temporal as it can show only one-time conditioned case.
        sql_query: real sql_query
        sql_plus_query: More redable sql_query
        "PLUS" indicates, it deals with the some of db specific facts like PCODE <-> NAME
    """
    agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
    cond_ops = ['=', '>', '<', 'OP']

    headers = tb1["header"]
    # select_header = headers[sql['sel']].lower()
    # try:
    #     select_table = tb1["name"]
    # except:
    #     print(f"No table name while headers are {headers}")
    select_table = tb1["id"]

    select_agg = agg_ops[sql_i1['agg']]
    select_header = headers[sql_i1['sel']]
    sql_query_part1 = f'SELECT {select_agg}({select_header}) '


    where_num = len(sql_i1['conds'])
    if where_num == 0:
        sql_query_part2 = f'FROM {select_table}'
        # sql_plus_query_part2 = f'FROM {select_table}'

    else:
        sql_query_part2 = f'FROM {select_table} WHERE'
        # sql_plus_query_part2 = f'FROM {select_table_refined} WHERE'
        # ----------------------------------------------------------------------------------------------------------
        for i in range(where_num):
            # check 'OR'
            # number_of_sub_conds = len(sql['conds'][i])
            where_header_idx, where_op_idx, where_str = sql_i1['conds'][i]
            where_header = headers[where_header_idx]
            where_op = cond_ops[where_op_idx]
            if i > 0:
                sql_query_part2 += ' AND'
                # sql_plus_query_part2 += ' AND'

            sql_query_part2 += f" {where_header} {where_op} {where_str}"

    sql_query = sql_query_part1 + sql_query_part2
    # sql_plus_query = sql_plus_query_part1 + sql_plus_query_part2

    return sql_query

def report_detail(hds, nlu,
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    # print(f's_sc: {s_sc[0]}')
    # print(f's_sa: {s_sa[0]}')
    # print(f's_wn: {s_wn[0]}')
    # print(f's_wc: {s_wc[0]}')
    # print(f's_wo: {s_wo[0]}')
    # print(f's_wv: {s_wv[0][0]}')
    print(f'===============================')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    # print(f'pr_wvi: {pr_wvi}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'g_sql_q:  {g_sql_q}')
    print(f'pr_sql_q: {pr_sql_q}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx / cnt:.3f}, acc_x = {cnt_x / cnt:.3f}\n',
          f'acc_sc = {cnt_sc / cnt:.3f}, acc_sa = {cnt_sa / cnt:.3f}, acc_wn = {cnt_wn / cnt:.3f}\n',
          f'acc_wc = {cnt_wc / cnt:.3f}, acc_wo = {cnt_wo / cnt:.3f}, acc_wv = {cnt_wv / cnt:.3f}')
    print(f'===============================')

def generate_sql_q(sql_i, tb):
    sql_q = []
    for b, sql_i1 in enumerate(sql_i):
        tb1 = tb[b]
        sql_q1 = generate_sql_q1(sql_i1, tb1)
        sql_q.append(sql_q1)

    return sql_q

def generate_sql_q1(sql_i1, tb1):
    """
        sql = {'sel': 5, 'agg': 4, 'conds': [[3, 0, '59']]}
        agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
        cond_ops = ['=', '>', '<', 'OP']
        Temporal as it can show only one-time conditioned case.
        sql_query: real sql_query
        sql_plus_query: More redable sql_query
        "PLUS" indicates, it deals with the some of db specific facts like PCODE <-> NAME
    """
    agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
    cond_ops = ['=', '>', '<', 'OP']

    headers = tb1["header"]
    # select_header = headers[sql['sel']].lower()
    # try:
    #     select_table = tb1["name"]
    # except:
    #     print(f"No table name while headers are {headers}")
    select_table = tb1["id"]

    select_agg = agg_ops[sql_i1['agg']]
    select_header = headers[sql_i1['sel']]
    sql_query_part1 = f'SELECT {select_agg}({select_header}) '


    where_num = len(sql_i1['conds'])
    if where_num == 0:
        sql_query_part2 = f'FROM {select_table}'
        # sql_plus_query_part2 = f'FROM {select_table}'

    else:
        sql_query_part2 = f'FROM {select_table} WHERE'
        # sql_plus_query_part2 = f'FROM {select_table_refined} WHERE'
        # ----------------------------------------------------------------------------------------------------------
        for i in range(where_num):
            # check 'OR'
            # number_of_sub_conds = len(sql['conds'][i])
            where_header_idx, where_op_idx, where_str = sql_i1['conds'][i]
            where_header = headers[where_header_idx]
            where_op = cond_ops[where_op_idx]
            if i > 0:
                sql_query_part2 += ' AND'
                # sql_plus_query_part2 += ' AND'

            sql_query_part2 += f" {where_header} {where_op} {where_str}"

    sql_query = sql_query_part1 + sql_query_part2
    # sql_plus_query = sql_plus_query_part1 + sql_plus_query_part2

    return sql_query

