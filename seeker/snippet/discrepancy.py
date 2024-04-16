#date: 2024-04-16T16:54:50Z
#url: https://api.github.com/gists/b021571ec2f6f738b1c5985caf61cfe0
#owner: https://api.github.com/users/fgenie

"""
우리의 예측 (지지난 주 회의): simple greedy 에 붙은 self-consistency each agent에 naive 하게 붙어있다
실제 구현: simple greedy model selection 전체 과정을 n번 수행하도록 되어있다. (얘도 rims + SC와 비슷하게 형태)
"""


def simple_greedy_self_consistency_our_thought(question):
    """ assumed combination of simple greedy + self-consistency """
    cot_ans = most_common([query_cot(question) for i in range(sc_num)])
    pal_ans = most_common([query_pal(question) for i in range(sc_num)])
    final_ans = cot_ans
    
    if cot_ans != pal_ans:
        final_ans = query_selection(cot_ans, pal_ans)
    return final_ans
  
  
def rims_self_consistency(question):
    """ current plan for rims+self-consistency loops in parallel """
    cot_ans = query_cot(question)
    pal_ans = query_pal(question)
    final_ans = cot_ans
    
    if cot_ans!=pal_ans:
        for i in range(sc_n):
            rims_ans = query_rims(x)
            final_ans = rims_ans
            answer_pool.append(final_ans)
        final_ans = answer_pool.most_common(1)[0]
    submission = final_ans
    return submission
            
        

# 실제 구현
# https://github.com/XuZhao0/Model-Selection-Reasoning/blob/8ee494276e958c3f88332d4be64f8a746395f11c/src/selection_math.py#L301
def simple_greedy_self_consistency_real(question):
    """ original simple_greedy + self-consistency  """
    for i in range(sc_n):
        # try majority agreement 
        cot_ans = query_cot(question)
        pal_ans = query_pal(question)
        final_ans = cot_ans
        if cot_ans!=pal_ans:
            # conflict --> selection per each inference
            selection_ans = query_selection(cot_ans, pal_ans)
            final_ans = selection_ans
        answer_pool.append(final_ans)
    
    submission = answer_pool.most_common(1)[0]
    return submission
  
  