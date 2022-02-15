#date: 2022-02-15T17:03:49Z
#url: https://api.github.com/gists/f1f1dc413b8a02a118d1003590236016
#owner: https://api.github.com/users/vaibhavtmnit

textdataloader = DataLoader(textdataset,batch_size = 2,collate_fn = partial(collate_fn_bert, tokenizer))

for i in textdataloader:
    print(i)
    
"""
Output

({'input_ids': tensor([[  101,  1045,  2293,  1052, 22123,  2953,  2818,   102,     0],
        [  101,  1052, 22123,  2953,  2818,  2003,  1996,  2190,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]])}, tensor([1, 1]))
({'input_ids': tensor([[  101,  2045,  2024,  2500,  2021,  2498,  2004,  2204,  2004,  1052,
         22123,  2953,  2818,   102],
        [  101,  1052, 22123,  2953,  2818,  2003,  2925,   102,     0,     0,
             0,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])}, tensor([1, 1]))
({'input_ids': tensor([[  101, 20022,  2474,  1052, 22123,  2953,  2818,   102],
        [  101,  1052, 22123,  2953,  2818,  5749,   102,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0]])}, tensor([1, 1]))

"""