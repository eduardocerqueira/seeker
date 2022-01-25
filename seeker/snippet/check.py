#date: 2022-01-25T16:53:27Z
#url: https://api.github.com/gists/37cc9e1f4a84e8977ce35b50be2b5b20
#owner: https://api.github.com/users/Maegner

import csv
import json

def has_more_than_one_tx():
    tx_per_add = {}
    suspect_list = set([])
    with open ('txs.csv') as file:
        data = csv.DictReader(file)
        for item in data:
            add = item['From']
            block = item['Blockno']
            value = float(item['Value_IN(ETH)'])
            if add in tx_per_add:
                tx_per_add[add]['num_transactions'] += 1
                tx_per_add[add]['value'] += value
                suspect_list.add(add)
            else:
                tx_per_add[add] = {'num_transactions': 1, 'value': value}

    with open ('more_than_one_tx.csv', 'w', encoding='UTF8', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['Address', 'Transaction Number','Total Value'])
        for add in suspect_list:
            writer.writerow([add,tx_per_add[add]['num_transactions'], tx_per_add[add]['value']])

def has_more_than_one_tx_per_block():
    tx_per_add_per_block = {}
    suspect_list = set([])
    with open ('txs.csv') as file:
        data = csv.DictReader(file)
        for item in data:
            add = item['From']
            block = item['Blockno']
            value = float(item['Value_IN(ETH)'])
            if add in tx_per_add_per_block:
                if block in tx_per_add_per_block[add]:
                    tx_per_add_per_block[add][block]['num_transactions'] += 1
                    tx_per_add_per_block[add][block]['value'] += value
                    suspect_list.add(add)
                else:
                    tx_per_add_per_block[add][block] = {'num_transactions': 1, 'value': value}
            else:
                tx_per_add_per_block[add] = { block: {'num_transactions': 1, 'value': value}}
    
    with open ('more_than_one_per_block.csv', 'w', encoding='UTF8', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['Address', 'Block', 'Transaction Number','Total Value'])
        total = 0
        totalTx = 0
        for add in suspect_list:
            for blockData in tx_per_add_per_block[add]:
                writer.writerow([
                    add,
                    blockData,
                    tx_per_add_per_block[add][blockData]['num_transactions'],
                    tx_per_add_per_block[add][blockData]['value']
                ])

has_more_than_one_tx_per_block()
has_more_than_one_tx()