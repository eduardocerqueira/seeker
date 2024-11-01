#date: 2024-11-01T16:46:54Z
#url: https://api.github.com/gists/b206fa843035be121bb8b5649ca68faa
#owner: https://api.github.com/users/zmerzouki

import requests
import json

# Connect to ERP system
erp_url = "https://erp.example.com/api/orders"
erp_response = requests.get(erp_url, headers={"Authorization": "**********"
erp_orders = erp_response.json()

# Data transformation function for MES
def transform_order_for_mes(order):
    return {
        "order_id": order["id"],
        "production_line": order["production_line"],
        "quantity": order["quantity"],
        "due_date": order["due_date"]
    }

# Connect to MES and send transformed data
mes_url = "https://mes.example.com/api/orders"
for order in erp_orders:
    mes_order = transform_order_for_mes(order)
    mes_response = requests.post(mes_url, json=mes_order, headers={"Authorization": "**********"
    if mes_response.status_code == 201:
        print(f"Order {mes_order['order_id']} synced successfully with MES.")
    else:
        print(f"Failed to sync order {mes_order['order_id']} with MES.")

'order_id']} with MES.")

