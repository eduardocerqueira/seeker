#date: 2022-03-03T16:52:34Z
#url: https://api.github.com/gists/a7c6b0adefeeb801362bb66cfba69f42
#owner: https://api.github.com/users/elisalikesblue

boxes = {
    "product 1" : {
        "box 1": 9.0,
        "box 2": 16.0,
        "box 3": 4.0,
        "box 4": 2.0
    },
    "product 2" : {
        "box 1": 3.0,
        "box 2": 5.0,
        "box 3": 4.0,
        "box 4": 2.0
    },
}

orders = {
    "order 1": {
        "product 1": 47,
        "product 2": 31,
    } 
}

def findBoxes(boxes, orders):
    packed_orders = {}
    for order in orders:
        packed_orders[order] = {}
        for product in orders[order]:
            product_name = product
            product_amount = orders[order][product]
            packed_orders[order][product] = {
                'packs': [],
                "quantity": 0
            }
            sorted_boxes = (sorted(dict(boxes[product_name]).items(), key=lambda x:x[1], reverse=True))
            sorted_boxes = dict(sorted_boxes)
           
            while product_amount >0:
                for box in sorted_boxes:
                    if boxes[product_name][box] <= product_amount:
                        
                        packed_orders[order][product]['quantity'] += boxes[product_name][box]
                        
                        packed_orders[order][product]['packs'].append({
                            box : boxes[product_name][box]
                        })
                        
                        product_amount = product_amount - boxes[product_name][box]
                        break
                    
                    if box == list(sorted_boxes)[-1] and product_amount < boxes[product_name][box]:
                        packed_orders[order][product]['quantity'] += product_amount
                        
                        packed_orders[order][product]['packs'].append({
                            box : product_amount
                        })
                        
                        product_amount = product_amount - boxes[product_name][box]
                        
                        
        return(packed_orders)
   print(find_boxes(boxes, orders))