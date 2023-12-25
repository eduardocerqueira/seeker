#date: 2023-12-25T16:58:20Z
#url: https://api.github.com/gists/142c73276ac73cdd840ecb0216a181a8
#owner: https://api.github.com/users/shiroamurha


class Node():
    
    def __init__(self, value: int):
        self.value = value
        self.up = None
        self.down = {"left": None, "right": None}
        self.height = 0
    
    def insert(self, node):

        side = "right" if self.value < node.value else "left" if self.value > node.value else 0 
        if side == 0:
            return 
        
        if self.down.get(side) is None:
            self.down.update({side: node})
            node.up = self
            node.height = node.up.height + 1

        else:
            self.down.get(side).insert(node)

    def get_root_node(self):
        if self.up is not None:
            return self.up.get_root_node()
        else:
            return self

    def delete(self):

        not_none_sides = []
        
        if self.down['right'] is not None:
            not_none_sides.append(self.down['right'])

        if self.down['left'] is not None:
            not_none_sides.append(self.down['left'])

        top_node = self.get_root_node()
        #print(top_node)
        
        # deleting pointers to the node which will be deleted
        if self.up.down['right'] is self:
            self.up.down['right'] = None

        elif self.up.down['left'] is self:
            self.up.down['left'] = None

        self.up = None
        
        # reinserting nodes into the tree
        _ = [top_node.insert(side) for side in not_none_sides]


    def right(self):
        return self.down["right"] if self.down["right"] is not None else '⧅'

    def left(self):
        return self.down['left'] if self.down["left"] is not None else '⧅'
    
    def __str__(self):
        return str(self.value)
    
    def __int__(self):
        return self.value

#############

class Tree():

    def __init__(self, root: Node):
        self.root = root

    def search_node(self, value, node = None):
        node = self.root if node is None else node
        
        if value > node.value:
            if node.down['right'] is not None:
                return self.search_node(value, node.down['right'])
            else: 
                return None
        elif value < node.value: 
            if node.down['left'] is not None:
                return self.search_node(value, node.down['left'])
            else: 
                return None
        else:
            return node

    def insert(self, value):
        self.root.insert(Node(value))
    
    def remove(self, value):

        node_to_remove = self.search_node(value)

        if node_to_remove is None:
            print("value not contained")

        elif node_to_remove != self.root:
            node_to_remove.delete()
            print('removed')

        else:
            print("cannot remove root node")
    
    def in_order(self):

        print("In order ", end='')
        self.in_order_()
        print()
    
    def in_order_(self, node = None):

        node = self.root if node is None else node

        if node.down['left'] is not None:
            self.in_order_(node.down['left'])

        print(node.value, end=', ')

        if node.down['right'] is not None:
            self.in_order_(node.down['right'])
        
    
    def get_max_height(self, node = None):

        node = self.root if node is None else node

        if node.down['left'] is not None:
            return self.get_max_height(node.down['left'])

        if node.down['right'] is not None:
            return self.get_max_height(node.down['right'])
        
        return node.height


    def get_tree_line(self, line, node = None, level = 0):

        indent = self.get_max_height()*2 - level
        node = self.root if node is None else node

        if line == []:
            gap = '  ' * indent
            line.append(gap)
        
        if node.height == level:
            line.append(str(node) + ', ')

        elif node.down["left"] is not None or node.down["right"] is not None:

            if node.down["left"] is not None:
                self.get_tree_line(line, node.left(), level)
            
            if node.down["right"] is not None:
                self.get_tree_line(line, node.right(), level)

        else:
            if node.down["left"] is None:
                line.append(node.left() + ', ')    
            if node.down["right"] is None:
                line.append(node.right() + ', ')

        return line

    def print_tree(self):
        for i in range(self.get_max_height()+1):
            print(''.join(self.get_tree_line(line = [], level = i)), end='\n\n')
        
   
def example():
    tree = Tree(Node(10))
    nos = [5, 13, 3, 9, 4, 2, 11, 15, 14, 17]
    _ = [tree.insert(no) for no in nos]

    tree.print_tree()
    tree.in_order()
    
    #print(tree.root.left().left().right())

if __name__ == "__main__":
    
    print("""
        ========= menu =========
          1- Create tree
          2- Add value
          3- Remove value
          4- Search value
          5- In order
          6- Show Tree
          7- Example
          0- sair
    """)

    opt = input('>>> ')
    while(True):
        match(opt):
            case '1':   
                tree = Tree(Node(int(input('  valor inicial: '))))
            case '2':
                tree.insert(int(input('  inserir valor: ')))
            case '3':
                tree.remove(int(input('  remover valor: ')))
            case '4':
                print(tree.search_node(int(input('  procurar valor: '))))
            case '5':
                tree.in_order()
            case '6':
                tree.print_tree()
            case '7':
                example()
            case '0':    
                break
        opt = input('>>> ')



    
