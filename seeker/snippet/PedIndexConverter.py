#date: 2023-08-10T17:02:35Z
#url: https://api.github.com/gists/4564584eeb32c5008b12cbcd2106ef63
#owner: https://api.github.com/users/potass13

class PedIndex():
    """
    0以上の整数を入力してそれに対応する血統上の親を出力する。
    ped_indexナンバリング方法 : 父(0)、母(1)、父父(2)、父母(3)、母父(4)、母母(5)、父父父(6)、父父母(7)、父母父(8)、父母母(9)、...
    """
    def __init__(self, ped_index: int=0):
        self.set_ped_index(ped_index)   
    
    def set_ped_index(self, ped_index: int):
        """
        0以上の整数を入力してそれに対応する血統上の親を格納する。
        """
        gen = -1
        string = ''
        tmp_index = ped_index
        str_parents = lambda x: '母' if x > 0 else '父'
        nth_bit_out = lambda x, nth: int(bin((x & 2**(nth-1)) >> (nth - 1)), 2) # n桁目のbitを出力
        
        if tmp_index >= 0 and tmp_index < 2**20:
            for i in range(1, 21):
                if tmp_index < 2**i:
                    gen = i
                    break
                else:
                    tmp_index = tmp_index - 2**i
        
        if gen > 0:
            state = ped_index - (0 if gen == 1 else 2**(gen)-2) # 2項目はsum([2**i for i in range(1, gen)])を計算している。なお、gen=1のときはsum(.)はゼロ。            
            for nth in range(gen, 0, -1):
                string = string + str_parents(nth_bit_out(state, nth))
        else:
            print('[ERROR] entered value is not valid. use set_ped_index-method.')
            state = -1
            string = 'ERROR'
        
        self._ped_index = ped_index 
        self._ped = string # 例: '父母父'
        self._ped_num_in_generation = state # 同世代内でのナンバリング。stateを2進数表示にして各bitの0->父、1->母に変換したものがself.pedに一致する。
        
        # 父、母のindexを取得する
        gen = len(self._ped) # ped_indexの世代
        tmp = 2**(gen+1)-2 # gen世代までの累積頭数。これはsum([2**i for i in range(1, gen+1)])を計算している。
        father_num_in_generation = self._ped_num_in_generation << 1 

        self._father_index = tmp+father_num_in_generation
        self._mother_index = self._father_index+1
        return
    
    def ped(self):
        """
        '父父母'といった形式で出力する。
        """
        return self._ped

    def get_dict(self):
        """
        値を辞書式で出力する。
        """
        return {'ped_index': self._ped_index, 'ped': self._ped, 'ped_num_in_generation': self._ped_num_in_generation}
    
    def get_father_index(self):
        """
        ped_indexの父のped_indexを出力する。
        """
        return self._father_index
    
    def get_mother_index(self):
        """
        ped_indexの母のped_indexを出力する。
        """
        return self._mother_index

class Node():
    """
    ノードのクラス。
    father, motherは該当するnodeのvalueが格納される。
    """
    def __init__(self, value: int, father: int=-2, mother: int=-2, name: str=None):
        self.value = value
        self.father = father
        self.mother = mother
        self.name = name
        
class PedIndexConverter():
    """
    netkeibaの血統表を上から順にスクレイピングした際には自然と2分探索木の深さ優先探索行きがけ順に相当する。
    具体的なナンバリング（max_generation=5）: 父(0)、父父(1)、父父父(2)、父父父父(3)、父父父父父(4)、父父父父母(5)、
                                            父父父母(6)、父父父母父(7)、父父父母母(8)、父父母(9)、父父母父(10)、父父母父父(11)、父父母父母(12)...
    この順序は非常に扱いづらく、max_generationが変わるとnk_indexが変わる厄介なものなので
    このナンバリング（nk_index）をPedIndexに基づくナンバリング（ped_index）に変換させる、そのためのクラス。
    """
    def __init__(self, max_generation: int=5):
        self._max_gen = max_generation
        self.root = Node(-1, 0, 1) # root nodeは-1とする
        tmp_dict = {self.root.value: self.root}
        total_num = 2**(self._max_gen+1)-2 # 1-max_generation世代までの血統表の馬の合計
            # 例えばmax_generation=5の場合は父系が(2**6-2)/2=31頭、牝系も同じく31頭、合計62頭分になる。
        f = lambda x: -2 if x >= total_num else x
        for i in range(total_num):
            pedi = PedIndex(i)
            tmp_dict[i] = Node(i, f(pedi.get_father_index()), f(pedi.get_mother_index())) # f(.)部分は(max_generation+1)世代へのnodeを無効にしている。
        self.node_dict = tmp_dict
        del tmp_dict
    
    def nk_index2ped_index(self, nk_index: int):
        """
        nk_index -> ped_indexに変換する。
        2分探索木の深さ優先探索行きがけ順の処理は再帰関数でなくpop-pushを使用している。
        """
        if nk_index >= 2**(self._max_gen+1)-2 or nk_index < 0:
            print('[ERROR] entered value is not valid. (nk_index = {}, max_generation = {})'.format(nk_index, self._max_gen))
            return -2
        
        bit_list = []
        bit_list.append(self.root.value)
        current_node = bit_list.pop()
        for i in range(0, nk_index+1):
            #print('{:02}: {}'.format(bit_list)) # pop-push確認用
            if self.node_dict[current_node].mother >= 0:
                bit_list.append(self.node_dict[current_node].mother)
            if self.node_dict[current_node].father >= 0: # father側を優先的に取り出すのでこのmother->fatherの順に処理している
                bit_list.append(self.node_dict[current_node].father)
                current_node = bit_list.pop()
        return self.node_dict[current_node].value
    
    def nk_index2ped(self, nk_index: int):
        """
        nk_index -> '父母父'というテキストに変換する。
        """
        ped_index = self.nk_index2ped_index(nk_index)
        return 'ERROR' if ped_index < 0 else PedIndex(ped_index).ped()