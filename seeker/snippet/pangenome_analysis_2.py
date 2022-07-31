#date: 2022-07-31T02:59:04Z
#url: https://api.github.com/gists/e1cb0fa0fc43ff3268bf90204108a047
#owner: https://api.github.com/users/JRlYun

import argparse
import os
from igraph import Graph
import re
import numpy as np
import pyfasta as pyf
from ete3 import Tree
import heapq
import random
import string
import argparse
import datetime
import itertools
import time
import sys
from multiprocessing import Pool
from copy import copy
import subprocess
import shlex


class Tools:

    @staticmethod
    def write_file(file_name, file_content, mode):
        """

        :param file_name:  file name
        :param file_content: content to write
        :param mode: 'a'/'w'
        :return:
        """
        with open(file_name, mode)as f:
            f.write(file_content)

    @staticmethod
    def read_file(file_name, mode):
        """

        :param file_name: file name
        :param mode: read: read / rl: readlines
        :return: file content
        """
        with open(file_name, 'r') as file:
            if mode == 'read':
                file_content = file.read()
            if mode == 'rl':
                file_content = file.readlines()
        return file_content

    @staticmethod
    def show_time():
        """
        Get current time
        :return: formatted time
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    @staticmethod
    def send_message(text, label='', out=False):
        """
        Print information on screen
        :param out: exit program or not
        :param text: the information will be sent.
        :param label: ERROR, WARNING, '' (for prompt)
        :return:
        """
        if label:
            print(f"[{Tools.show_time()}] {label}: {text}")
        else:
            print(f"[{Tools.show_time()}] {text}")
        if out:
            sys.exit(0)

    @staticmethod
    def progress_bar(ntd=0, ttn=100, scale=0.4, message='', symbol='▋'):
        """
        Print process bar on screen
        :param ntd: Number of Tasks Done
        :param ttn: Total Task Number
        :param scale: arrows ('>') length scaling
        :param message: extra information
        :param symbol: symbol to print
        :return:
        """
        h = int(ntd * 100 / ttn)  # percentage of number task done
        i = int(h * scale)  # length of arrow ('>')
        j = int(100 * scale - i)  # length of spaces
        arrows = symbol * i + ' ' * j
        sys.stdout.write(f'\r{message}' + arrows + f'[{h}%]')  # 在控制台输出
        sys.stdout.flush()  # 实时刷新标准输出
        if h == 100:
            # print(f'\n{message} finished!')
            pass

    @staticmethod
    def get_parser():
        # 生成argparse对象
        parser = argparse.ArgumentParser(description="pangenome analysis")
        parser.add_argument('-tr', '--tree_file', type=str, help='tree file in nwk')
        parser.add_argument('-fd', '--faa_dir', type=str, help='the directory of all genome faa')
        parser.add_argument('-t', '--threads', type=int, help='the threads used for orthofinder')
        parser.add_argument('-n', '--target_num', type=int, help='target number per clade')
        parser.add_argument('-ec1', '--expansion_coefficient_1', type=float,
                            help='expansion_coefficient of the first mcl')
        parser.add_argument('-ec2', '--expansion_coefficient_2', type=float,
                            help='expansion_coefficient of the second mcl')
        args = parser.parse_args()
        return args

    @staticmethod
    def orthofinder_time():
        """

        :return: formatted orthofinder time: jun_23
        """
        month = datetime.date.today().strftime("%b")
        day = datetime.date.today().strftime("%d")
        return month + str(day)

    @staticmethod
    def do_orthofinder(clade_dir, threads, expansion_coefficient):
        """

        :param clade_dir: the dir for orthofinder
        :param threads: run parallel
        :param expansion_coefficient: mcl clustering parameter
        :return:
        """
        cmd_orthofinder = f'orthofinder -og -f {clade_dir} -I {float(expansion_coefficient)} -t {threads}'
        os.system(cmd_orthofinder)



class PhyloTree(Tree):
    CLADEMEMBERDICT = {}
    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)


    def get_clades(self, target_node, target_numbers_per_clade):  # tree 选取出节点最多的
        parent = target_node.up  # 找深度最高的节点的父节点，只找一个
        all_tree_nodes = [i.name for i in self if i.name]  # tree的所有nodes的name
        # print('after_cut: %s,\n长度为 %d' % (str(all_tree_nodes), len(all_tree_nodes)))
        if parent:  # 对节点进行判断，有父节点的情况
            # print('parent:  %s\n长度为:%d' % (parent, len(parent)))
            child_num_from_parent = len(parent)
            if child_num_from_parent >= target_numbers_per_clade:  # 如果子分支clade算得的基因组数量大于等于设置的目标clade的基因组数量总数
                clades_list = [i.name for i in parent][0: target_numbers_per_clade]  # 提取到clade_list里面

                # 得到去除分支的剩余 leafs
                surplus_nodes = [i for i in all_tree_nodes if i not in clades_list]  # 剩余的tree上的节点的list

                #
                key_name = self.__class__.name_file(target_numbers_per_clade)
                # print(f'key_name={key_name}: \n clade_list={clades_list}')
                PhyloTree.CLADEMEMBERDICT[key_name] = clades_list


                # 去除树的分支
                self.prune(surplus_nodes)  # 按照clade_list切除树上的分支
                # print(tree)
                target_node_2 = self.get_target_node(self)  # 在修剪后的Tree上得到深度最高的节点之一的name
                self.get_clades(target_node_2, target_numbers_per_clade)  # 再把这个对象给当前tree进行重复处理

                # midpoint rooting
                # midpoint = tree.get_midpoint_outgroup()
                # tree.set_outgroup(midpoint)
            else:
                self.get_clades(parent, target_numbers_per_clade)  # 如果clade里面的数量小于设置的clade数量，将重复程序，找到其他的基因组来满足clade数量
        else:
            key_name = f'clade_last_{len(all_tree_nodes)}.txt'  # Tree无法修剪，最后的clade直接输出
            PhyloTree.CLADEMEMBERDICT[key_name] = all_tree_nodes



    @classmethod
    def name_file(cls, target_numbers_per_clade):
        name_after = ''.join(random.sample(list(string.ascii_letters), 6))
        file_name = f'clade_{name_after}_{target_numbers_per_clade}'  # clade内的数量
        if not os.path.exists(file_name):
            return file_name
        else:
            return cls.name_file(target_numbers_per_clade)

    @classmethod
    def get_children_parentNum(cls, node, num=0):
        # print(node.name)
        if node.is_root():  # 如果node没有parent，则返回true
            return num
        else:  # 如果node有parent,则num+1
            num += 1
            parent_node = node.up
            return cls.get_children_parentNum(parent_node, num)

    # 得到树上最深的节点之一
    @staticmethod
    def get_target_node(tree):
        all_node_parent_dict = {}  # key基因组name  values节点数量
        for i in tree:  # i tree里面每一个leaf的信息
            # print(i.name)
            a = tree.get_children_parentNum(i)
            all_node_parent_dict[i.name] = a  # leaf 的name 以及到根节点的parent的数量
        # print(all_node_parent_dict)
        num_list = set(all_node_parent_dict.values())  #
        max_2_list = heapq.nlargest(1, num_list)
        # print(max_2_list)
        node_most_parent = [k for k, v in all_node_parent_dict.items() if v in max_2_list][0]  # 找到最大深度的其中一个节点
        print(f'the most deep node in tree: {node_most_parent}')
        target_node = tree.search_nodes(name=node_most_parent)[0]  # 得到name的这个节点（对象）
        return target_node


# clade
class SubPanGenome:
    def __init__(self, clade_name: str = None,
                 cladeMember: list = None,
                 graph: object = None):
        self.clade_name = clade_name
        self.cladeMember = cladeMember
        self.graph = graph

    def getCladeMember(self, cladeName, faaDir):
        if not os.path.exists(cladeName):
            os.mkdir(cladeName)
        for member in self.cladeMember:
            cmd = f'cp {faaDir}/{member}.faa {cladeName}'
            subPro = subprocess.Popen(shlex.split(cmd))
            subPro.wait()

    # clade做orthofinder
    def clade_orthofinder(self, threads, expansion_coefficient):
        self.do_orthofinder(self.clade_name, threads, expansion_coefficient)

    # clade的构建网络
    def build_graph(self, GraphFileName, seq_ID_file):
        """
        :都在 Orthofinder的 WorkingDirectory目录下
        :param GraphFileName:  OrthoFinder_graph.txt
        :param seq_ID_file: SequenceIDs.txt
        :return: 每个clade的权重网络
        """
        GraphFile = Tools.read_file(GraphFileName, 'read')
        IDs = self.GetIDS(seq_ID_file)
        matrix = np.zeros(shape=(len(IDs), len(IDs)))
        graph_matrix = GraphFile.split('begin')[-1].strip().strip(')').replace('\n', '')
        for row in graph_matrix.split('$'):
            p1 = r'\d+\s'
            p2 = r'\d+:\d+.\d+\s'
            rowNumbers = re.findall(p1, row)
            if rowNumbers:
                rowN = int(rowNumbers[0].strip())
                colNums = re.findall(p2, row)
                if colNums:
                    for col in colNums:
                        colN = int(col.strip().split(':')[0])
                        values = float(col.strip().split(':')[1])
                        matrix[rowN, colN] = values
        self.graph = Graph.Weighted_Adjacency(matrix.tolist())
        self.graph.vs['name'] = IDs
        # self.graph = g
        return self.graph


    # 每个clade的og取三条并汇总成一个clade文件
    def get_three_sequence_per_og(self, ClusterFileName, Graph, all_clade_out_dir, faaDir):
        """
        :param ClusterFileName:  clusters_OrthoFinder_I(expansion_coefficient).txt
        :param Graph:   每个clade形成的网络
        :param clade:   每个clade的名字
        :expansion_coefficient: 膨胀系数
        :all_clade_out_dir: 选取的三条序列汇总之后存放的文件夹
        :return:
        """
        sequenceInf = self.getGenomesInf(faaDir)
        if not os.path.exists(all_clade_out_dir):
            os.mkdir(all_clade_out_dir)
        # Selected_Nodes_pair = {}
        # OG_IDs = {}
        ClusterFile = Tools.read_file(ClusterFileName, 'read')
        mcl_matrix = ClusterFile.split('begin')[-1].strip().strip(')').replace('\n', '')
        for row in mcl_matrix.split('$'):
            seqs = ''
            ClusterNumbers = re.findall(r'\d+\s', row)
            if ClusterNumbers:
                # OGName = ClusterNumbers[0]
                subGraphNumbers = [int(num.strip()) for num in ClusterNumbers[1:]]
                sub = Graph.subgraph(subGraphNumbers)
                # OG_IDs[OGName] = sub.vs['name']
                if len(subGraphNumbers) <= 3:
                    SelectNodes = sub.vs['name']
                else:
                    SelectNodes = [n['name'] for n in sorted(sub.vs, key=lambda X: X.degree(), reverse=True)[:3]]
                for k in SelectNodes:
                    seqs += f'>{k}\n{sequenceInf[k]}\n'
                clade_name = self.clade_name + '.fasta'
                Tools.write_file(os.path.join(all_clade_out_dir, clade_name), seqs, 'a')


    @staticmethod
    def getGenomesInf(genomeDir):
        """
        :param genomeDir: orthofinder的 og 目录
        :return:
        """
        genomeInf = {}
        for genome in os.listdir(genomeDir):
            if genome.split('.')[-1] in ['fa', 'faa', 'fasta', 'fna']:
                Fasta = pyf.Fasta(os.path.join(genomeDir, genome))
                for k, v in Fasta.items():
                    genomeInf[k.strip()] = v[:].strip()
        return genomeInf

    @staticmethod
    # seq_ID_file: SequenceIDs.txt
    def GetIDS(seq_ID_file):
        IDs = [IDs.strip().split(': ')[-1] for IDs in Tools.read_file(seq_ID_file, 'read').split('\n')]
        return IDs

    @staticmethod
    def do_orthofinder(clade_name, threads, expansion_coefficient):
        cmd_orthofinder = f'orthofinder -og -f {clade_name} -I {float(expansion_coefficient)} -t {threads}'
        os.system(cmd_orthofinder)


class SubGraph(Graph):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    tree = PhyloTree('tree_clade.nwk')
    try:
        target_node = PhyloTree.get_target_node(tree)
        tree.get_clades(target_node, 9)
    except Exception as e:
        pass

    print(tree.CLADEMEMBERDICT)
