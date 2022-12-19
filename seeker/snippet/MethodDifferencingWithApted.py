#date: 2022-12-19T16:41:52Z
#url: https://api.github.com/gists/9770e045bc88b360e2dd34dd5495038d
#owner: https://api.github.com/users/saikat107

# Dependencies 
# pip install tree-sitter;
# pip install apted;
# git clone https://github.com/tree-sitter/tree-sitter-c-sharp sitter-libs/cs;
# git clone https://github.com/tree-sitter/tree-sitter-python sitter-libs/py;
# git clone https://github.com/tree-sitter/tree-sitter-java sitter-libs/java;
# mkdir -p "parser";
################## Run the following code to generate parser ################
# from tree_sitter import Language
# import os
# libs = [os.path.join("sitter-libs", d) for d in os.listdir("sitter-libs")]
# Language.build_library(
    # Store the library in the `build` directory
#    'parser/languages.so',
#    libs,
# )
##############################################################################

import tree_sitter
from tree_sitter import Language, Parser
import os
from typing import List, Tuple, Dict
from apted import APTED, Config


def dfs_print(
    node: tree_sitter.Node,
    code: bytes,
    indent: int = 0
) -> None:
    """
    This function prints the tree structure of a given node.
    :param node: tree_sitter.Node, the node to be printed.
    :param indent: int, the number of indentations to be printed.
    :return: None
    """
    for _ in range(indent):
        print('\t', end='')
    _type = str(node.type)
    # get_code(node, code) if len(node.children) == 0 else ""
    value = str(node.id)
    print(f"{_type} {value}")
    for child in node.children:
        dfs_print(child, code, indent + 1)


def tokenize_code_without_comment(
    node: tree_sitter.Node,
    code: bytes
):
    tokens = "**********"
    stack = [node]
    while len(stack) > 0:
        top = stack.pop()
        if "comment" in str(top.type):
            continue
        elif "string" in str(top.type) or len(top.children) == 0:
            tokens.append(get_code(top, code))
        else:
            stack.extend(top.children[::-1])
    return " ".join(tokens)


def get_code(
    node: tree_sitter.Node,
    code: bytes
):
    return code[node.start_byte:node.end_byte].decode()


def get_method_signature(
    node: tree_sitter.Node,
    code: bytes
) -> str:
    tokens = "**********"
    if len(node.children) == 0:
        tokens.append(get_code(node, code))
    else:
        for child in node.children:
            if str(child.type) != 'block' and \
                str(child.type) != 'constructor_body':
                tokens.extend(get_method_signature(child, code))
    return tokens


def extract_package_name(
    full_ast: tree_sitter.Node,
    code
):
    stack = [full_ast]
    package_dec_node = None
    while len(stack) > 0:
        top = stack.pop()
        if str(top.type) == 'package_declaration':
            package_dec_node = top
            break
        stack.extend(top.children)
    if package_dec_node is not None:
        package_name = get_code(package_dec_node.children[1], code) + "."
    else:
        package_name = ""
    return package_name


def is_inner_method(method_node):
    root = method_node.parent
    while root is not None:
        if str(root.type) == 'method_declaration'or \
            str(root.type) == 'constructor_declaration':
            return True
        root = root.parent
    return False


def get_line_numbers_to_method_mapping(
    full_AST: tree_sitter.Node
) -> Dict[int, tree_sitter.Node]:
    mapping = {}
    stack = [full_AST]
    while len(stack) > 0:
        top = stack.pop()
        if top.type == 'method_declaration' or \
            str(top.type) == 'constructor_declaration':
            assert isinstance(top, tree_sitter.Node)
            sl, _ = top.start_point
            fl, _ = top.end_point
            for l in range(sl, fl+1):
                mapping[l] = top
        else:
            for child in top.children:
                stack.append(child)
    return mapping


def get_class_name(class_node, code):
    name = None
    for child in class_node.children:
        if str(child.type) == 'identifier':
            name = get_code(child, code)
            break
    if name is None:
        name = "<UNRESOLVED_CLASS_NAME>"
    return name


def get_method_name(method_node, code):
    name = None
    for child in method_node.children:
        if str(child.type) == 'identifier':
            name = get_code(child, code)
            break
    if name is None:
        name = "<UNRESOLVED_METHOD_NAME>"
    return name


def extract_qualified_method_name(
    method_node: tree_sitter.Node,
    code: bytes
) -> str:
    root = method_node
    class_name = None
    while str(root.type) != "class_declaration":
        if root.parent is None:
            class_name = "<UNRESOLVED_CLASS>"
            break
        root = root.parent
    if class_name is None:
        class_name = get_class_name(root, code)
    method_name = get_method_name(method_node, code)
    return class_name + "::" + method_name



class TreeSitterCodeEditConfig(Config):
    def __init__(self, code1, code2, ignore_comments=True):
        self.code1 = code1
        self.code2 = code2
        self.ignore_comments = ignore_comments

    def delete(self, node):
        """Calculates the cost of deleting a node"""
        type1 = str(node.type)
        if self.ignore_comments and "comment" in type1:
            return 0
        return 1

    def insert(self, node):
        """Calculates the cost of inserting a node"""
        type1 = str(node.type)
        if self.ignore_comments and "comment" in type1:
            return 0
        return 1

    def rename(self, node1, node2):
        value1 = self.code1[node1.start_byte:node1.end_byte].decode() if len(
            node1.children) == 0 else ""
        value2 = self.code2[node2.start_byte:node2.end_byte].decode() if len(
            node2.children) == 0 else ""
        type1 = str(node1.type)
        type2 = str(node2.type)
        if self.ignore_comments and (
            "comment" in type1 or "comment" in type2 # In the case of comments, we don't care what the values are.
        ):
            return 0
        if type1 == type2:
            if value1 == value2: 
                return 0
        return 1


class MethodDiffAnalyzer:
    def __init__(
        self,
        parser_path: str,
        language: str
    ) -> None:
        if not os.path.exists(parser_path):
            raise ValueError(
                f"Language parser does not exist at {parser_path}. Please run `setup.sh` to properly set the "
                f"environment!")
        self.lang_object = Language(parser_path, language)
        self.parser = Parser()
        self.parser.set_language(self.lang_object)

    def parse_code(
            self,
            file_path: str,
    ) -> Tuple[str, tree_sitter.Node]:
        """
        This function parses a given code and return the root node.
        :param file_path: str, the path to the file to be parsed.
        :return: tree_sitter.Node, the root node of the parsed tree.
        :return: bytes, the code of the file.
        """
        with open(file_path, 'r') as f:
            code = f.read().encode()
            tree = self.parser.parse(code)
            return code, tree.root_node

    def get_method_diffs(
        self,
        buggy_file_full_path: str,
        fixed_file_full_path: str
    ) -> List:
        buggy_code, buggy_ast = self.parse_code(buggy_file_full_path)
        fixed_code, fixed_ast = self.parse_code(fixed_file_full_path)
        buggy_package_name = extract_package_name(buggy_ast, buggy_code)
        fixed_package_name = extract_package_name(fixed_ast, fixed_code)
        config = TreeSitterCodeEditConfig(buggy_code, fixed_code)
        apted = APTED(buggy_ast, fixed_ast, config)
        mapping = apted.compute_edit_mapping()
        method_diffs = []
        for bm_node, fm_node in mapping:
            if not bm_node or not fm_node:
                continue
            if is_inner_method(bm_node) or is_inner_method(fm_node):
                continue
            if str(bm_node.type) == 'method_declaration' and \
                    str(fm_node.type) == 'method_declaration':
                m_apted = APTED(bm_node, fm_node, config)
                methos_dist = m_apted.compute_edit_distance()
                if methos_dist > 0:
                    buggy_method_sig = " ".join(
                        get_method_signature(bm_node, buggy_code))
                    buggy_method = get_code(bm_node, buggy_code)
                    buggy_method_fully_qualified_name = buggy_package_name + \
                        extract_qualified_method_name(bm_node, buggy_code)
                    fixed_method_sig = " ".join(
                        get_method_signature(fm_node, fixed_code))
                    fixed_method = get_code(fm_node, fixed_code)
                    fixed_method_fully_qualified_name = fixed_package_name + \
                        extract_qualified_method_name(bm_node, buggy_code)
                    method_diffs.append(
                        {
                            "buggy": {
                                "fully_qualified_name": buggy_method_fully_qualified_name,
                                "signature": buggy_method_sig,
                                "body": buggy_method
                            },
                            "fixed": {
                                "fully_qualified_name": fixed_method_fully_qualified_name,
                                "signature": fixed_method_sig,
                                "body": fixed_method
                            }
                        }
                    )
        return method_diffs


if __name__ == "__main__":
    analyzer = MethodDiffAnalyzer(
        parser_path="parser/languages.so",
        language="java"
    )
    method_edits = analyzer.get_method_diffs(
        buggy_file_full_path="tests/buggy.java",
        fixed_file_full_path="tests/fixed.java"
    )
    import json
    print(json.dumps(method_edits, indent=4))
