"""This file includes parser methods that turn raw full text into formats that can be cited by llms"""

from bs4 import Tag
from bs4 import BeautifulSoup

class Node():
	def __init__(self, id, content, parent_id=None):
		self.id = id
		self.children = []
		self.parent_id = parent_id
		self.content = content

class RootNode(Node):
	pass
class SectionNode(Node):
	pass
class ParagraphNode(Node):
	pass
class SentenceNode(Node): # a bit trickier than others since there is no specific xml tag
	pass

# specials (these will have additional fields and logic)
#class ImageNode(Node):
#	pass
#class TableNode(Node):
#	pass

def construct_node_id(parent_id: str, current_id: str) -> str:
    """Constructs a node ID from a parent ID and current ID."""
    if parent_id is None or parent_id == "":
        return current_id
    return f"{parent_id}/{current_id}"


def parse_xml(root_tag: Tag, parent_node: Node = None, parent_id: str = None, level_index: int = 0) -> RootNode:
    """Parses an XML string into a tree of Node objects."""
    
    if parent_node is None:
        node = RootNode("", "", None)
    else:
        id = construct_node_id(parent_id, f"{root_tag.name}-{level_index}")
        if root_tag.name == "sec":
            content = root_tag.find("title")
            node = SectionNode(id, content.text if content else "", parent_id)
        elif root_tag.name == "p":
            content = root_tag.text
            node = ParagraphNode(id, content, parent_id)
        else:
            return
        parent_node.children.append(node)

    
    children = root_tag.find_all(recursive=False)
    
    tag_dict = {
        "sec": 0,
        "p": 0,
    }
    
    for i in range(len(children)):
        child = children[i]   
        parse_xml(child, node, node.id, tag_dict.get(child.name, 0))
        if child.name in tag_dict:
            tag_dict[child.name] += 1

    return node

def to_llm_string(paper_tree: RootNode) -> str:
    pass

def to_debug_string(paper_tree: RootNode, children_level: int = 1) -> str:
    s = f"ID: {paper_tree.id} "
    s += f"Content: {paper_tree.content} "
    if paper_tree.children and children_level > 0:
        s += f"Children:\n"
        for child in paper_tree.children:
            s += to_debug_string(child, children_level - 1)
    return s

if __name__ == "__main__":
    test_xml_2 = ""
    soup = BeautifulSoup(test_xml_2, 'xml')
    root_tag = soup.find()
    tree = parse_xml(root_tag)
    
    print(to_debug_string(tree,3))




#def validate_field
    

