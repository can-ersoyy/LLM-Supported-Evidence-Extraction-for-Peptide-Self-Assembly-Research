"""This file includes parser methods that turn raw full text into formats that can be cited by llms"""

from bs4 import Tag
from bs4 import BeautifulSoup
import spacy


class Node:
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


class SentenceNode(
    Node
):  # a bit trickier than others since there is no specific xml tag
    pass


class TableNode(Node):
    def __init__(self, id, content, parent_id, caption, columns, rows):
        super().__init__(id, content, parent_id)
        self.caption = caption
        self.columns = columns
        self.rows = rows


# specials (these will have additional fields and logic)
# class ImageNode(Node):
# pass


def get_body_tag_from_xml_root(xml_string: str) -> Tag:
    """Gets the body tag (entry point) from xml string"""

    body = BeautifulSoup(xml_string, "xml").find("body")
    return body


def construct_node_id(parent_id: str, current_id: str) -> str:
    """Constructs a node ID from a parent ID and current ID."""

    if parent_id is None or parent_id == "":
        return current_id
    return f"{parent_id}/{current_id}"


def add_sentences(root: RootNode):
    """Fills the node tree with sentences. Uses spacy model en_core_sci_sm for sentence segmentation."""

    nlp = spacy.load(
        "en_core_sci_sm",
        config={
            "vars": {"include_static_vectors": False},
            "components": {
                "tok2vec": {"model": {"embed": {"include_static_vectors": False}}}
            },
        },
    )

    def _add_sentences(root, nlp):

        for node in root.children:
            if isinstance(node, ParagraphNode):
                doc = nlp(node.content)
                sentences = doc.sents
                for i, sentence in enumerate(sentences):
                    child_id = construct_node_id(node.id, f"s-{i}")
                    sentence_node = SentenceNode(child_id, sentence, node.id)
                    node.children.append(sentence_node)
            else:
                _add_sentences(node, nlp)

        return root

    return _add_sentences(root, nlp)


def parse_table_node(id, parent_id, table_wrap_tag: Tag) -> TableNode:
    caption_tag = table_wrap_tag.find("caption")
    caption = caption_tag.get_text(separator=" ", strip=True) if caption_tag else ""

    tr_tags = table_wrap_tag.find_all("tr")

    columns = []
    rows = []

    if tr_tags:
        header_tr = tr_tags[0]
        header_cells = header_tr.find_all(["th", "td"])
        columns = [cell.get_text(separator=" ", strip=True) for cell in header_cells]

        for tr in tr_tags[1:]:
            row_cells = tr.find_all(["td", "th"])
            rows.append(
                [cell.get_text(separator=" ", strip=True) for cell in row_cells]
            )

    content = caption

    return TableNode(
        id=id,
        content=content,
        parent_id=parent_id,
        caption=caption,
        columns=columns,
        rows=rows,
    )


def parse_xml(xml_string: str) -> RootNode:
    """Parses an XML string into a tree of Node objects."""

    root_tag = get_body_tag_from_xml_root(xml_string)

    def _parse(
        root_tag: Tag,
        parent_node: Node = None,
        parent_id: str = None,
        level_index: int = 0,
    ):

        if parent_node is None:
            node = RootNode("", "", None)
        else:
            id = construct_node_id(parent_id, f"{root_tag.name}-{level_index}")
            if root_tag.name == "sec":
                content = root_tag.find("title")
                node = SectionNode(id, content.text if content else "", parent_id)
            elif root_tag.name == "p":
                nested_tables = root_tag.find_all("table-wrap")
                if nested_tables:
                    # Clone or extract table text out of p so .text doesn't swallow it
                    p_copy = BeautifulSoup(str(root_tag), "xml").find("p")
                    for tw in p_copy.find_all("table-wrap"):
                        tw.decompose()  # Remove table from paragraph text (this is the copy)
                    content = p_copy.get_text(separator=" ", strip=True)
                else:
                    content = root_tag.get_text(separator=" ", strip=True)
                node = ParagraphNode(id, content, parent_id)
            elif root_tag.name == "table-wrap":
                id = construct_node_id(parent_id, f"table-{level_index}")
                node = parse_table_node(
                    id, parent_id, root_tag
                )  # returns TableNode directly
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
            _parse(child, node, node.id, tag_dict.get(child.name, 0))
            if child.name in tag_dict:
                tag_dict[child.name] += 1

        return node

    return _parse(root_tag)


def calculate_level_from_name(name: str) -> int:
    """Calculates the level of a node based on its name (ID). Utility function"""
    tags = name.split("/")
    level = len(tags)
    return level


def to_llm_string(paper_tree: RootNode) -> str:
    """Returns a given node tree into an llm readable format."""

    tab_level = calculate_level_from_name(paper_tree.id) - 1
    s = ""
    if not isinstance(paper_tree, RootNode):
        s = "\t" * tab_level + f"[{paper_tree.id}]"

    if isinstance(paper_tree, ParagraphNode):
        s += "\n"
    elif isinstance(paper_tree, SectionNode):
        s += f" {paper_tree.content}\n"
    elif isinstance(paper_tree, SentenceNode):
        s += f" {paper_tree.content}\n"
    elif isinstance(paper_tree, TableNode):
        s += f" {paper_tree.content}\n"

        indent = "\t" * tab_level

        header_str = " | ".join(paper_tree.columns)
        s += f"{indent}\tColumns: {header_str}\n"

        # rows with location annotation
        for idx, row in enumerate(paper_tree.rows):
            row_id = f"[{paper_tree.id}/row-{idx}]"
            row_str = " | ".join(str(cell) for cell in row)
            s += f"{indent}\t{row_id} {row_str}\n"

    for child in paper_tree.children:
        s += to_llm_string(child)

    return s


def to_debug_string(paper_tree: RootNode) -> str:
    """Returns a given node tree into a debug string."""
    """
    (RootNode
        (SectionNode
            (ParagraphNode)
        )
        (SectionNode
            (ParagraphNode
                (SentenceNode)
                (SentenceNode)
                (SentenceNode)
            )
            (ParagraphNode)
            (SectionNode
                (ParagraphNode
                    (SentenceNode)
                    (SentenceNode)
                    (TableNode)
                )
            )
        )
    )
    """

    def _render(node: Node, depth: int) -> list[str]:
        indent = "\t" * depth

        # Leaf node: collapsed on one line
        if not node.children:
            return [f"{indent}({type(node).__name__}, {node.id})"]

        # Parent node: open tag, render children, close tag
        lines = [f"{indent}({type(node).__name__}, {node.id}"]
        for child in node.children:
            lines.extend(_render(child, depth + 1))
        lines.append(f"{indent})")

        return lines

    return "\n".join(_render(paper_tree, 0))


def get_node_by_id(paper_tree: Node, id: str) -> Node:
    """Returns a node by its ID. Utility function for validation."""
    id = id.replace("[", "").replace("]", "")

    if paper_tree.id == id:
        return paper_tree
    for child in paper_tree.children:
        node = get_node_by_id(child, id)
        if node:
            return node
    return None


def extraction_ready_string_from_xml(xml_text: str) -> str:
    """Returns an extraction-ready string from an XML string."""

    root_node = parse_xml(xml_text)
    root_node = add_sentences(root_node)
    return to_llm_string(root_node)
