import re
import uuid


class Node:
    """Constituency parse tree node."""
    
    def __init__(self, tag, children):
        self.tag = tag  # POS tag
        self.children = children  # Children
        self.text = ''  # Span string
        self.start_index = -1  # Beginning of span
        self.end_index = -1  # End of span (exclusive)
        
        # Metadata
        self.optional = False
        self.group = ''
        self.sent_index = -1
        self.node_index = -1
        self.label = -1
    
    def __repr__(self):
        return f'Node(i={self.start_index}, j={self.end_index}, label={self.tag}, text=\'{self.text}\')'

    def build(self):
        """ 
        Initializes root node by building string spans for each non-terminal
        constituent, clearing children of terminal constituents, and 
        initializing span boundaries (start, end) for each constituent.
        """
        
        def _build_string(node):
            tokens = []
            if isinstance(node, str):
                tokens = [node]
            elif isinstance(node, Node):
                for child in node.children:
                    tokens.extend(_build_string(child))
                node.text = ' '.join(tokens)
            return tokens
    
        def _clear_leaves(node):
            for child in node.children:
                if all(isinstance(subchild, str) for subchild in child.children):
                    child.children = []
                _clear_leaves(child)
        
        
        def _init_span_boundaries(node):
            if len(node.children):
                for child in node.children:
                    _init_span_boundaries(child)
                node.start_index = node.children[0].start_index
                node.end_index = node.children[-1].end_index
            else:
                node.start_index = _init_span_boundaries.n
                node.end_index = _init_span_boundaries.n + 1
                _init_span_boundaries.n += 1

        
        _build_string(self)
        _clear_leaves(self)
        _init_span_boundaries.n = 0
        _init_span_boundaries(self)
        
        return self
    
    
    def flatten(self):
        """Flattens constituency tree with pre-order traversal."""
        
        nodes = []
        stack = [self]
        while stack:
            node = stack.pop()
            nodes.append(node)
            for child in reversed(node.children):
                stack.append(child)
        return nodes

    def leaves(self):
        """Returns leaves of constituency tree."""

        nodes = []
        stack = [self]
        while stack:
            node = stack.pop()
            if len(node.children) == 0:
                nodes.append(node)
            for child in reversed(node.children):
                stack.append(child)
        return nodes
    
    def to_json(self):
        """Exports node metadata to json"""

        return {
            'tag': self.tag,
            'text': self.text,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'optional': self.optional,
            'group': self.group,
            'sent_index': self.sent_index,
            'node_index': self.node_index,
            'label': self.label,
        }


def build_tree(s):
    """Creates and initializes constituency parse tree."""
    
    open_b, close_b = '(', ')'
    open_pattern, close_pattern = re.escape(open_b), re.escape(close_b)
    node_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
    leaf_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
    token_re = re.compile(
        '%s\s*(%s)?|%s|(%s)'
        % (open_pattern, node_pattern, close_pattern, leaf_pattern)
    )

    # Walk through each token, updating a stack of trees.
    stack = [(None, [])]  # List of (node, children) tuples.
    for match in token_re.finditer(s):
        token = match.group()

        # Beginning of a tree/subtree.
        if token[0] == open_b:
            if len(stack) == 1 and len(stack[0][1]) > 0:
                raise ValueError
            label = token[1:].lstrip()
            stack.append((label, []))
        # End of a tree/subtree.
        elif token == close_b:
            if len(stack) == 1:
                if len(stack[0][1]) == 0:
                    raise ValueError
                else:
                    raise ValueError
            label, children = stack.pop()
            stack[-1][1].append(Node(label, children))
        # Leaf node.
        else:
            if len(stack) == 1:
                raise ValueError
            stack[-1][1].append(token)

    # Check that we got exactly one complete tree.
    if len(stack) > 1:
        raise ValueError
    elif len(stack[0][1]) == 0:
        raise ValueError
    else:
        assert stack[0][0] is None
        assert len(stack[0][1]) == 1

    tree = stack[0][1][0]
    
    # Build tree
    tree = tree.build()

    return tree


def _optional(node):
    """Sets node as optional for compression. Used in post-processing."""

    node.optional = True

    return node


def _group(*nodes):
    """Tags group of nodes with same uuid. Used in post-processing."""

    _uuid = str(uuid.uuid4().hex)
    for node in nodes:
        node.group = _uuid

    return nodes


def _node_exists(node):
    """Determines if node exists."""

    return node is not None


def _match_attribute(node, attr, candidates, reverse=False):
    """Matches node attribute with candidate attributes."""

    if reverse:
        return getattr(node, attr) not in candidates
    return getattr(node, attr) in candidates


def _sentence(i, node, parent_node, parent_children):
    """Removes sentence (S)."""

    if _match_attribute(node, 'tag', ('S',)):
        if (
            _node_exists(parent_node)
            and len(parent_children[i + 1:]) >= 1
            and _match_attribute(parent_children[i + 1], 'tag', (',',))
        ):
            return _group(
                node,
                _optional(parent_children[i + 1]),
            )

        return _group(node)

    return []


def _parenthetical(i, node, parent_node, parent_children):
    """Removes parenthetical (PRN)."""

    if _match_attribute(node, 'tag', ('PRN',)):
        if (
            i == 1
            and _node_exists(parent_node)
            and len(node.children) == 3
            and _match_attribute(node.children[0], 'text', ('-LRB-'))
            and _match_attribute(node.children[2], 'text', ('-RRB-'))
        ):
            return _group(
                _optional(parent_children[0]),
                node,
            )

        return _group(node)

    return []


def _fragment(i, node, parent_node, parent_children):
    """Removes fragment (FRAG)."""

    if _match_attribute(node, 'tag', ('FRAG',)):
        return _group(node)

    return []


def _adjective_phrases(i, node, parent_node, parent_children):
    """Removes adjectives (JJ) and adjective phrases (ADJP) in NPs."""

    if (
        _node_exists(parent_node)
        and _match_attribute(parent_node, 'tag', ('NP',))
        and _match_attribute(node, 'tag', ('JJ', 'JJR', 'JJS', 'ADJP'))
    ):
        return _group(node)

    return []


def _adverbial_phrases(i, node, parent_node, parent_children):
    """Removes adverbs (RB) and adverbial phrases (ADVP)."""

    if (
        _match_attribute(node, 'tag', ('ADVP', 'RB', 'RBR', 'RBS'))
        and _match_attribute(node, 'text', ("n't",), reverse=True)
    ):
        if (
            _node_exists(parent_node)
            and len(parent_children[i + 1:]) >= 1
            and _match_attribute(parent_children[i + 1], 'tag', (',',))
        ):
            return _group(
                parent_children[i],
                _optional(parent_children[i + 1]),
            )

        return _group(node)

    return []


def _relative_clause(i, node, parent_node, parent_children):
    """Removes relative clauses (SBAR)."""

    if _match_attribute(node, 'tag', 'SBAR'):
        return _group(node)

    return []


def _conjoined_noun_phrase(i, node, parent_node, parent_children):
    """Removes conjoined verb phrases (NP-[CC-NP])."""

    if (
        _node_exists(parent_node)
        and len(parent_children) >= 3
        and _match_attribute(parent_children[i - 2], 'tag', ('NP',))
        and _match_attribute(parent_children[i - 1], 'tag', ('CC',))
        and _match_attribute(parent_children[i], 'tag', ('NP',))
    ):
        return _group(
            _optional(parent_children[i - 1]),
            parent_children[i],
        )

    return []


def _conjoined_verb_phrase(i, node, parent_node, parent_children):
    """Removes conjoined verb phrases (VP-[CC-VP])."""

    if (
        _node_exists(parent_node)
        and len(parent_children) >= 3
        and _match_attribute(parent_children[i - 2], 'tag', ('VP',))
        and _match_attribute(parent_children[i - 1], 'tag', ('CC',))
        and _match_attribute(parent_children[i], 'tag', ('VP',))
    ):
        return _group(
            _optional(parent_children[i - 1]),
            parent_children[i],
        )

    return []


def _conjoined_sentences(i, node, parent_node, parent_children):
    """Removes conjoined sentences (S-[,-CC-S])."""

    # Case 1: Contains comma.
    if (
        _node_exists(parent_node)
        and len(parent_children[:i]) >= 3
        and _match_attribute(parent_children[i - 3], 'tag', ('S',))
        and _match_attribute(parent_children[i - 2], 'tag', (',',))
        and _match_attribute(parent_children[i - 1], 'tag', ('CC',))
        and _match_attribute(parent_children[i - 1], 'text', ('and', 'but',))
        and _match_attribute(parent_children[i], 'tag', ('S',))
    ):
        return _group(
            _optional(parent_children[i - 2]),
            _optional(parent_children[i - 1]),
            parent_children[i],
        )

    # Case 2: Doesn't contain comma.
    if (
        _node_exists(parent_node)
        and len(parent_children[:i]) >= 2
        and _match_attribute(parent_children[i - 2], 'tag', ('S',))
        and _match_attribute(parent_children[i - 1], 'tag', ('CC',))
        and _match_attribute(parent_children[i - 1], 'text', ('and', 'but',))
        and _match_attribute(parent_children[i], 'tag', ('S',))
    ):
        return _group(
            _optional(parent_children[i - 1]),
            parent_children[i],
        )

    return []


def _appositive_phrases(i, node, parent_node, parent_children):
    """Removes appositive noun phrases (NP-[,-NP-,])"""

    if (
        i == 3
        and _node_exists(parent_node)
        and len(parent_children) == 4
        and _match_attribute(parent_node, 'tag', ('NP',))
        and _match_attribute(parent_children[i - 3], 'tag', ('NP',))
        and _match_attribute(
            parent_children[i - 2],
            'text',
            (',', '(', '[', '{', '-LRB-', '-LSB-', '-LCB-'),
        )
        and _match_attribute(parent_children[i - 1], 'tag', ('NP',))
        and _match_attribute(
            parent_children[i],
            'text',
            (',', ')', ']', '}', '-RRB-', '-RSB-', '-RCB-'),
        )
    ):
        return _group(
            _optional(parent_children[i - 2]),
            parent_children[i - 1],
            _optional(parent_children[i]),
        )

    return []


def _prepositional_phrases(i, node, parent_node, parent_children):
    """Removes prepositional phrases (PP) in NP and VPs."""


    def _valid_verb_phrase(node):
        return (
            _match_attribute(
                node,
                'tag',
                ('VP', 'VB', 'VBD', 'VGB', 'VBN', 'VBP', 'VBZ', 'PRT'),
            )
            and _match_attribute(
                node,
                'text',
                ('are', 'be', 'is', 'was', 'were'),
                reverse=True,
            )
        )


    # Case 1: Modifying NP
    if (
        _node_exists(parent_node)
        and len(parent_children[:i]) >= 1
        and _match_attribute(node, 'tag', ('PP',))
    ):
        # Case 1a: Contains comma.
        if (
            len(parent_children[:i]) >= 2
            and _match_attribute(parent_children[i - 1], 'tag', (',',))
            and _match_attribute(parent_children[i - 2], 'tag', ('NP', 'PP',))
        ):
            # Case 1a': Also contains forwards comma.
            if (
                len(parent_children[i + 1:]) >= 1
                and _match_attribute(parent_children[i + 1], 'tag', (',',))
            ):
                return _group(
                    _optional(parent_children[i - 1]),
                    parent_children[i],
                    _optional(parent_children[i + 1]),
                )

            return _group(
                _optional(parent_children[i - 1]),
                parent_children[i],
            )

        # Case 1b: Doesn't contain comma.
        if _match_attribute(parent_children[i - 1], 'tag', ('NP', 'PP',)):
            return _group(parent_children[i])

    # Case 2: Modifying VP
    if (
        _node_exists(parent_node)
        and len(parent_children[:i]) >= 1
        and _match_attribute(node, 'tag', ('PP',))
    ):
        # Case 2a: Contains comma.
        if (
            len(parent_children[:i]) >= 2
            and _match_attribute(parent_children[i - 1], 'tag', (',',))
            and _valid_verb_phrase(parent_children[i - 2])
        ):
            # Case 2a': Also contains forwards comma.
            if (
                len(parent_children[i + 1:]) >= 1
                and _match_attribute(parent_children[i + 1], 'tag', (',',))
            ):
                return _group(
                    _optional(parent_children[i - 1]),
                    parent_children[i],
                    _optional(parent_children[i + 1]),
                )

            return _group(
                _optional(parent_children[i - 1]),
                parent_children[i],
            )

        # Case 2b: Doesn't contain comma.
        if _valid_verb_phrase(parent_children[i - 1]):
            return _group(parent_children[i])

    return []


def _intro_prepositional_phrases(i, node, parent_node, parent_children):
    """Removes introductory prepositional phrases (PP)."""

    if (
        i == 0
        and _node_exists(parent_node)
        and _match_attribute(node, 'tag', ('PP',))
    ):
        # Matches comma.
        if (
            len(parent_children[i + 1:]) >= 1
            and _match_attribute(parent_children[i + 1], 'tag', (',',))
        ):
            return _group(
                parent_children[i],
                _optional(parent_children[i + 1]),
            )

        return _group(parent_children[i])

    return []


def find_compressions(tree, include_optional=False):
    """Finds compressions for the constituency parse tree."""

    global rules


    def _search_tree(i, node, parent_node=None, parent_children=None):
        nodes = []
        for rule in rules:
            nodes.extend(rule(i, node, parent_node, parent_children))
        for i, child in enumerate(node.children):
            nodes.extend(
                _search_tree(i, child, node, node.children)
            )
        return nodes


    def _filter_duplicates(nodes):
        unique_spans = set()
        unique_nodes = []
        for node in nodes:
            if (node.start_index, node.end_index) in unique_spans:
                continue
            unique_spans.add((node.start_index, node.end_index))
            unique_nodes.append(node)
        return unique_nodes


    nodes = _filter_duplicates(
        _search_tree(
            i=0,
            node=tree,
            parent_node=None,
            parent_children=None,
        )
    )

    return (
        nodes if include_optional
        else [node for node in nodes if not node.optional]
    )


rules = [
    _sentence,
    _fragment,
    _parenthetical,
    _adjective_phrases,
    _adverbial_phrases,
    _relative_clause,
    _conjoined_noun_phrase,
    _conjoined_verb_phrase,
    _conjoined_sentences,
    _appositive_phrases,
    _prepositional_phrases,
    _intro_prepositional_phrases,
]
