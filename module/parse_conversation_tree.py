from dataclasses import dataclass
from anytree import Node, RenderTree
from .types import Conversation, ConversationMapping, MessageRecord, ChatMessage


@dataclass
class ConversationMappingWrapper:
    _mapping: ConversationMapping
    _root: MessageRecord

    def __init__(self, mapping: ConversationMapping):
        self._mapping = mapping
        try:
            self._root = next((node for node in mapping.values() if not node.parent))
        except StopIteration:
            raise ValueError("No root node found in mapping")

    def __getitem__(self, key):
        return self._mapping[key]
    
    def __iter__(self):
        return iter(self._mapping)
    
    def __len__(self):
        return len(self._mapping)
    
    def __contains__(self, key):
        return key in self._mapping
    
    def __str__(self):
        return str(self._mapping)
    
    def to_dict(self):
        return {k: v.to_dict() for k, v in self._mapping.items()}
    
    @classmethod
    def from_dict(cls, dct):
        return cls({k: MessageRecord.from_dict(v) for k, v in dct.items()})

    @property
    def root(self):
        return self._root
    

@dataclass
class ConversationTraversalTree:
    _conversation: Conversation
    _mapping: ConversationMappingWrapper
    
    root: Node
    current_node: Node

    def __init__(self, conversation: Conversation):
        self._conversation = conversation
        self._mapping = wrap_conversation_mapping(conversation.mapping)
        
        self.root = Node(self._mapping.root)
        self.current_node = self.root

    def add_child(self, parent_node, child_node):
        parent = self.find_node(parent_node)
        child = Node(child_node, parent=parent)
        # check if child already exists in the coversation
        # TODO

    def find_node(self, node_name):
        for node in self.root.descendants:
            if node.name == node_name:
                return node
        return None

    def traverse(self):
        for pre, _, node in RenderTree(self.root):
            yield pre, node.name

    def set_current_node(self, node_name):
        node = self.find_node(node_name)
        if node:
            self.current_node = node

    def get_current_node(self):
        return self.current_node.name


    

def wrap_conversation_mapping(mapping: ConversationMapping) -> ConversationMappingWrapper:
    wrapped_mapping = ConversationMappingWrapper(mapping)
    return wrapped_mapping


def get_current_node_record(conversation: Conversation) -> MessageRecord:
    return conversation.mapping[conversation.current_node]


def get_current_nodes_depth(conversation: Conversation) -> int:
    # Create a list of all the parents of the current node
    parents = []
    traversal_ref = get_current_node_record(conversation)
    while traversal_ref.parent:
        parents.append(traversal_ref.parent)
        traversal_ref = conversation.mapping[traversal_ref.parent]

    # The depth is the number of parents
    return len(parents)