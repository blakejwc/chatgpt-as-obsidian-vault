import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from anytree import RenderTree, NodeMixin, PreOrderIter
from .types import Conversation, ConversationMapping, MessageRecord


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)   


@dataclass
class TraversalNode(NodeMixin):
    """
    A node class for traversal in a conversation tree.
    """
    name: str
    data: MessageRecord

    def __str__(self):
        return f"TraversalNode(name={self.name}, data=MessageRecord(id={self.data.id}, ...), parent={self.parent}, children={self.children}))"
    
    def __repr__(self):
        return f"<TraversalNode data={self.data.to_dict()}>"
    
    @classmethod
    def from_message_record(cls, msg_record: MessageRecord) -> 'TraversalNode':
        """Create a TraversalNode from a MessageRecord."""
        return cls(name=msg_record.id, data=msg_record)
    
    @property
    def message_id(self) -> str:
        """The message id."""
        return self.data.id
    
    @property
    def parent_id(self) -> Optional[str]:
        """The parent message id."""
        return self.data.parent
    
    @property
    def children_ids(self) -> List[str]:
        """The children message ids."""
        return self.data.children.copy()


@dataclass
class ConversationMappingWrapper:
    """
    A wrapper for conversation mapping with utility methods and properties.
    """
    _record_mapping: ConversationMapping
    _root_record: MessageRecord
    node_dict: Dict[str, TraversalNode]

    def __init__(self, mapping: ConversationMapping):
        self._record_mapping = mapping
        self.node_dict = {}
        try:
            self._root_record = next((record for record in mapping.values() if record.parent is None))
            logger.debug(f"Found root record {self._root_record.id}: {self._root_record}")
        except StopIteration:
            raise ValueError("No root record found in conversation mapping")
        
        for new_uuid, new_message_record in mapping.items():
            new_traversal_node = TraversalNode.from_message_record(new_message_record)
            logger.debug(f"Created new traversal node {new_traversal_node}")
            if new_uuid == self._root_record.id:
                self.node_dict[new_uuid] = new_traversal_node
                logger.debug(f"Added root traversal node {new_uuid} to node dict with parent={new_traversal_node.parent}")
            else:
                self.node_dict[new_uuid] = new_traversal_node
                logger.debug(f"Added traversal node {new_uuid} to node dict with parent={new_traversal_node.parent}")

        for uuid, message_record in mapping.items():
            parent_id = message_record.parent
            traversal_node = self.node_dict[uuid]
            if parent_id:
                if parent_id == self._root_record.id:
                    logger.debug(f"Parent of traversal node {uuid} is root node {self._root_record.id}")                
                traversal_node.parent = self.node_dict[parent_id]
                logger.debug(f"Added parent {message_record.parent} to traversal node {uuid}")
            elif uuid != self._root_record.id:
                raise ValueError(f"Message record {uuid} has no parent, but is not root record {self._root_record.id}")

        logger.debug(f"Traversal node dictionary initialized with {len(self.node_dict)} nodes and root node {self.node_dict[self._root_record.id]}")
        
        # Log the entire node dictionary
        logger.debug("Traversal node dictionary:")
        for pre, _, node in RenderTree(self.node_dict[self._root_record.id]):
            treestr = f"{pre}{node.name}"
            logger.debug(treestr)
                
    def find_root_record(self) -> MessageRecord:
        """Find the root message record in the mapping and return a deepcopy."""
        logger.debug("Finding root message record")
        for record in self._record_mapping.values():
            if not record.parent:
                logger.debug(f"Found root message record {record}")
                return copy.deepcopy(record)
        raise ValueError("No root message record found in mapping")

    def __getitem__(self, key):
        return self._record_mapping[key]
    
    def __iter__(self):
        return iter(self._record_mapping)
    
    def __len__(self):
        return len(self._record_mapping)
    
    def __contains__(self, key):
        return key in self._record_mapping
    
    def __str__(self):
        return str(self._record_mapping)

    @property
    def root_record(self) -> MessageRecord:
        """A deepcopy of the root node in the mapping."""
        return copy.deepcopy(self._root_record)
    
    @property
    def record_mapping(self) -> ConversationMapping:
        """A deepcopy of the mapping."""
        return copy.deepcopy(self._record_mapping)


def wrap_conversation_mapping(mapping: ConversationMapping) -> tuple[MessageRecord, ConversationMappingWrapper]:
    """Wrap a conversation mapping in a wrapper class."""
    wrapped_mapping = ConversationMappingWrapper(mapping)
    return wrapped_mapping.root_record, wrapped_mapping


@dataclass
class ConversationTraversalTree:
    """
    A tree structure to traverse through a conversation.
    """
    _wrapped_mapping: ConversationMappingWrapper
    _conversation: Conversation
    root_node: TraversalNode
    current_node: TraversalNode
    node_dict: Dict[str, TraversalNode]

    def __init__(self, conversation: Conversation):
        logger.debug(f"Initializing new traversal tree for conversation {conversation.conversation_id}")
        self._conversation = conversation
        self._wrapped_mapping = ConversationMappingWrapper(conversation.mapping)
        self._set_node_dict_reference(self._wrapped_mapping.node_dict)

    def _set_node_dict_reference(self, node_dict: dict[str, TraversalNode]):
        """Set the node dictionary reference."""
        self.node_dict = node_dict
        mapping_root_record = self._wrapped_mapping.find_root_record()
        self.root_node = self.node_dict[mapping_root_record.id]
        self.current_node = self.root_node
        logger.debug(f"Conversation Traversal Tree initalized with root node: {self.root_node}")

    @property
    def conversation_ref(self) -> Conversation:
        """A reference to the conversation."""
        return self._conversation

    @property
    def mapping_ref(self) -> ConversationMappingWrapper:
        """A reference to the wrapped conversation mapping."""
        return self._wrapped_mapping
            
    def find_node(self, node_name) -> Optional[TraversalNode]:
        """Find a node by name."""
        logger.debug(f"Finding node {node_name}")
        for node in self.root_node.descendants:
            if node.name == node_name:
                logger.debug(f"Found node {node_name}")
                return node
        logger.debug(f"Node {node_name} not found")
        return None

    def traverse(self):
        """Traverse through the tree."""
        for pre, _, node in RenderTree(self.root_node):
            logger.debug(f"Traversing node {node.name}")
            yield pre, node

    def set_current_node(self, node_name):
        """Set the current node."""
        logger.debug(f"Attempting to update current node to {node_name}")
        node = self.find_node(node_name)
        if node:
            self.current_node = node
            logger.debug(f"Current node is now {node_name}")
        else:
            logger.debug(f"Node {node_name} not found. Current node not updated.")


def find_shortest_and_longest_paths(conversation: Conversation) -> tuple[list[str], list[str], list[list[str]]]:
    """Find the shortest and longest paths between root and leaf nodes."""
    traversal_tree = ConversationTraversalTree(conversation)

    paths = []

    # Helper function to traverse from leaf to root and build the path
    def build_path_from_leaf(leaf_node: Optional[TraversalNode]) -> List[str]:
        path: List[str] = []
        while leaf_node is not None:
            path.append(leaf_node.name)
            leaf_node = leaf_node.parent
        
        return path[::-1]

    # Traverse tree and build paths for all leaf nodes
    node: TraversalNode
    for node in PreOrderIter(traversal_tree.root_node):
        if not node.children:  # This is a leaf
            paths.append(build_path_from_leaf(node))
    
    # Determine shortest and longest paths
    shortest_path = min(paths, key=len)
    longest_path = max(paths, key=len)
    
    return shortest_path, longest_path, paths

