import logging
from dataclasses import dataclass
from typing import Optional, Union
from anytree import RenderTree, NodeMixin
from .types import Conversation, ConversationMapping, MessageRecord


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class ConversationMappingWrapper:
    """
    A wrapper for conversation mapping with utility methods and properties.
    """
    _mapping: ConversationMapping
    _root: MessageRecord

    def __init__(self, mapping: ConversationMapping):
        self._mapping = mapping
        try:
            self._root = next((node for node in mapping.values() if not node.parent))
            logger.debug(f"Found root node on init {self._root}")
        except StopIteration:
            raise ValueError("No root node found in mapping")
        
    def find_root(self) -> MessageRecord:
        logger.debug("Finding root node")
        for node in self._mapping.values():
            if not node.parent:
                logger.debug(f"Found root node {node}")
                return node
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

    @property
    def root(self) -> MessageRecord:
        return self._root
    
    @property
    def mapping(self) -> ConversationMapping:
        return self._mapping


def wrap_conversation_mapping(mapping: ConversationMapping) -> tuple[MessageRecord, ConversationMappingWrapper]:
    wrapped_mapping = ConversationMappingWrapper(mapping)
    return wrapped_mapping.root, wrapped_mapping   


@dataclass
class TraversalNode(NodeMixin):
    """
    A node class for traversal in a conversation tree.
    """
    name: str
    data: MessageRecord
    parent: Optional['TraversalNode'] = None

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"<TraversalNode data={self.data.to_dict()}>"


@dataclass
class ConversationTraversalTree:
    """
    A tree structure to traverse through a conversation.
    """
    root_node: TraversalNode
    current_node: TraversalNode

    def __init__(self, conversation: Conversation):
        logger.debug(f"Initializing new traversal tree for conversation {conversation.conversation_id}")
        self._conversation = conversation
        root_record, self._wrapped_mapping = wrap_conversation_mapping(conversation.mapping)
        logger.debug(f"Root message record: {root_record}")
        
        self.root_node = self.make_traversal_node(root_record, None)
        self.current_node = self.root_node
        logger.debug(f"Traversal tree initalized with root node: {self.root_node}")
        logger.debug(f"Traversal tree initalized with current node: {self.current_node}")

    @classmethod
    def make_traversal_node(cls, message_record: MessageRecord, parent: Optional[TraversalNode] = None) -> TraversalNode:
        return TraversalNode(message_record.id, parent=parent, data=message_record)

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def mapping(self) -> ConversationMappingWrapper:
        return self._wrapped_mapping

    def add_child(self, parent: Union[str, TraversalNode], child: Union[MessageRecord, TraversalNode]):
        log_child_id = child.id if isinstance(child, MessageRecord) else child.name
        log_parent_id = parent.id if isinstance(parent, MessageRecord) else parent
        logger.debug(f"Adding child {log_child_id} of type {type(child)} to parent {log_parent_id} of type {type(parent)}")
        if isinstance(child, TraversalNode) and child.name == self.root_node.name:
            logger.debug(f"Child node {child.name} is root node. Parent should be None.")
            if parent is not None:
                logger.error(f"Parent node {parent} is not None.")
                raise ValueError(f"Parent node {parent} is not None.")
            else:
                logger.debug(f"Parent node {parent} is None. Child node {child.name} is root node.")
                return
            
        if isinstance(parent, str):
            parent_node = self.find_node(parent)
        else:
            logger.debug(f"Parent is not a string. Parent is {parent}")
            parent_node = parent if isinstance(parent, TraversalNode) else None
        

        if parent_node is None and isinstance(parent, str):
            if parent in self.mapping:
                logger.info(f"Parent node {parent} not found in tree, but found in mapping.")
                mapping_parent = self.mapping[parent].parent
                if mapping_parent is not None:
                    logger.info(f"Parent node {parent} has a parent in mapping.")
                    self.add_child(mapping_parent, self.mapping[parent])
                    parent_node = self.find_node(parent)
                else:
                    logger.info(f"Parent node {parent} has no parent in mapping. Checking if parent is root node.")
                    if parent == self.root_node.name:
                        logger.info(f"Parent node {parent} is root node. Using root node as parent.")
                        parent_node = self.root_node
                    else:
                        raise ValueError(f"Parent node {parent} not found")
            else:
                raise ValueError(f"Parent node {parent} not found")
            
        if parent_node is None:
            raise ValueError(f"Parent node {parent} not found")

        child_node = self.make_traversal_node(child, parent_node) if isinstance(child, MessageRecord) else child if isinstance(child, TraversalNode) else None
        if child_node is None:
            raise ValueError(f"Child node {child} not found")
        
        logger.debug(f"Child node: {child_node}")
        logger.debug(f"Parent node: {parent_node}")
        child_node.parent = parent_node
        logger.debug(f"Child has parent: {child_node.parent.name}")
        if child_node in parent_node.children:
            logger.debug(f"Parent has child: {child_node.name}")
        else:
            logger.error(f"Parent does not have child: {child_node.name}")
            # check if child doesn't exists in the coversation
            if child_node.name not in self._conversation.mapping:
                logger.exception(f"Child node {child_node.name} not found in conversation mapping.")
            else:
                logger.warning(f"Child node {child_node.name} found in conversation mapping. Hacking it in.")
                parent_node.children = parent_node.children + (child_node,)
            

    def find_node(self, node_name):
        logger.debug(f"Finding node {node_name}")
        for node in self.root_node.descendants:
            if node.name == node_name:
                logger.debug(f"Found node {node_name}")
                return node
        logger.debug(f"Node {node_name} not found")
        return None

    def traverse(self):
        for pre, _, node in RenderTree(self.root_node):
            logger.debug(f"Traversing node {node.name}")
            yield pre, node

    def set_current_node(self, node_name):
        logger.debug(f"Attempting to update current node to {node_name}")
        node = self.find_node(node_name)
        if node:
            logger.debug(f"Found node {node_name}")
            self.current_node = node
        else:
            logger.debug(f"Node {node_name} not found. Current node not updated.")

    def get_current_node(self) -> str:
        return self.current_node.name

    def get_current_node_data(self) -> MessageRecord:
        return self.current_node.data
    

def build_conversation_traversal_tree(conversation: Conversation) -> ConversationTraversalTree:
    logger.debug(f"Building traversal tree for conversation {conversation.conversation_id}")
    traversal_tree = ConversationTraversalTree(conversation)
    logger.debug(f"Traversal tree root node: {traversal_tree.root_node}")
    logger.debug(f"Traversal tree current node: {traversal_tree.current_node}")
    for id, message_record in conversation.mapping.items():
        if id != message_record.id:
            logger.warn(f"Message record id {id} does not match message record {message_record}")
        logger.debug(f"Adding message record to traversal tree: {message_record}")
        if message_record.parent:
            logger.debug(f"Message record {message_record.id} has parent {message_record.parent}")
            traversal_tree.add_child(message_record.parent, message_record)
        
    return traversal_tree



# Find shorted and longest paths between root and leaf nodes
def find_shortest_and_longest_paths(conversation: Conversation) -> tuple[list[str], list[str]]:
    traversal_tree = build_conversation_traversal_tree(conversation)
    logger.debug(f"Traversing conversation {conversation.conversation_id}")
    paths = []
    # traverse through the tree and find all paths
    for pre, node in traversal_tree.traverse():
        if not node.children:
            logger.debug(f"Found leaf node {node.name}")
            path = [node.name]
            while node.parent:
                path.append(node.parent.name)
                node = node.parent
            path.reverse()
            paths.append((pre, path))
    
    logger.debug(f"Found {len(paths)} paths")
    shortest_path = min(paths, key=lambda path: len(path[1]))
    logger.debug(f"Shortest path: {shortest_path}")
    longest_path = max(paths, key=lambda path: len(path[1]))
    logger.debug(f"Longest path: {longest_path}")
    return shortest_path[1], longest_path[1]
