from dataclasses import asdict, dataclass
from typing import Any, Dict, List
from enum import Enum


class RoleType(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'

    def to_json(self):
        return self.value
    
    @classmethod
    def from_json(cls, item):
        return cls(item)
    

class ContentType(Enum):
    TEXT = "text"
    CODE = "code"
    TETHER_BROWSE_DISPLAY = "tether_browse_display"
    TETHER_QUOTE = "tether_quote"
    SYSTEM_ERROR = "system_error"
    STDERR = "stderr"
    EXECUTION_OUTPUT = "execution_output"
    UNKNOWN = "unknown"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, item):
        return cls(item)


@dataclass
class Content:
    content_type: ContentType
    metadata: dict
    
    def __str__(self):
        return "Unknown content"
    
    def to_dict(self):
        dct = asdict(self)
        dct['content_type'] = self.content_type.to_json()
        return dct

    @classmethod
    def from_dict(cls, dct):
        dct['content_type'] = ContentType.from_json(dct['content_type'])
        return cls(**dct)


@dataclass
class NullTypeContent(Content):
    # TODO: Do I need a content type for null content?

    # No content exists for this message
    def __str__(self):
        return "Null Content"
    
    def to_dict(self):
        return super().to_dict()


@dataclass
class TextContent(Content):
    parts: List[str]
    
    def __str__(self):
        return ' '.join(self.parts)
    
    def to_dict(self):
        return super().to_dict()


@dataclass
class CodeContent(Content):
    language: str
    text: str
    
    def __str__(self):
        return f'```{self.language}\n{self.text}\n```'
    
    def to_dict(self):
        return super().to_dict()


@dataclass
class TetherBrowseDisplayContent(Content):
    result: str
    summary: str

    def __str__(self):
        return f'Result:\n```\n{self.result}\n```\n\nSummary:\n```\n{self.summary}\n```'
    
    def to_dict(self):
        return super().to_dict()


@dataclass
class TetherQuoteContent(Content):
    url: str
    domain: str
    text: str
    title: str

    def __str__(self):
        return f'Title: {self.title}\nDomain: {self.domain}\nURL: {self.url}\nText: {self.text}'
    
    def to_dict(self):
        return super().to_dict()


@dataclass
class SystemErrorContent(Content):
    name: str
    text: str

    def __str__(self):
        return f'Name: {self.name}\nText:\n```\n{self.text}\n```'
    
    def to_dict(self):
        return super().to_dict()


@dataclass
class StderrContent(Content):
    text: str

    def __str__(self):
        return f'Standard Error:\n```\n{self.text}\n```'
    
    def to_dict(self):
        return super().to_dict()


@dataclass
class ExecutionOutputContent(Content):
    text: str

    def __str__(self):
        return f'Execution Output:\n```\n{self.text}\n```'
    
    def to_dict(self):
        return super().to_dict()


@dataclass
class Author:
    role: RoleType
    name: str | None
    metadata: dict

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, dct):
        dct['role'] = RoleType.from_json(dct['role'])
        return cls(**dct)


@dataclass
class FinishDetails:
    type: str
    stop: str

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)


@dataclass
class ChatMessageMetadata:
    message_type: str | None
    model_slug: str
    finish_details: FinishDetails
    timestamp_: str

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, dct):
        dct['finish_details'] = FinishDetails.from_dict(dct['finish_details'])
        return cls(**dct)


@dataclass
class ChatMessage:
    id: str
    create_time: float | None
    update_time: float | None
    author: Author
    content: Content
    status: str | None
    end_turn: bool | None
    weight: float | None
    metadata: ChatMessageMetadata
    recipient: str | None

    def to_dict(self):
        dct = asdict(self)
        dct['author'] = self.author.to_dict()
        dct['content'] = self.content.to_dict()
        dct['metadata'] = self.metadata.to_dict()
        return dct
    
    @classmethod
    def from_dict(cls, dct):
        dct['author'] = Author.from_dict(dct['author'])
        dct['content'] = Content.from_dict(dct['content'])
        dct['metadata'] = ChatMessageMetadata.from_dict(dct['metadata'])
        return cls(**dct)


@dataclass
class MessageRecord:
    id: str
    parent: str | None
    children: List[str]
    message: ChatMessage | None

    def to_dict(self):
        dct = asdict(self)
        dct['message'] = self.message.to_dict() if self.message else None
        return dct
    
    @classmethod
    def from_dict(cls, dct):
        dct['message'] = ChatMessage.from_dict(dct['message']) if dct['message'] else None
        return cls(**dct)


ConversationMapping = Dict[str, MessageRecord]
    
@dataclass
class Conversation:
    title: str
    create_time: float
    update_time: float
    mapping: ConversationMapping
    moderation_results: List[Any]
    current_node: str
    plugin_ids: List[str] | None
    conversation_id: str
    conversation_template_id: str | None
    id: str

    def to_dict(self):
        dct = asdict(self)
        dct['mapping'] = {k: v.to_dict() for k, v in dct['mapping'].items()}
        return dct
    
    @classmethod
    def from_dict(cls, dct):
        dct['mapping'] = {k: MessageRecord.from_dict(v) for k, v in dct['mapping'].items()}
        return cls(**dct)
