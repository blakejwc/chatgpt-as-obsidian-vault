from .logconfig import setup_logger
import copy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Union
from enum import Enum

logger = setup_logger(__name__)


def debug_log_dict(dct: dict, name: str, depth: int = 0):
    """
    Log the provided dictionary in a clean form, with all keys at all depths,
    and in a consistent order.

    :param dct: Dictionary to log.
    :param name: Name of the dictionary (for logging purposes).
    :param depth: Current depth of the recursion (used for indentation in logging).
    """
    indent = "  " * depth
    logger.trace(f"{indent}{name}:")

    for key, value in sorted(dct.items()):
        if isinstance(value, dict):
            debug_log_dict(value, key, depth + 1)
        else:
            logger.trace(f"{indent}  {key}: {value}")


class RoleType(Enum):
    """
    OpenAI's API has three roles: system, user, and assistant.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    UNKNOWN = "unknown"

    def to_json(self) -> str:
        return self.value

    @classmethod
    def from_json(cls, item) -> "RoleType":
        return cls(item)


class ContentType(Enum):
    """
    The content_type field in the OpenAI API can have one of the following values:
    - text
    - code
    - tether_browse_display
    - tether_quote
    - system_error
    - stderr
    - execution_output

    If the content_type is not one of the above, the content_type field will be set to one of the following:
    - null
    - unknown
    """
    TEXT = "text"
    CODE = "code"
    TETHER_BROWSE_DISPLAY = "tether_browse_display"
    TETHER_QUOTE = "tether_quote"
    SYSTEM_ERROR = "system_error"
    STDERR = "stderr"
    EXECUTION_OUTPUT = "execution_output"
    NULL = "null"
    UNKNOWN = "unknown"

    def to_json(self) -> str:
        """Returns the string representation of the enum value."""
        return self.value

    @classmethod
    def from_json(cls, item) -> "ContentType":
        """Returns the enum value corresponding to the provided string."""
        try:
            return cls(item)
        except ValueError:
            logger.error(f"Unexpected content type: {item}. Defaulting to UNKNOWN.")
            return cls.UNKNOWN


@dataclass
class Content:
    """
    The base class for all content types. All content types have a content_type field and a metadata field.
    """
    content_type: ContentType
    metadata: dict

    def __str__(self):
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        dct = asdict(self)
        dct["content_type"] = self.content_type.to_json()
        return dct

    @classmethod
    def from_dict(cls, dct) -> "Content":
        """
        First, checks whether dct is None. If it is None, the code sets content_type to ContentType.NULL
        and metadata to an empty dict.

        Then, `if content_type is not None:` checks whether `content_type` is `None`.
        If it is not `None`, `dct['content_type'] = ContentType.from_json(content_type)`
        attempts to convert the string value to its enum equivalent.
        If `content_type` is `None`, this means that 'content_type' was not a key in the
        original dct and the code does not attempt to convert it to an enum.

        The rationale here is to distinguish between two scenarios:

        `'content_type'` key is not present in `dct` (=> `content_type` is `None`)
        `'content_type'` key is present but has an invalid value (=> `content_type` is not `None`,
        and we try to convert it, potentially raising an error or logging a warning in `ContentType.from_json`)
        """
        if dct is None:
            logger.debug(f"Content.from_dict got None. Setting to NULL.")
            return NullTypeContent(content_type=ContentType.NULL, metadata={})

        content_type = dct.get("content_type")
        if content_type is not None:
            logger.debug(
                f"Content.from_dict got: {dct}, which is of type: {type(dct).__name__}"
            )
            dct["content_type"] = ContentType.from_json(content_type)

        # verify_dict checks that dct has the correct keys and fills in missing keys with default values
        is_valid = cls.verify_dict(dct)
        if is_valid:
            if "metadata" not in dct:
                dct["metadata"] = {}
                logger.debug(
                    f"Content.from_dict got a dict with no metadata. Setting to empty dict."
                )
            return cls(**dct)
        elif (
            content_type is not None
        ):  # content_type was in the original dict or verify_dict added it
            if cls.has_only_metadata_key(
                dct
            ):  # dict has only the metadata key, in addition to content_type
                logger.debug(
                    f"Content.from_dict got a dict with only a metadata key. Setting to NULL."
                )
                new_metadata = copy.deepcopy(dct["metadata"])
                if new_metadata is not None:
                    new_metadata["__debug"] = {}
                    new_metadata["__debug"]["content_type"] = dct["content_type"]
                    new_metadata["__debug"]["original_dict"] = dct
                return NullTypeContent(
                    content_type=ContentType.NULL, metadata=new_metadata
                )
            else:  # dict has keys other than metadata and content_type
                logger.debug(
                    f"Content.from_dict got an invalid dict. Setting to UNKNOWN."
                )
                if "metadata" in dct:
                    new_metadata = copy.deepcopy(dct["metadata"])
                else:
                    new_metadata = {}
                if new_metadata is not None:
                    new_metadata["__debug"] = {}
                    new_metadata["__debug"]["content_type"] = dct["content_type"]
                    new_metadata["__debug"]["original_dict"] = dct
                return UnknownContent(
                    content_type=ContentType.UNKNOWN, metadata=new_metadata
                )
        else:  # content_type was not in the original dict and verify_dict did not add it
            logger.debug(f"Content.from_dict got an invalid dict. Setting to UNKNOWN.")
            if "metadata" in dct:
                new_metadata = copy.deepcopy(dct["metadata"])
                if new_metadata is not None:
                    new_metadata["__debug"] = {}
                    new_metadata["__debug"]["original_dict"] = dct
                return UnknownContent(
                    content_type=ContentType.UNKNOWN, metadata=new_metadata
                )
            else:
                new_metadata = {}
                new_metadata["__debug"] = {}
                new_metadata["__debug"]["original_dict"] = dct
                return UnknownContent(
                    content_type=ContentType.UNKNOWN, metadata=new_metadata
                )

    @classmethod
    def verify_dict(cls, dct) -> bool:
        """
        Checks that the provided dictionary has the correct keys for the dataclass.
        If the dictionary has extra keys, they are logged as warnings.
        If the dictionary is missing keys, they are added with default values.
        """
        expected_keys = set(cls.__annotations__.keys())
        actual_keys = set(dct.keys())
        content_base_class_keys = set(Content.__annotations__.keys())
        actual_keys = actual_keys - content_base_class_keys
        is_valid = actual_keys.issubset(expected_keys)
        extra_keys = actual_keys - expected_keys
        extra_keys_dict_for_printing = {k: dct[k] for k in extra_keys}
        if not is_valid:
            logger.warning(
                f"Unexpected keys in {cls.__name__}: {extra_keys_dict_for_printing}"
            )
            logger.debug(f"Expected keys: {expected_keys}")
            logger.debug(f"Actual keys: {actual_keys}")

        # Check that all the fields in the dataclass, and all fields in the parent dataclasses, are present
        # in the dictionary. If not, set them to their default values.
        for field_name, field_type in {
            **cls.__annotations__,
            "content_type": str,
        }.items():
            if field_name not in dct:
                logger.debug(
                    f"{cls.__name__}.verify_dict: No {field_name} found in {dct}. Setting to default value."
                )
                dct[field_name] = cls.get_default_value(field_type)
                is_valid = False

        return is_valid

    @staticmethod
    def get_default_value(field_type: Any) -> Any:
        """Returns the default value for the provided field type."""
        if field_type == str:
            return ""
        elif field_type == int:
            return 0
        elif field_type == float:
            return 0.0
        elif field_type == bool:
            return False
        elif field_type == dict:
            return {}
        elif field_type == list:
            return []
        elif issubclass(field_type, Enum):
            default_enum_value = next(
                (e for e in field_type if e.name == "UNKNOWN"), None
            )
            if default_enum_value is None:
                logger.error(
                    f"No 'UNKNOWN' member in enum {field_type}. Defaulting to first enum value."
                )
                default_enum_value = next(iter(field_type))
            return default_enum_value
        else:
            return None

    @staticmethod
    def has_only_metadata_key(dct) -> bool:
        """
        Checks to see if the provided dictionary has only the 'content_type' key and the 'metadata' key,
        or only the metadata key.
        """
        if len(dct.keys()) == 1:
            return "metadata" in dct
        elif len(dct.keys()) == 2:
            return "metadata" in dct and "content_type" in dct
        else:
            return False


@dataclass
class UnknownContent(Content):
    """
    The UnknownContent class is used when the content_type field in the OpenAI API is not one of the following:
    - text
    - code
    - tether_browse_display
    - tether_quote
    - system_error
    - stderr
    - execution_output

    # TODO: Add the missing documentation for when and why UNKNOWN is used vs NULL.
    """
    # Can have any number of keys
    def __str__(self):
        # Return a string representation of all the dataclass fields
        return " ".join([f"{k}: {v}" for k, v in self.__dict__.items()])

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return super().to_dict()

    @classmethod
    def verify_dict(cls, dct) -> bool:
        """Returns True since UnknownContent can have any number of keys."""
        # Don't check keys for UnknownContent since by definition we don't know what they should be.
        return True


@dataclass
class NullTypeContent(Content):
    """
    The NullTypeContent class is used when the content_type field is missing,
    or when the content_type exists but does not contain any content.
    """
    # TODO: Do I need a content type for null content?

    # No content exists for this message
    def __str__(self):
        return "Null Content"

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return super().to_dict()

    @classmethod
    def verify_dict(cls, dct) -> bool:
        """Returns True if the dictionary has the correct keys, False otherwise."""
        has_correct_keys = True
        if len(dct.keys()) > len(Content.__dataclass_fields__):
            logger.debug(
                f"NullTypeContent.verify_dict: Invalid keys found in {dct}. Setting to UNKNOWN."
            )
            dct["content_type"] = ContentType.UNKNOWN
            has_correct_keys = False
        return super().verify_dict(dct) and has_correct_keys


@dataclass
class TextContent(Content):
    """
    The TextContent class is used when the content_type field in the OpenAI API is 'text'.
    """
    parts: List[str]

    def __str__(self):
        return " ".join(self.parts)

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return super().to_dict()

    @classmethod
    def verify_dict(cls, dct) -> bool:
        """Returns True if the dictionary has the correct keys, False otherwise."""
        return super().verify_dict(dct)


@dataclass
class CodeContent(Content):
    """
    The CodeContent class is used when the content_type field in the OpenAI API is 'code'.
    """
    language: str
    text: str

    def __str__(self):
        return f"```{self.language}\n{self.text}\n```"

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return super().to_dict()

    @classmethod
    def verify_dict(cls, dct) -> bool:
        """Returns True if the dictionary has the correct keys, False otherwise."""
        return super().verify_dict(dct)


@dataclass
class TetherBrowseDisplayContent(Content):
    """
    The TetherBrowseDisplayContent class is used when the content_type field in the OpenAI API is 'tether_browse_display'.
    """
    result: str
    summary: str

    def __str__(self):
        return f"Result:\n```\n{self.result}\n```\n\nSummary:\n```\n{self.summary}\n```"

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return super().to_dict()

    @classmethod
    def verify_dict(cls, dct) -> bool:
        """Returns True if the dictionary has the correct keys, False otherwise."""
        return super().verify_dict(dct)


@dataclass
class TetherQuoteContent(Content):
    """
    The TetherQuoteContent class is used when the content_type field in the OpenAI API is 'tether_quote'.
    """
    url: str
    domain: str
    text: str
    title: str

    def __str__(self):
        return f"Title: {self.title}\nDomain: {self.domain}\nURL: {self.url}\nText: {self.text}"

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return super().to_dict()

    @classmethod
    def verify_dict(cls, dct) -> bool:
        """Returns True if the dictionary has the correct keys, False otherwise."""
        return super().verify_dict(dct)


@dataclass
class SystemErrorContent(Content):
    """
    The SystemErrorContent class is used when the content_type field in the OpenAI API is 'system_error'.
    """
    name: str
    text: str

    def __str__(self):
        return f"Name: {self.name}\nText:\n```\n{self.text}\n```"

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return super().to_dict()

    @classmethod
    def verify_dict(cls, dct) -> bool:
        """Returns True if the dictionary has the correct keys, False otherwise."""
        return super().verify_dict(dct)


@dataclass
class StderrContent(Content):
    """
    The StderrContent class is used when the content_type field in the OpenAI API is 'stderr'.
    """
    text: str

    def __str__(self):
        return f"Standard Error:\n```\n{self.text}\n```"

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return super().to_dict()

    @classmethod
    def verify_dict(cls, dct) -> bool:
        """Returns True if the dictionary has the correct keys, False otherwise."""
        return super().verify_dict(dct)


@dataclass
class ExecutionOutputContent(Content):
    """
    The ExecutionOutputContent class is used when the content_type field in the OpenAI API is 'execution_output'.
    """
    text: str

    def __str__(self):
        return f"Execution Output:\n```\n{self.text}\n```"

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return super().to_dict()

    @classmethod
    def verify_dict(cls, dct) -> bool:
        """Returns True if the dictionary has the correct keys, False otherwise."""
        return super().verify_dict(dct)


@dataclass
class Author:
    """
    The Author class is used to represent the author of a message.
    """
    role: RoleType
    name: Union[str, None]
    metadata: dict

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return asdict(self)

    @classmethod
    def from_dict(cls, dct) -> "Author":
        """Returns an Author object from the provided dictionary."""
        logger.debug(
            f"Author.from_dict got: {dct}, which is of type: {type(dct).__name__}"
        )
        dct["role"] = RoleType.from_json(dct["role"])
        return cls(**dct)


@dataclass
class FinishDetails:
    """
    The FinishDetails class is used to represent the finish_details field in the OpenAI API.
    """
    type: Union[str, None]
    stop_tokens: Union[str, None]

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return asdict(self)

    @classmethod
    def from_dict(cls, dct) -> "FinishDetails":
        """Returns a FinishDetails object from the provided dictionary."""
        logger.debug(
            f"FinishDetails.from_dict got: {dct}, which is of type: {type(dct).__name__}"
        )
        if "type" not in dct:
            logger.debug(
                f"FinishDetails.from_dict got no `type`. Setting to None."
            )
            dct["type"] = None
        if "stop_tokens" not in dct:
            logger.debug(
                f"FinishDetails.from_dict got no `stop_tokens`. Setting to None."
            )
            dct["stop_tokens"] = None
        return cls(**dct)

@dataclass
class Attachments:
    """
    The Attachments class is used to represent the attachments field in the ChatMessageMetadata dataclass.
    """
    name: str
    id: str
    size: int

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return asdict(self)

    @classmethod
    def from_dict(cls, dct) -> "Attachments":
        """Returns an Attachments object from the provided dictionary."""
        return cls(**dct)

@dataclass
class ChatMessageMetadata:
    """
    The ChatMessageMetadata class is used to represent the metadata field in the OpenAI API.
    """
    # TODO: check the jsonschema to verfiy that these are the correct keys
    is_complete: Union[bool, None]
    message_type: Union[str, None]
    model_slug: Union[str, None]
    finish_details: Union[FinishDetails, None]
    timestamp_: Union[str, None]
    parent_id: Union[str, None]
    invoked_plugin: Union[str, None]
    attachments: Union[List[Attachments], None] # TODO: Seems like it doesn't always exist, but the question is whether it will ever be an empty list. Will there be a case where it's None?
    # TODO: implement a dataclass for the aggregate_results
    aggregate_result: Union[Dict[str, Any], None]
    # Example of what the json looks like:
    # {'aggregate_result': {'status': 'success', 'run_id': '8742332d-a649-4225-9cc5-a25019945e7d', 'start_time': 1691869619.314078, 'update_time': 1691869619.6085324, 'code': '# Importing necessary modules\nimport json\nfrom jsonschema import validate, exceptions\n\n# Loading JSON Schema\nwith open("/mnt/data/conversations_jsonschema.json", "r") as schema_file:\n    schema = json.load(schema_file)\n\n# Loading JSON Data\nwith open("/mnt/data/conversations_of_one.json", "r") as data_file:\n    data = json.load(data_file)\n\n# Validating the JSON data against the schema\nvalidation_errors = []\ntry:\n    validate(instance=data, schema=schema)\nexcept exceptions.ValidationError as e:\n    validation_errors.append(str(e))\n\nvalidation_errors', 'end_time': 1691869619.6085324, 'final_expression_output': '[]', 'in_kernel_exception': None, 'system_exception': None, 'messages': [], 'jupyter_messages': [{'msg_type': 'status', 'parent_header': {'msg_id': 'c9b51a21-238da2197eee0b560753c724_2_1', 'version': '5.3'}, 'content': {'execution_state': 'busy'}}, {'msg_type': 'execute_input', 'parent_header': {'msg_id': 'c9b51a21-238da2197eee0b560753c724_2_1', 'version': '5.3'}}, {'parent_header': {'msg_id': 'c9b51a21-238da2197eee0b560753c724_2_1', 'version': '5.3'}, 'msg_type': 'execute_result', 'content': {'data': {'text/plain': '[]'}}}, {'msg_type': 'status', 'parent_header': {'msg_id': 'c9b51a21-238da2197eee0b560753c724_2_1', 'version': '5.3'}, 'content': {'execution_state': 'idle'}}], 'timeout_triggered': None}, 'is_complete': True, 'message_type': None, 'model_slug': 'gpt-4-code-interpreter', 'parent_id': '05f69fc8-77b7-4c28-b185-b226bc7b22d9', 'timestamp_': 'absolute'}
    is_user_system_message: Union[bool, None]
    user_context_message_data: Union[Dict[str, Any], None]
    # Example of what the text looks like:
    # [tapestrymd.types]TRACE:             about_model_message: ## Cues
    #    
    # "generate and approach for yourself”:
    # ”## Methodology
    # 1. Task Breakdown:
    #  Start with the overarching objective and distill it into granular, actionable sub-tasks for clarity and manageability.
    # 2. Domain Alignment:
    # Associate each sub-task with its corresponding domain, highlighting the fundamental principles or techniques it employs.
    # 3. Operational Sequence:
    # Strategize the order of execution, keeping in mind interdependencies and workflow optimization for the sub-tasks.
    # 4. Adaptive Refinement:
    # During each phase, actively monitor LLM outputs. Offer feedback and adjust directions to enhance accuracy and precision.
    # 5. Holistic Assembly:
    # Synthesize the results of individual sub-tasks to formulate the comprehensive and intended outcome.”
    #   
    # "compress excerpt" interpretation:
    # "Make the excerpt of text I've included, and delineated with triple backticks, as few characters as possible by using strategies and techniques like sentence reorganizing with contextually equivalent replacements, abbreviations, aggressive devoweling, typographical tricks, UTF-8 alternative characters and symbols, substitution with specialized notation from analogous mathematical and scientific concepts, cue and keyword supplementation for style and language simplification — forgo any human readability, the only requirement is that it maintains the contextual nuances and unique information in a form that makes the text usable to ChatGPT."
    # [tapestrymd.types]TRACE:             about_user_message: Your focused domains:
    # [KRR]:[1(KR&R)-1a(Ont:FrmlRep,concepts,rels)-1b(SmNtwks:GrphStrcts,knwldg)-1c(DscLog:Frm,con,rol,indv)-1d(RbSys:Rls,inf,dc)-1e(Frms:DtaStrcts,strtSitu)-1f(NmLog:Rsn,incmpInf)-1g(BelRev:Mthd,bel,upd,newInf)];
    # [PMA]:[4(Pt,M&A)-4a(PtRecog:ID,rgl,dta)-4b(ConMetr:Und,ids,trms)-4c(AnlRsn:Prb,sml,sol)-4d(StrMap:Cor,doms)-4e(Gest:Pt,whl,sm,prt)];
    # [EMRG]:[5(Erg)-5a(SO:SponOrd,lclInt)-5b(Syn:CoopInt,otcm,indComp)-5c(PhTr:SudSft,bev,chng,cond)-5d(CelAuto:MathMdl,cls,evl,lclrl)];
    # [DSIE]:[6(DSIE,SD,Rx)-6a(DtCl:Rmv,corct,inacc)-6b(ETL:Prcs,dta,anl)-6c(DBNorm:Org,dta,rdncy,intg)-6d(RegEx:Ptrn,charComb,strs)-6e(SD:Struct,dta,acc,usbl)];
    # [CODE_TH]:[7(CTh)-7a(LinC:Code,linAlg)-7b(HD:DifPos,strs,lngth)-7c(ParChk:ErrDet,mech)-7d(MLD:DecodeMthd,errPrb)-7e(TbC&LDPC:ErrCorct,prfmnc)];
    # [SFSI]:[8(Sum&FS)-8a(ExSum:Pick,sntnc,origCnt,sum)-8b(AbSum:Gen,sum,sameInf)-8c(FzLog:LogSys,val,tr,fls)-8d(FzSt:Set,elem,deg,mem)-8e(SimMtrcs:Msr,qunt,sim,dtPts)];
    # [ERDS]:[9(ErrCorct&ERDS)-9a(BlkC:ErrCorct,dta,blks)-9b(ConvC:Trnsf,mbit,nbit,sym)-9c(RSCode:Blk,ErrCorct,dgtlComm)-9d(Cksum:Mthd,errDet,stg,trnsm)-9e(ErrResDS:Strcts,errs,corr)];

    # [Preferences]:[1(Markdown Formatting)&&2(High Divergent Thinking)]

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        return asdict(self)

    @classmethod
    def from_dict(cls, dct) -> "ChatMessageMetadata":
        """Returns a ChatMessageMetadata object from the provided dictionary."""
        logger.debug(
            f"ChatMessageMetadata.from_dict got: {dct}, which is of type: {type(dct).__name__}"
        )
        # TODO: check the jsonschema to see if these are optional
        if "finish_details" in dct:
            dct["finish_details"] = FinishDetails.from_dict(dct["finish_details"])
        else:
            logger.debug(
                f"ChatMessageMetadata.from_dict got no `finish_details`. Building FinishDetails from empty dictionary."
            )
            dct["finish_details"] = FinishDetails.from_dict({})

        if "attachments" in dct and dct["attachments"] is not None:
            dct["attachments"] = [Attachments.from_dict(attachment) for attachment in dct["attachments"]]
        elif "attachments" in dct and dct["attachments"] is None:
            logger.debug(
                f"ChatMessageMetadata.from_dict got `attachments` as None. Setting to None."
            )
            dct["attachments"] = None
        else:
            logger.debug(
                f"ChatMessageMetadata.from_dict got no `attachments`. Setting to None."
            )
            dct["attachments"] = None

        if "timestamp_" not in dct:
            logger.debug(
                f"ChatMessageMetadata.from_dict got no `timestamp_`. Setting to None."
            )
            dct["timestamp_"] = None

        if "message_type" not in dct:
            logger.debug(
                f"ChatMessageMetadata.from_dict got no `message_type`. Setting to None."
            )
            dct["message_type"] = None

        if "model_slug" not in dct:
            logger.debug(
                f"ChatMessageMetadata.from_dict got no `model_slug`. Setting to None."
            )
            dct["model_slug"] = None

        if "is_complete" not in dct:
            logger.debug(
                f"ChatMessageMetadata.from_dict got no `is_complete`. Setting to None."
            )
            dct["is_complete"] = None

        if "parent_id" not in dct:
            logger.debug(
                f"ChatMessageMetadata.from_dict got no `parent_id`. Setting to None."
            )
            dct["parent_id"] = None

        if "invoked_plugin" not in dct:
            logger.debug(
                f"ChatMessageMetadata.from_dict got no `invoked_plugin`. Setting to None."
            )
            dct["invoked_plugin"] = None

        if "aggregate_result" not in dct:
            logger.debug(
                f"ChatMessageMetadata.from_dict got no `aggregate_result`. Setting to None."
            )
            dct["aggregate_result"] = None

        if "is_user_system_message" not in dct:
            logger.debug(
                f"ChatMessageMetadata.from_dict got no `is_user_system_message`. Setting to None."
            )
            dct["is_user_system_message"] = None

        return cls(**dct)


@dataclass
class ChatMessage:
    """
    The ChatMessage class is used to represent a message in the OpenAI API.
    """
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

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        dct = asdict(self)
        dct["author"] = self.author.to_dict()
        dct["content"] = self.content.to_dict()
        dct["metadata"] = self.metadata.to_dict()
        return dct

    @classmethod
    def from_dict(cls, dct) -> "ChatMessage":
        """Returns a ChatMessage object from the provided dictionary."""
        dct["author"] = Author.from_dict(dct["author"])

        # TODO: check the jsonschema to see if this is optional and if it can be empty
        if len(dct["metadata"]) == 0:
            logger.debug(
                f"ChatMessage.from_dict got empty metadata. Building ChatMessageMetadata from empty dict."
            )
            dct["metadata"] = ChatMessageMetadata.from_dict({})
        elif dct["metadata"] is not None:
            dct["metadata"] = ChatMessageMetadata.from_dict(dct["metadata"])
        else:
            logger.debug(
                f"ChatMessage.from_dict got no metadata. Building ChatMessageMetadata from empty dict"
            )
            dct["metadata"] = ChatMessageMetadata.from_dict({})

        # Determine the Content subclass using content_type
        content_type = ContentType(dct["content"]["content_type"])
        if content_type == ContentType.TEXT:
            dct["content"] = TextContent.from_dict(dct["content"])
        elif content_type == ContentType.CODE:
            dct["content"] = CodeContent.from_dict(dct["content"])
        elif content_type == ContentType.TETHER_BROWSE_DISPLAY:
            dct["content"] = TetherBrowseDisplayContent.from_dict(dct["content"])
        elif content_type == ContentType.TETHER_QUOTE:
            dct["content"] = TetherQuoteContent.from_dict(dct["content"])
        elif content_type == ContentType.SYSTEM_ERROR:
            dct["content"] = SystemErrorContent.from_dict(dct["content"])
        elif content_type == ContentType.STDERR:
            dct["content"] = StderrContent.from_dict(dct["content"])
        elif content_type == ContentType.EXECUTION_OUTPUT:
            dct["content"] = ExecutionOutputContent.from_dict(dct["content"])
        else:
            dct["content"] = NullTypeContent.from_dict(dct["content"])

        debug_log_dict(
            dct, f"ChatMessage.from_dict got dictionary of type: {type(dct).__name__}"
        )
        return cls(**dct)


@dataclass
class MessageRecord:
    """
    The MessageRecord class is used to represent a message in a conversation.
    """
    id: str
    parent: str | None
    children: List[str]
    message: ChatMessage | None

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        dct = asdict(self)
        dct["message"] = self.message.to_dict() if self.message else None
        return dct

    @classmethod
    def from_dict(cls, dct) -> "MessageRecord":
        """Returns a MessageRecord object from the provided dictionary."""
        debug_log_dict(
            dct, f"MessageRecord.from_dict got dictionary of type: {type(dct).__name__}"
        )
        dct["message"] = (
            ChatMessage.from_dict(dct["message"]) if dct["message"] else None
        )
        return cls(**dct)


# TODO: Add documentation for ConversationMapping
ConversationMapping = Dict[str, MessageRecord]


@dataclass
class Conversation:
    """
    The Conversation class is used to represent a conversation in the OpenAI API.
    """
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

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the dataclass."""
        dct = asdict(self)
        dct["mapping"] = {
            k: v.to_dict() if not isinstance(v, dict) else v
            for k, v in dct["mapping"].items()
        }
        return dct

    @classmethod
    def from_dict(cls, dct) -> "Conversation":
        """Returns a Conversation object from the provided dictionary."""
        debug_log_dict(
            dct, f"Conversation.from_dict got dictionary of type: {type(dct).__name__}"
        )
        dct["mapping"] = {
            k: MessageRecord.from_dict(v) for k, v in dct["mapping"].items()
        }
        return cls(**dct)
