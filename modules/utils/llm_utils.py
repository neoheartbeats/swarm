from os import error

import ujson as json
from pydantic import BaseModel

type Content = str | list[dict[str, str]]
type ChatMessage = dict[str, Content]
type ChatMessageList = list[ChatMessage]


def chat_message(role: str, content: Content) -> ChatMessage:
    return dict(role=role, content=content)


def message_system(content) -> ChatMessage:
    return chat_message(role="system", content=content)


def message_user(content) -> ChatMessage:
    return chat_message(role="user", content=content)


def message_assistant(content) -> ChatMessage:
    return chat_message(role="assistant", content=content)


def message_ipython(content) -> ChatMessage:
    return chat_message(role="ipython", content=content)


def from_chat_file(file: str) -> ChatMessageList | None:
    """Read chat data from FILENAME."""
    try:
        return json.load(open(file, "r", encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise error(f"Error reading chat file: {e}")


def make_chat_file(file: str, chat_message_list: ChatMessageList) -> None:
    """Make chat data as FILENAME."""
    return json.dump(
        chat_message_list,
        open(file, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )


def to_chat_file(file: str, chat_message_list: list) -> None:
    """Append chat data to existing FILENAME."""
    cml = from_chat_file(file=file)
    if cml:
        make_chat_file(file=file, chat_message_list=(cml + chat_message_list))
    else:
        return make_chat_file(file=file, chat_message_list=chat_message_list)


class SamplingParam(BaseModel):
    model: str
    messages: list
    temperature: float
    top_p: float
    max_tokens: int
    n: int
    tools: list | None = None
    tool_choice: str | None = "auto"
    presence_penalty: float | None = None
    extra_body: dict[str, float] | None = None
