# OMC 日志里的 "all generation" 是模型原文，没有 OpenAI tool_calls。
# 这里不自己写 Hermes / Qwen XML 解析，而是按名字加载本机 vLLM 注册的 parser，
# 顺序与 vLLM serve 一致：先 reasoning parser，再把剩下的 content 交给 tool-call parser。
#
# 名字和启动参数相同，例如：
#   --reasoning-parser qwen3 --tool-call-parser hermes      # Qwen3 Instruct
#   --reasoning-parser qwen3 --tool-call-parser qwen3_xml   # Qwen3.5 / Qwen3-Coder
# 换模型只改这两个名字，评分代码不变。评测机需要安装 vLLM，且 parser 名在该版本里存在。

import json
from types import SimpleNamespace


class ParsedGeneration:
    def __init__(self, reasoning_content, content, tool_calls):
        self.reasoning_content = reasoning_content or ""
        self.content = content or ""
        # OpenAI 形状：[{type, function: {name, arguments}}]。空列表表示 parser 判定没调用。
        self.tool_calls = tool_calls or []


def _load_parser_managers():
    """兼容 vLLM 把 tool parser 从 entrypoints 挪到 vllm.tool_parsers 的两个版本。"""
    errors = []
    candidates = (
        ("vllm.tool_parsers", "vllm.reasoning"),
        ("vllm.entrypoints.openai.tool_parsers", "vllm.reasoning"),
    )
    for tool_mod, reason_mod in candidates:
        try:
            tool_manager = __import__(tool_mod, fromlist=["ToolParserManager"]).ToolParserManager
            reason_manager = __import__(
                reason_mod, fromlist=["ReasoningParserManager"]
            ).ReasoningParserManager
            return tool_manager, reason_manager
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{tool_mod}: {exc}")
    raise ImportError(
        "OMC BFCL 需要本机安装 vLLM，才能按名字调用 reasoning / tool-call parser。\n"
        + "\n".join(errors)
    )


def _registered_names(manager, attr):
    list_fn = getattr(manager, "list_registered", None)
    if callable(list_fn):
        try:
            return list(list_fn())
        except Exception:  # noqa: BLE001
            pass
    table = getattr(manager, attr, None)
    if isinstance(table, dict):
        return sorted(table)
    return []


def _construct_parser(parser_cls, tokenizer, chat_template_kwargs):
    """新 parser 吃 chat_template_kwargs（enable_thinking）；旧 parser 只收 tokenizer。"""
    try:
        return parser_cls(tokenizer, chat_template_kwargs=chat_template_kwargs)
    except TypeError:
        return parser_cls(tokenizer)


def _dummy_request(chat_template_kwargs):
    # extract_* 只读 tools / tool_choice。不构造完整 ChatCompletionRequest，避免绑死某一版协议类。
    return SimpleNamespace(
        model="omc",
        messages=[],
        tools=None,
        tool_choice="auto",
        parallel_tool_calls=True,
        include_reasoning=True,
        skip_special_tokens=False,
        chat_template_kwargs=dict(chat_template_kwargs),
    )


def _reasoning_markers(parser):
    start = getattr(parser, "start_token", None) or getattr(
        parser, "reasoning_start_str", None
    )
    end = getattr(parser, "end_token", None) or getattr(parser, "reasoning_end_str", None)
    if callable(start):
        start = None
    if callable(end):
        end = None
    return start or "", end or ""


def prompt_opens_think(prompt, start_token):
    """Qwen chat template 在 enable_thinking 时把 <think> 写进 prompt，生成正文里往往没有开头标签。"""
    if not prompt or not start_token or start_token not in prompt:
        return False
    marker = "<|im_start|>assistant"
    idx = prompt.rfind(marker)
    tail = prompt[idx:] if idx >= 0 else prompt
    return tail.rfind(start_token) > tail.rfind("</think>")


def prepare_for_reasoning_parser(generation, prompt, start_token, end_token, enable_thinking):
    """旧版 qwen3 reasoning parser 要求正文里同时有开始和结束标签。
    prompt 已经打开 <think>、正文只有 </think> 时，补上开始标签再交给 vLLM parser。
    新版 parser 初始就在 reasoning 状态，多一个开始标签会被吞掉，结果相同。
    """
    text = generation or ""
    if not enable_thinking or not start_token or not end_token:
        return text
    if start_token in text or end_token not in text:
        return text
    if prompt_opens_think(prompt, start_token):
        return start_token + text
    return text


def _split_reasoning(parser, text, request):
    if hasattr(parser, "extract_reasoning"):
        return parser.extract_reasoning(text, request)
    if hasattr(parser, "extract_reasoning_content"):
        return parser.extract_reasoning_content(text, request)
    raise TypeError(
        f"{type(parser).__name__} 没有 extract_reasoning / extract_reasoning_content"
    )


def _arguments_to_str(arguments):
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)


def tool_calls_to_openai(extracted):
    """vLLM ExtractedToolCallInformation / ToolCall -> verify_ans 用的 OpenAI dict。"""
    if extracted is None:
        return []
    if isinstance(extracted, dict):
        calls = extracted.get("tool_calls") or []
    else:
        calls = getattr(extracted, "tool_calls", None) or []
    openai_calls = []
    for tc in calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            name = fn.get("name")
            arguments = fn.get("arguments")
            tc_id = tc.get("id")
            tc_type = tc.get("type") or "function"
        else:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) if fn is not None else None
            arguments = getattr(fn, "arguments", None) if fn is not None else None
            tc_id = getattr(tc, "id", None)
            tc_type = getattr(tc, "type", None) or "function"
        if not name:
            continue
        openai_calls.append(
            {
                "id": tc_id,
                "type": tc_type,
                "function": {
                    "name": str(name),
                    "arguments": _arguments_to_str(arguments),
                },
            }
        )
    return openai_calls


def _extracted_content(extracted):
    if extracted is None:
        return None
    if isinstance(extracted, dict):
        return extracted.get("content")
    return getattr(extracted, "content", None)


class VLLMGenerationParser:
    def __init__(
        self,
        tokenizer,
        reasoning_parser="",
        tool_call_parser="",
        enable_thinking=True,
    ):
        if not tool_call_parser:
            raise ValueError("tool_call_parser 不能为空，例如 hermes 或 qwen3_xml。")

        tool_manager, reason_manager = _load_parser_managers()
        chat_template_kwargs = {"enable_thinking": bool(enable_thinking)}
        self.enable_thinking = bool(enable_thinking)
        self.tokenizer = tokenizer
        self.request = _dummy_request(chat_template_kwargs)
        self.reasoning_parser = None
        self.reasoning_name = reasoning_parser or ""
        self.tool_name = tool_call_parser

        if reasoning_parser:
            try:
                reason_cls = reason_manager.get_reasoning_parser(reasoning_parser)
            except KeyError as exc:
                names = _registered_names(reason_manager, "reasoning_parsers")
                raise KeyError(
                    f"reasoning parser `{reasoning_parser}` 不在当前 vLLM 里。"
                    f" 可用: {', '.join(names) or '(无)'}"
                ) from exc
            self.reasoning_parser = _construct_parser(
                reason_cls, tokenizer, chat_template_kwargs
            )

        try:
            tool_cls = tool_manager.get_tool_parser(tool_call_parser)
        except KeyError as exc:
            names = _registered_names(tool_manager, "tool_parsers")
            raise KeyError(
                f"tool-call parser `{tool_call_parser}` 不在当前 vLLM 里。"
                f" 可用: {', '.join(names) or '(无)'}"
            ) from exc
        self.tool_parser = _construct_parser(tool_cls, tokenizer, chat_template_kwargs)
        # 没配 reasoning parser 时，与 vLLM serving 一样让 tool parser 不要吃掉 <think>。
        if self.reasoning_parser is None and hasattr(self.tool_parser, "skip_reasoning_parsing"):
            self.tool_parser.skip_reasoning_parsing = True

    def _maybe_set_prompt(self, prompt):
        parser = self.reasoning_parser
        if parser is None or not prompt or not hasattr(parser, "set_prompt_token_ids"):
            return
        encode = getattr(self.tokenizer, "encode", None)
        if not callable(encode):
            return
        try:
            token_ids = encode(prompt, add_special_tokens=False)
        except TypeError:
            token_ids = encode(prompt)
        except Exception:
            return
        try:
            parser.set_prompt_token_ids(token_ids)
        except Exception:
            return

    def parse(self, model_output, prompt=""):
        text = model_output or ""
        request = self.request
        reasoning = ""
        content = text

        if self.reasoning_parser is not None:
            self._maybe_set_prompt(prompt)
            start, end = _reasoning_markers(self.reasoning_parser)
            prepared = prepare_for_reasoning_parser(
                text, prompt, start, end, self.enable_thinking
            )
            reasoning_content, parsed_content = _split_reasoning(
                self.reasoning_parser, prepared, request
            )
            reasoning = reasoning_content or ""
            content = parsed_content if parsed_content is not None else ""
            # 补过的开始标签不属于模型输出，从 reasoning 里拿掉。
            if prepared != text and reasoning.startswith(start):
                reasoning = reasoning[len(start) :]

        extracted = self.tool_parser.extract_tool_calls(content, request)
        tool_calls = tool_calls_to_openai(extracted)
        leftover = _extracted_content(extracted)
        if leftover is not None:
            content = leftover or ""

        return ParsedGeneration(reasoning, content, tool_calls)
