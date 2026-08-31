# 三套 test case 构造器（BFCL / HumanEval 共用）
#
# - create_omc_test : 严格 Ceval 结构。BFCL 的 prompt 由 apply_chat_template(messages, tools=)
#                     在外部渲染后塞进 sentence["prompt"]。
# - create_gpu_test : 顶层与 OMC 一致；sentence 用 messages，BFCL 另带 tools / tool_choice，
#                     可直接喂给 chat.completions.create。
# - create_api_test : OpenAI chat.completions 请求体（messages / tools / temperature / max_tokens / ...）
#
# 三种 mode 都 NOT carry category / eval_type 字段——编码在 testCaseName 的 vertical 段。

from transformers import AutoTokenizer

mp = None
_tokenizer_cache = {}


def set_model_config(model_module):
    global mp
    mp = model_module


def _get_tokenizer():
    path = mp.TOKENIZER_CONFIG_PATH
    if path not in _tokenizer_cache:
        _tokenizer_cache[path] = AutoTokenizer.from_pretrained(path)
    return _tokenizer_cache[path]


def render_chat_template(system_prompt, user_prompt):
    """把 system + user 用 tokenizer 的 chat template 渲染成单串字符串（HumanEval 等无 tools）。"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return render_chat_template_messages(messages)


def render_chat_template_messages(messages, tools=None, enable_thinking=None):
    """按真实 serving 路径渲染：messages + 可选 OpenAI tools。

    enable_thinking 默认读 model_config.ENABLE_THINKING。
    模板不支持 tools / enable_thinking 时降级，避免非 Qwen tokenizer 直接炸。
    """
    tokenizer = _get_tokenizer()
    if enable_thinking is None:
        enable_thinking = enable_thinking_flag()
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if tools:
        kwargs["tools"] = tools

    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            return tokenizer.apply_chat_template(messages, **kwargs)


def enable_thinking_flag():
    if mp is None:
        raise ValueError("You must call `set_model_config(mp)` first.")
    return bool(getattr(mp, "ENABLE_THINKING", False))


def _common_top_level():
    if mp is None:
        raise ValueError("You must call `set_model_config(mp)` first.")
    return {
        "inferType": mp.INFER_TYPE,
        "tokenizerType": mp.TOKENIZER_TYPE,
        "tokenizerPath": mp.TOKENIZER_PATH,
        "modelType": mp.MODEL_TYPE,
        "modelPath": mp.MODEL_PATH,
        "weightDir": mp.WEIGHT_DIR,
        "prefixPrompt": mp.PREFIX_PROMPT,
        "pmtCacheOperation": mp.PMT_CACHE_OP,
        "pfxInitTokenLen": mp.PFX_INIT_TOKEN_LEN,
        "loraCfgPath": mp.LORA_CFG_PATH,
    }


def _resolve_stop_seq(extra_stop_seq):
    """模型默认 stopSeq + 数据集额外追加的截止符。
    BFCL 不传 extra_stop_seq；HumanEval 之类需要代码续写"刹车"的数据集可以加 `\\nclass `、`\\nif __name__` 等。"""
    if mp is None:
        raise ValueError("You must call `set_model_config(mp)` first.")
    stop_seq = list(mp.STOP_SEQ)
    if extra_stop_seq:
        # 保留顺序、去重
        for s in extra_stop_seq:
            if s not in stop_seq:
                stop_seq.append(s)
    return stop_seq


def _common_sentence_params(maxGenTokens, extra_stop_seq=None):
    if mp is None:
        raise ValueError("You must call `set_model_config(mp)` first.")
    return {
        "callbackFreq": mp.CALLBACK_FREQ,
        "sampleFlag": mp.SAMPLE_FLAG,
        "seed": mp.SEED,
        "topK": mp.TOPK,
        "topP": mp.TOPP,
        "temperature": mp.TEMPERATURE,
        "maxGenTokens": maxGenTokens,
        "repetitionPenalty": mp.REPETITIONPENALTY,
        "initTokenLen": mp.INIT_TOKEN_LEN,
        "isAsync": mp.IS_ASYNC,
        "stopSeq": _resolve_stop_seq(extra_stop_seq),
        "enableThinking": enable_thinking_flag(),
    }


def create_omc_test(testCaseName, prompt, expect="", maxGenTokens=2000, extra_stop_seq=None):
    """OMC：严格 Ceval 结构。sentences[0] 只含 prompt + Ceval 既有采样参数。
    enableThinking 只是标记（是否开思考已经渲进 prompt），OMC 工具不认这个字段。
    expect 只给评分用，放在字段末尾，不参与推理。"""
    sentence = {"prompt": prompt}
    sentence.update(_common_sentence_params(maxGenTokens, extra_stop_seq))
    sentence["expect"] = expect
    return {
        "testCaseName": testCaseName,
        **_common_top_level(),
        "sentences": [sentence],
    }


def create_gpu_test(
    testCaseName,
    messages,
    expect="",
    maxGenTokens=2000,
    extra_stop_seq=None,
    tools=None,
    tool_choice="auto",
):
    """GPU：顶层与 OMC 同款；sentence 用 messages，BFCL 另带 tools / tool_choice。
    enableThinking 从 model_config.ENABLE_THINKING 写入，evaluator 读这个字段决定请求。
    expect 只给评分用，放在字段末尾，不会送进 vLLM。
    category / eval_type 不进字段，evaluator 从 testCaseName 解析。"""
    sentence = {"messages": messages}
    if tools:
        sentence["tools"] = tools
        sentence["tool_choice"] = tool_choice
    sentence.update(_common_sentence_params(maxGenTokens, extra_stop_seq))
    sentence["expect"] = expect
    return {
        "testCaseName": testCaseName,
        **_common_top_level(),
        "sentences": [sentence],
    }


def create_api_test(
    testCaseName,
    messages,
    expect="",
    maxGenTokens=2000,
    extra_stop_seq=None,
    tools=None,
    tool_choice="auto",
):
    """API：OpenAI chat.completions 请求体（可直接 POST /v1/chat/completions）。
    testCaseName / expect 是评估元数据，服务端会忽略。expect 放最后。"""
    if mp is None:
        raise ValueError("You must call `set_model_config(mp)` first.")
    case = {
        "model": mp.model_name,
        "messages": messages,
    }
    if tools:
        case["tools"] = tools
        case["tool_choice"] = tool_choice
    case["stream"] = False
    case["temperature"] = mp.TEMPERATURE
    case["top_p"] = mp.TOPP
    case["max_tokens"] = maxGenTokens
    case["seed"] = mp.SEED
    case["stop"] = _resolve_stop_seq(extra_stop_seq)
    case["top_k"] = mp.TOPK
    case["repetition_penalty"] = mp.REPETITIONPENALTY
    case["chat_template_kwargs"] = {"enable_thinking": enable_thinking_flag()}
    case["testCaseName"] = testCaseName
    case["expect"] = expect
    return case
