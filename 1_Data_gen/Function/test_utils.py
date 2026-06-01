# BFCL 数据生成的三套 test case 构造器
#
# - create_omc_test : 严格遵循原版 Ceval 结构，sentence 只允许出现 Ceval 既有字段。
#                     system + user 通过 apply_chat_template 在外部拼成单一 prompt 字符串塞进 sentence["prompt"]。
# - create_gpu_test : 顶层与 OMC 一致（保留 inferType / tokenizerPath 等），但 sentence
#                     从 prompt 改成 messages（OpenAI / vLLM 风格 role/content 列表），可直接喂给
#                     chat.completions.create。
# - create_api_test : 顶层即 OpenAI 调用结构（messages / options 都在最外层）。
#
# 三种 mode 都 NOT carry category / eval_type 字段——这两个属性已经编码在 testCaseName
# 的 vertical 段里（{project}-bfcl-test-{category}-{i}），evaluator 那侧统一从 testCaseName
# 解析 category，再用 verify_ans.CATEGORY_TO_EVAL_TYPE 查 eval_type，这样三种 mode 完全对等。

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
    """把 system + user 用 tokenizer 的 chat template 渲染成单串字符串。"""
    tokenizer = _get_tokenizer()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


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
    }


def create_omc_test(testCaseName, prompt, expect="", maxGenTokens=2000, extra_stop_seq=None):
    """OMC：严格 Ceval 结构。sentences[0] 只含 prompt/expect + Ceval 既有采样参数。
    system prompt 已经在 prompt 里通过 apply_chat_template 渲染好。"""
    sentence = {"prompt": prompt, "expect": expect}
    sentence.update(_common_sentence_params(maxGenTokens, extra_stop_seq))
    return {
        "testCaseName": testCaseName,
        **_common_top_level(),
        "sentences": [sentence],
    }


def create_gpu_test(testCaseName, messages, expect="", maxGenTokens=2000, extra_stop_seq=None):
    """GPU：顶层与 OMC 同款；sentence 用 messages 而非 prompt，可以直接送 vLLM
    chat.completions.create。category / eval_type 不进字段，evaluator 从 testCaseName 解析。"""
    sentence = {"messages": messages, "expect": expect}
    sentence.update(_common_sentence_params(maxGenTokens, extra_stop_seq))
    return {
        "testCaseName": testCaseName,
        **_common_top_level(),
        "sentences": [sentence],
    }


def create_api_test(testCaseName, messages, expect="", maxGenTokens=2000, extra_stop_seq=None):
    """API：OpenAI 风格顶层结构（messages / options 都在最外层）。
    category / eval_type 不进字段，evaluator 从 testCaseName 解析。"""
    if mp is None:
        raise ValueError("You must call `set_model_config(mp)` first.")
    return {
        "model": mp.model_name,
        "testCaseName": testCaseName,
        "messages": messages,
        "expect": expect,
        "stream": False,
        "options": {
            "seed": 99,
            "num_predict": maxGenTokens,
            "temperature": mp.TEMPERATURE,
            "top_k": mp.TOPK,
            "top_p": mp.TOPP,
            "repeat_penalty": mp.REPETITIONPENALTY,
            "stop": _resolve_stop_seq(extra_stop_seq),
        },
    }
