import regex as re

# 全局固定 GPT2 官方预分词正则，写死内部
_GPT2_PRE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pre_tokenize(
    text: str,
    special_tokens: list[str] | None = None,
    base_pattern: re.Pattern | None = None,
) -> list[str]:
    """
    通用预分词入口，默认内置GPT2预分词规则
    :param text: 待切分原始字符串
    :param base_pattern: 可选自定义正则；不传则自动使用内置GPT2正则
    :param special_tokens: 特殊token列表，最长优先匹配分割
    :return: 预分词字符串列表
    """
    pattern = base_pattern if base_pattern is not None else _GPT2_PRE_PATTERN

    if not special_tokens:
        return pattern.findall(text)

    # 特殊token按长度降序，最长匹配优先
    sorted_specials = sorted(special_tokens, key=lambda s: len(s), reverse=True)
    escaped = [re.escape(tok) for tok in sorted_specials]
    split_pat = re.compile(f"({'|'.join(escaped)})")
    parts = split_pat.split(text)

    result = []
    for part in parts:
        if part == "":
            continue
        if part in special_tokens:
            result.append(part)
        else:
            result.extend(pattern.findall(part))
    return result
