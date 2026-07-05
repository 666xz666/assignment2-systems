import torch

from ..nn import TransformerLM
from .function import softmax


@torch.no_grad()
def decode(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
) -> torch.Tensor:
    r"""
    自回归解码函数，修复Top-p逻辑BUG，带全局缓存保证Prompt永不丢失

    https://www.bilibili.com/video/BV1taSFBNEG4/
    ## 核心原理
    基于自回归方式逐token生成文本，每一步利用上文预测下一个词；
    采用温度缩放调节分布尖锐程度，搭配Top-p核采样控制随机选词范围；
    设计双序列缓存机制，避免上下文窗口截断丢失原始输入Prompt。

    ### 1. 温度缩放
    对最后位置输出logits做缩放，调节预测分布平滑程度：

    $$z_i = \frac{\text{logit}_i}{T}$$

    $T$为temperature，$T\to0$分布尖锐偏向高概率词，$T>1$分布更平缓随机性更强。

    ### 2. Top-p 核采样
    将概率从大到小排序，累加概率首次超过阈值$p$，截断后续低概率token，仅在候选集合内重归一化采样：

    $$\sum_{i=1}^k p_i \ge \text{top\_p}$$

    ### 终止条件
    1. 采样得到EOS结束符，提前终止生成
    2. 生成token数量达到max_new_tokens上限，停止迭代

    Args:
        model: TransformerLM
            仅解码器结构语言模型，内部固定上下文窗口长度，使用RoPE旋转位置编码
        prompt_ids: torch.Tensor, shape [B, seq_len]
            初始提示词token编号，支持批量维度输入
        max_new_tokens: int
            最大允许生成的新词数量上限
        temperature: float
            温度缩放系数，必须大于0
        top_p: float
            核采样累积概率阈值，取值范围 $0<\text{top\_p}\le 1$
        eos_token_id: int
            文本结束符<|endoftext|>对应的token编号，触发提前终止生成

    Returns:
        torch.Tensor, shape [B, total_seq_len]
            原始提示词序列拼接生成内容后的完整token序列
    """
    # 输入合法性校验
    assert temperature > 0.0, "温度系数必须大于0"
    assert 0.0 < top_p <= 1.0, "top-p阈值需要在(0, 1]区间内"
    # 获取模型预设最大上下文窗口长度
    ctx_len = model.context_length

    # 全局缓存：完整保存全部序列，最终返回使用，永远不会因窗口截断丢失开头Prompt
    full_tokens = prompt_ids.clone()
    # 局部窗口序列：仅送入模型做前向推理，长度严格受限上下文上限，防止位置编码越界
    tokens = prompt_ids.clone()

    for _ in range(max_new_tokens):
        # 若推理序列长度超限，只保留末尾ctx_len个token，全局完整序列不受影响
        if tokens.size(1) > ctx_len:
            tokens = tokens[:, -ctx_len:]

        # 模型前向传播，输出每个位置完整词表logits [batch, seq_len, vocab_size]
        logits = model(tokens)
        # 只取出序列最后一位logits，用于预测下一个token
        last_logits = logits[:, -1, :]

        # 1. 温度缩放处理
        scaled_logits = last_logits / temperature

        # 2. Top-p核采样流程
        # 对logits做softmax转为概率分布
        probs = softmax(scaled_logits, dim=-1)
        # 按概率从高到低排序，同时保存排序对应的原始词表下标
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        # 计算排序后概率的累积和，用于判断截断位置
        cum_p = torch.cumsum(sorted_probs, dim=-1)

        # 标记累积概率超出top_p阈值的位置，这部分token需要屏蔽
        mask = cum_p > top_p
        # 强制保留概率最高的第一个token，避免全部被屏蔽后出现全-inf数值异常
        mask[:, 0] = False

        # 根据排序下标，提取排序后对应的原始logits
        sorted_logits = torch.gather(scaled_logits, dim=-1, index=sorted_idx)
        # 超出核范围的token置负无穷，softmax后概率趋近0
        sorted_logits[mask] = -float("inf")

        # 对筛选后的logits重新归一化，得到核内有效概率分布
        final_probs = softmax(sorted_logits, dim=-1)
        # 在核内多项式采样，得到排序空间内的下标
        next_token_idx = torch.multinomial(final_probs, num_samples=1)
        # 将排序空间下标映射回原始词表真实token编号
        next_token = torch.gather(sorted_idx, dim=-1, index=next_token_idx)

        # 推理窗口追加新token，作为下一轮上文输入
        tokens = torch.cat([tokens, next_token], dim=1)
        # 全局完整序列同步追加，保证最终输出包含全部历史
        full_tokens = torch.cat([full_tokens, next_token], dim=1)

        # 单样本场景下，采样出结束符则提前跳出循环
        if full_tokens.size(0) == 1 and next_token.item() == eos_token_id:
            break

    return full_tokens
