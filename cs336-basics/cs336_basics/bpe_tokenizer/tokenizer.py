from __future__ import annotations
from typing import Iterator, Iterable, List, Tuple
from pathlib import Path
import json
from functools import lru_cache

from cs336_basics.bpe_tokenizer.utils import pre_tokenize


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        GPT2 风格 BPE 分词器，对齐 tiktoken 行为
        :param vocab: id -> bytes 词表
        :param merges: 有序合并规则列表 [(a,b), ...]
        :param special_tokens: 特殊字符串列表
        """
        # 正向词表 id → bytes
        self.vocab: dict[int, bytes] = vocab
        # 反向词表 bytes → id
        self.bytes_to_id: dict[bytes, int] = {b: idx for idx, b in vocab.items()}

        # 合并优先级：pair -> 优先级序号（越小越先合并）
        self.merge_rank: dict[tuple[bytes, bytes], int] = {}
        for rank, pair in enumerate(merges):
            self.merge_rank[pair] = rank

        self.special_tokens: list[str] = (
            special_tokens if special_tokens is not None else []
        )

    def _bpe_merge(self, raw_bytes: bytes) -> list[bytes]:
        """对单个字节串执行完整BPE合并，返回子词字节列表"""
        if not raw_bytes:
            return []
        tokens: list[bytes] = [bytes([b]) for b in raw_bytes]

        while True:
            min_rank = None
            min_idx = -1
            # 本轮完整从头到尾扫描，严格匹配GPT2每轮全局选最小
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                r = self.merge_rank.get(pair)
                if r is None:
                    continue
                if (min_rank is None) or r < min_rank:
                    min_rank = r
                    min_idx = i
            if min_idx == -1:
                break
            # 原地合并，减少列表拷贝，算法逻辑不变
            new_token = tokens[min_idx] + tokens[min_idx + 1]
            tokens[min_idx : min_idx + 2] = [new_token]
        return tokens

    def encode(self, text: str) -> list[int]:
        """完整字符串编码 → token id 列表，严格对齐 tiktoken gpt2"""
        segments = pre_tokenize(text, self.special_tokens)
        ids: list[int] = []
        for seg in segments:
            if seg in self.special_tokens:
                seg_b = seg.encode("utf-8")
                ids.append(self.bytes_to_id[seg_b])
            else:
                seg_bytes = seg.encode("utf-8")
                sub_tokens = self._bpe_merge(seg_bytes)
                for st in sub_tokens:
                    ids.append(self.bytes_to_id[st])
        return ids

    def decode(self, ids: list[int]) -> str:
        """id列表还原原始字符串，encode往返可逆"""
        total_bytes = b""
        for token_id in ids:
            total_bytes += self.vocab[token_id]
        return total_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式迭代编码，逐段生成id，极低内存占用"""
        for chunk in iterable:
            chunk_ids = self.encode(chunk)
            for tid in chunk_ids:
                yield tid


def load_bpe_tokenizer(config_dir: str, special_tokens: list[str]) -> BPETokenizer:
    vocab_path = Path(config_dir) / "vocab.json"
    merges_path = Path(config_dir) / "merges.txt"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
    vocab: dict[int, bytes] = {}
    for sid_str, byte_list in vocab_raw.items():
        sid = int(sid_str)
        vocab[sid] = bytes(byte_list)

    merges: List[Tuple[bytes, bytes]] = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            part1, part2 = line.split()
            b1 = bytes(int(x) for x in part1.split(","))
            b2 = bytes(int(x) for x in part2.split(","))
            merges.append((b1, b2))

    return BPETokenizer(vocab, merges, special_tokens=special_tokens)


@lru_cache(maxsize=None)
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


byte2char = bytes_to_unicode()
char2byte = {v: k for k, v in byte2char.items()}


def gpt2_str_to_bytes(token_str: str) -> bytes:
    out = []
    for c in token_str:
        if c == "Ġ":
            # Ġ 专门还原为空格字节 32
            out.append(32)
        else:
            out.append(char2byte[c])
    return bytes(out)


def load_bpe_tokenizer_gpt2(config_dir: str, special_tokens: list[str]) -> BPETokenizer:
    vocab_path = Path(config_dir) / "vocab_gpt2.json"
    merges_path = Path(config_dir) / "merges_gpt2.txt"

    # 1. 读取 GPT2 标准词表 {token_str: id}，反向构造 id -> bytes
    with open(vocab_path, "r", encoding="utf-8") as f:
        str_to_id = json.load(f)

    vocab: dict[int, bytes] = {}
    for token_str, sid_str in str_to_id.items():
        sid = int(sid_str)
        vocab[sid] = gpt2_str_to_bytes(token_str)

    # 2. 读取 GPT2 merges.txt，每行两个 token 字符串，不再 split(",")
    merges: List[Tuple[bytes, bytes]] = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"[加载警告] 跳过第{line_idx}行格式异常: {repr(line)}")
                continue
            part1, part2 = parts
            b1 = gpt2_str_to_bytes(part1)
            b2 = gpt2_str_to_bytes(part2)
            merges.append((b1, b2))

    tokenizer = BPETokenizer(vocab, merges, special_tokens=special_tokens)

    # 自动绑定 eos_id
    eos_str = "<|endoftext|>"
    if eos_str in special_tokens:
        tokenizer.eos_id = tokenizer.encode(eos_str)[0]

    return tokenizer
