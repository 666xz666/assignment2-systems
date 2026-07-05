import os
import pathlib
from collections import Counter, defaultdict
from typing import BinaryIO
import multiprocessing

from cs336_basics.bpe_tokenizer.utils import pre_tokenize


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_token_bytes: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    Split on ANY of the provided special tokens.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # 参数校验：必须是非空bytes列表
    assert isinstance(special_token_bytes, list), (
        "split_tokens must be a list of bytestrings"
    )
    assert len(special_token_bytes) > 0, "split_tokens cannot be empty list"
    for tok in special_token_bytes:
        assert isinstance(tok, bytes), (
            "Each split token must be represented as a bytestring"
        )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 遍历所有分隔符，找到最早出现的匹配位置
            min_pos = None
            for tok in special_token_bytes:
                pos = mini_chunk.find(tok)
                if pos != -1:
                    if min_pos is None or pos < min_pos:
                        min_pos = pos

            if min_pos is not None:
                chunk_boundaries[bi] = initial_position + min_pos
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args) -> Counter[tuple[int, ...]]:
    """
    单进程处理文件分片，GPT2预分词 + 特殊token隔离，返回词频
    args: (input_path: Path, start: int, end: int, special_tokens: list[str])

    >>> str = 'hello'
    >>> str.encode("utf-8", errors="ignore")
    b'hello'
    >>> tu = tuple(str.encode("utf-8", errors="ignore"))
    >>> tu
    (104, 101, 108, 108, 111)
    >>>
    """
    input_path, start, end, special_tokens = args
    word_counter = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        raw_data = f.read(end - start)

    chunk = raw_data.decode("utf-8", errors="ignore")

    # 直接调用全局预分词函数，不再重复实现分词逻辑
    token_str_list = pre_tokenize(chunk, special_tokens=special_tokens)

    for word_str in token_str_list:
        if word_str in special_tokens:
            # 不需要统计特殊token
            continue
        word_byte = word_str.encode("utf-8", errors="ignore")
        token_tuple = tuple(word_byte)
        word_counter[token_tuple] += 1

    return word_counter


class BPETrainer:
    def __init__(
        self,
        input_path: pathlib.Path,
        vocab_size: int,
        special_tokens: list[str],
        desired_num_chunks: int | None = 1000,
    ):
        # ========== 新增健壮性校验 ==========
        base_byte_vocab_size = 256
        num_special = len(special_tokens)
        min_required_vocab = base_byte_vocab_size + num_special
        if vocab_size < min_required_vocab:
            raise ValueError(
                f"目标词表大小过小！"
                f"\n基础字节词表固定256个，特殊token共{num_special}个，"
                f"\n最少需要 vocab_size = {min_required_vocab}，"
                f"\n当前传入 vocab_size = {vocab_size}，请调大词表上限。"
            )

        # 入参
        self.input_path: pathlib.Path = input_path
        self.target_vocab_size: int = vocab_size
        self.special_tokens: list[str] = special_tokens
        self.desired_num_chunks: int = desired_num_chunks

        # 特殊token字节缓存
        self.special_token_bytes: list[bytes] = [
            s.encode("utf-8", errors="ignore") for s in self.special_tokens
        ]
        self.special_token_byte_set: set[bytes] = set(self.special_token_bytes)

        # 全局词表映射
        self.vocab: dict[int, bytes] = {}
        self.next_token_id = 0

        # Pair统计结构（增量BPE核心结构）这里存的int全部都是词表中对应的id
        self.word_counts: Counter[tuple[int, int]] = Counter()
        self.pair_counts: Counter[tuple[int, int]] = Counter()
        self.pair_to_words: defaultdict[tuple[int, int], set[tuple[int, ...]]] = (
            defaultdict(set)
        )
        self.cur_max_pair_freq: int = -1
        self.candidates: set[tuple[int, int]] = set()
        self.changed_pairs: set[tuple[int, int]] = set()

        # 合并规则列表（输出结果）
        self.merges: list[tuple[bytes, bytes]] = []

    def _init_vocab_mappings(self) -> None:
        """初始化基础词表：0~255单字节 + 自定义特殊Token"""
        # 0-255 原始字节
        for byte_val in range(256):
            b = bytes([byte_val])
            self.vocab[self.next_token_id] = b
            self.next_token_id += 1
        # 插入特殊token，避免重复
        for sp_byte in self.special_token_bytes:
            self.vocab[self.next_token_id] = sp_byte
            self.next_token_id += 1

    def _init_counts(self) -> None:
        """多进程分片统计全局预分词频次，适配process_chunk输出int元组"""
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, self.desired_num_chunks, self.special_token_bytes
            )

        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((self.input_path, start, end, self.special_tokens))

        num_processers = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_processers) as pool:
            # 流式处理多线程结果（因为操作符合交换律）
            for res in pool.imap_unordered(process_chunk, tasks):
                self.word_counts.update(res)
                # 及时释放内存
                del res

        # 初始化pair信息
        for word_ids, cnt in self.word_counts.items():
            # 这里id肯定是0-255之间，就等于字节值，word_bytes == word_ids
            for p in zip(word_ids[:-1], word_ids[1:]):
                # 统计pair信息
                self.pair_counts[p] += cnt
                self.cur_max_pair_freq = max(
                    self.cur_max_pair_freq, self.pair_counts[p]
                )
                self.pair_to_words[p].add(tuple(word_ids))
                self.candidates.add(p)

    def _get_new_words(
        self,
        pair: tuple[int, int],
        word: tuple[int, ...],
        word_count: int,
        pair_id: int,
    ) -> list[int]:
        """把word中的pair替换成新词汇id， 返回new_word"""
        new_word = []
        i = 0
        has_pair = False
        while i < len(word):
            if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                has_pair = True
                new_word.append(pair_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)

        # 这里操作是先把旧word影响一整个删掉，
        # 再添加new_word的影响
        # 直觉上只要修改best_pair两边的pair
        # 但是时间复杂度是一样的（极端情况word是best_pair循环组成的）
        # 而整体操作比较简单
        if has_pair:
            for p in zip(word[:-1], word[1:]):
                self.pair_counts[p] -= word_count
                if self.pair_counts[p] <= 0:
                    # 剪枝
                    del self.pair_counts[p]
                self.pair_to_words[p].discard(word)

            for p in zip(new_word[:-1], new_word[1:]):
                self.pair_counts[p] += word_count
                self.pair_to_words[p].add(new_word)
                self.changed_pairs.add(p)

        return new_word

    def _merge_one_step(self) -> bool:
        """【增量更新版】一轮BPE合并：仅修改受影响单词与Pair，不再全局重统计"""
        if not self.pair_counts:
            return False

        # 更新candidates
        self.candidates.update(self.changed_pairs)
        self.changed_pairs = set()
        local_max = max(self.pair_counts[p] for p in self.candidates)
        if local_max < self.cur_max_pair_freq:
            self.cur_max_pair_freq = max(self.pair_counts.values())
            self.candidates = {
                p
                for p, cnt in self.pair_counts.items()
                if cnt == self.cur_max_pair_freq
            }
        else:
            self.cur_max_pair_freq = local_max
            self.candidates = {
                p
                for p in self.candidates
                if self.pair_counts[p] == self.cur_max_pair_freq
            }

        # 获取best_pair
        best_pair = max(
            self.candidates, key=lambda p: (self.vocab[p[0]], self.vocab[p[1]])
        )

        # 更新词汇和合并规则
        new_id = self.next_token_id
        self.next_token_id += 1
        self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
        self.merges.append((self.vocab[best_pair[0]], self.vocab[best_pair[1]]))

        # 被影响的单词, 不加list会直接引用
        affected_words = list(self.pair_to_words[best_pair])
        for word in affected_words:
            count = self.word_counts.pop(word)
            new_word = self._get_new_words(best_pair, word, count, new_id)
            self.word_counts[new_word] += count

        return True

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """主训练入口，迭代合并直到词表达到目标大小"""
        # 初始化流程（仅此处全局统计一次pair）
        self._init_vocab_mappings()
        self._init_counts()

        # 迭代增量合并
        while len(self.vocab) < self.target_vocab_size:
            success = self._merge_one_step()
            if not success:
                # 没有可合并的pair
                break

        return self.vocab, self.merges
