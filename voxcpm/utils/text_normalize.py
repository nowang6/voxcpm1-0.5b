# some functions are copied from https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/utils/frontend_utils.py
"""文本正则化（text normalization）工具：TTS 前置清洗——去除 Markdown/表情符号/换行、
统一标点与全半角、把特殊符号与数字改写为可朗读的中文/英文读法，并支持长文本分段。"""
import re
import regex
import inflect
from wetext import Normalizer

# \u5339\u914d\u8fde\u7eed\u6c49\u5b57\uff08CJK \u7edf\u4e00\u8868\u610f\u6587\u5b57\u533a\u95f4\uff09\u7684\u6b63\u5219\uff0c\u7528\u4e8e\u5224\u65ad\u6587\u672c\u662f\u5426\u542b\u4e2d\u6587
chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")


# whether contain chinese character
# \u5224\u65ad\u6587\u672c\u4e2d\u662f\u5426\u5305\u542b\u4e2d\u6587\u5b57\u7b26
def contains_chinese(text):
    """\u5224\u65ad\u6587\u672c\u4e2d\u662f\u5426\u5305\u542b\u6c49\u5b57\u3002"""
    return bool(chinese_char_pattern.search(text))


# replace special symbol
# 替换特殊符号：把数学/比较符号改写成对应的中文读法
def replace_corner_mark(text):
    """把特殊符号替换为中文读法：上标 2/3、根号、约等于、小于号。"""
    text = text.replace("²", "平方")  # 上标 ² -> “平方”
    text = text.replace("³", "立方")  # 上标 ³ -> “立方”
    text = text.replace("√", "根号")  # 根号符号 -> “根号”
    text = text.replace("≈", "约等于")  # 约等号 -> “约等于”
    text = text.replace("<", "小于")  # 小于号 -> “小于”
    return text


# remove meaningless symbol
# 移除无意义符号：各类括号/反引号/破折号替换为空格或直接删除
def remove_bracket(text):
    """移除对朗读无意义的符号：全角括号与方头括号换成空格，反引号删除，中文破折号换成空格。"""
    text = text.replace("（", " ").replace("）", " ")  # 全角圆括号 -> 空格
    text = text.replace("【", " ").replace("】", " ")  # 全角方头括号 -> 空格
    text = text.replace("`", "").replace("`", "")  # 反引号（Markdown 代码标记）-> 删除
    text = text.replace("——", " ")  # 中文破折号 -> 空格
    return text


# spell Arabic numerals
# 把阿拉伯数字拼写为英文单词（用于英文文本的数字读法）
def spell_out_number(text: str, inflect_parser):
    """把文本中连续的阿拉伯数字段转换为英文单词读法（如 21 -> twenty-one）。"""
    new_text = []
    st = None  # 当前数字段的起始下标；None 表示当前不在数字段中
    for i, c in enumerate(text):
        if not c.isdigit():
            # 遇到非数字字符：若此前处于数字段，先把该段转成单词再拼接
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            # 记录数字段的起点
            if st is None:
                st = i
    # 扫描结束后若仍停在数字段，把结尾这段数字也转成单词
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
# 分段逻辑：
# 1. 每段最大长度 token_max_n、最小长度 token_min_n；最后一段若短于 merge_len 则并入上一段
# 2. 句长按语言计算（中文按字符数，其他语言按分词数）
# 3. 依据标点切分句子
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
    """把长文本按标点切分并贪心合并成若干适合 TTS 朗读的段落。

    Args:
        text: 待切分的文本。
        tokenize: 分词函数（非中文语言用于计算句长）。
        lang: 语言（"zh" 为中文，其余按英文等处理），决定句长计算方式与断句标点。
        token_max_n: 单段最大长度，超过则在之前断开成段。
        token_min_n: 单段最小长度，断开前当前段至少达到该长度。
        merge_len: 末段合并阈值，末段短于该值时并入前一段。
        comma_split: 是否把中英文逗号也作为断句标点（切分更细）。

    Returns:
        切分合并后的段落列表。
    """
    def calc_utt_length(_text: str):
        # 计算句长：中文按字符数，其他语言按分词数
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        # 判断末段是否过短（少于 merge_len）而需要并入上一段
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    # 按语言选择断句标点集合（中文包含全角标点）
    if lang == "zh":
        pounc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    else:
        pounc = [".", "?", "!", ";", ":"]
    if comma_split:
        # 需要更细粒度切分时，把中英文逗号也纳入断句标点
        pounc.extend(["，", ","])
    st = 0
    utts = []
    # 逐字符扫描，遇到断句标点即切出一个短句（标点并入句尾）
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            # 标点后紧跟引号时把引号并入该句，避免引号孤悬在下一句开头
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1
    if len(utts) == 0:
        # 整段没有任何断句标点：直接在末尾补一个句号作为唯一句子
        if lang == "zh":
            utts.append(text + "。")
        else:
            utts.append(text + ".")
    final_utts = []
    cur_utt = ""
    # 贪心合并：合并后会超过 token_max_n、且当前段已达到 token_min_n 时断开成段
    for utt in utts:
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    # 处理最后一段：过短则并入前一段，否则单独成段
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


# remove blank between chinese character
# 去除中文字符之间的空格（只保留英文单词之间的空格，实现空白归一化）
def replace_blank(text: str):
    """删除汉字间的空格：仅当空格两侧均为 ASCII 字符（英文单词场景）时才保留该空格。"""
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            # 空格前后都是非空格的 ASCII 字符（即英文单词之间）才保留，否则丢弃
            if (text[i + 1].isascii() and text[i + 1] != " ") and (text[i - 1].isascii() and text[i - 1] != " "):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def clean_markdown(md_text: str) -> str:
    """清理 Markdown 语法：去除代码块/行内代码/图片/HTML 标签/标题符号，链接只保留文本，列表符号删除，并压缩多余空行。"""
    # 去除代码块 ``` ```（包括多行）
    md_text = re.sub(r"```.*?```", "", md_text, flags=re.DOTALL)

    # 去除内联代码 `code`
    md_text = re.sub(r"`[^`]*`", "", md_text)

    # 去除图片语法 ![alt](url)
    md_text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", md_text)

    # 去除链接但保留文本 [text](url) -> text
    md_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md_text)

    # 替换无序列表符号
    md_text = re.sub(r"^(\s*)-\s+", r"\1", md_text, flags=re.MULTILINE)

    # 去除HTML标签
    md_text = re.sub(r"<[^>]+>", "", md_text)

    # 去除标题符号（#）
    md_text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.MULTILINE)

    # 去除多余空格和空行
    md_text = re.sub(r"\n\s*\n", "\n", md_text)  # 多余空行
    md_text = md_text.strip()

    return md_text


def clean_text(text):
    """通用文本清洗：去 Markdown 语法、去表情符号，换行/制表符归一化为空格，中文弯引号转英文直引号。"""
    # 去除 Markdown 语法
    text = clean_markdown(text)
    # 匹配并移除表情符号
    text = regex.compile(r"\p{Emoji_Presentation}|\p{Emoji}\uFE0F", flags=regex.UNICODE).sub("", text)
    # 去除换行符
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")  # 制表符同样归一化为空格
    text = text.replace("“", '"').replace("”", '"')  # 中文弯引号统一为英文直引号
    return text


class TextNormalizer:
    """文本正则化器：按语言（中文/英文）选择相应 TN 流水线，把原始文本改写为适合 TTS 朗读的规范文本。"""

    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer: 分词器（供分段等场景使用），可为 None。
        """
        self.tokenizer = tokenizer
        # 中英文各自的 wetext TN 模型；中文额外开启儿化音去除（remove_erhua）
        self.zh_tn_model = Normalizer(lang="zh", operator="tn", remove_erhua=True)
        self.en_tn_model = Normalizer(lang="en", operator="tn")
        # inflect 引擎：把阿拉伯数字转为英文单词读法
        self.inflect_parser = inflect.engine()

    def normalize(self, text, split=False):
        """对文本做正则化：先判定语言，再走中文或英文的 TN 流水线。

        中文流水线：预修复等号/连字符的误判 -> wetext 中文 TN -> 去汉字间空格 ->
        特殊符号转中文读法 -> 去无意义符号；英文流水线：wetext 英文 TN + 数字拼写。

        Args:
            text: 待正则化的原始文本。
            split: 是否分段输出（当前实现仅在 split=False 时直接返回文本）。

        Returns:
            split 为 False 时返回正则化后的文本。
        """
        # 去除 Markdown 语法，去除表情符号，去除换行符
        # 按是否包含汉字判定语言，决定走中文还是英文 TN 规则
        lang = "zh" if contains_chinese(text) else "en"
        text = clean_text(text)
        if lang == "zh":
            # —— 中文 TN 前的针对性修复 ——
            text = text.replace(
                "=", "等于"
            )  # 修复 ”550 + 320 等于 870 千卡。“ 被错误正则为 ”五百五十加三百二十等于八七十千卡.“
            if re.search(r"([\d$%^*_+≥≤≠×÷?=])", text):  # 避免 英文连字符被错误正则为减
                text = re.sub(r"(?<=[a-zA-Z0-9])-(?=\d)", " - ", text)  # 修复 x-2 被正则为 x负2
            # wetext 中文 TN：数字转为中文读法等
            text = self.zh_tn_model.normalize(text)
            # 依次：去汉字间空格、特殊符号转中文读法、去无意义符号
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = remove_bracket(text)
        else:
            # wetext 英文 TN：数字/日期/单位等转为英文读法
            text = self.en_tn_model.normalize(text)
            # 再把剩余的阿拉伯数字拼写为英文单词
            text = spell_out_number(text, self.inflect_parser)
        if split is False:
            # 不分段时直接返回正则化结果
            return text
