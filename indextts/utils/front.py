# -*- coding: utf-8 -*-
import traceback
import re

class TextNormalizer:
    def __init__(self):
        # self.normalizer = Normalizer(cache_dir="textprocessing/tn")
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": ".",
            "·": ",",
            "、": ",",
            "...": "…",
            "……": "…",
            "$": ".",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }

    def match_email(self, email):
        # 正则表达式匹配邮箱格式：数字英文@数字英文.英文
        pattern = r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$'
        return re.match(pattern, email) is not None

    def use_chinese(self, s):
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', s))
        has_digit = bool(re.search(r'\d', s))
        has_alpha = bool(re.search(r'[a-zA-Z]', s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True
        else:
            return False

    def load(self):
        # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        # sys.path.append(model_dir)
        import platform
        if platform.system() == "Darwin":
            from wetext import Normalizer
            self.zh_normalizer = Normalizer(remove_erhua=False,lang="zh",operator="tn")
            self.en_normalizer = Normalizer(lang="en",operator="tn")
        else: # This is line 70
            pass # Keep this pass for the else block
        #from tn.chinese.normalizer import Normalizer as NormalizerZh # (Still commented out)
        #from tn.english.normalizer import Normalizer as NormalizerEn # (Still commented out)
        #self.zh_normalizer = NormalizerZh(remove_interjections=False, remove_erhua=False,overwrite_cache=False) # (Still commented out)
        #self.en_normalizer = NormalizerEn(...) # (Still commented out)

    # This function definition seems duplicated in your original paste, removing the duplicate
    # def infer(self, text): # This should now be correctly unindented relative to the 'else' block
    #     pass # <-- This pass was added here based on the error

    def infer(self, text): # This is the actual function definition
        pass # <--- ADD THIS LINE TO FIX INDENTATION ERROR
        pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
        replaced_text = pattern.sub(lambda x: self.char_rep_map[x.group()], text)
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            # Since normalizers are never initialized due to commenting,
            # we should probably just return the basic cleaned text here.
            # Otherwise, the original code would return "" below,
            # let's return the text after basic punctuation replacement for now.
            return replaced_text
            # return "" # Original code would return this

        # This part will likely never run because the normalizers are None
        try:
            normalizer = self.zh_normalizer if self.use_chinese(replaced_text) else self.en_normalizer
            result = normalizer.normalize(replaced_text)
        except Exception:
            result = "" # Fallback to empty string on error
            print(traceback.format_exc())
        result = self.restore_pinyin_tone_numbers(replaced_text, result)
        return result

    def pinyin_match(self, pinyin):
        pattern = r"(qun)(\d)"
        repl = r"qvn\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(quan)(\d)"
        repl = r"qvan\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(que)(\d)"
        repl = r"qve\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(qu)(\d)"
        repl = r"qv\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(ju)(\d)"
        repl = r"jv\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(jue)(\d)"
        repl = r"jve\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(xun)(\d)"
        repl = r"xvn\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(xue)(\d)"
        repl = r"xve\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(xu)(\d)"
        repl = r"xv\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(juan)(\d)"
        repl = r"jvan\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(jun)(\d)"
        repl = r"jvn\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)

        pattern = r"(xuan)(\d)"
        repl = r"xvan\g<2>"
        pinyin = re.sub(pattern, repl, pinyin)
        return pinyin

    def restore_pinyin_tone_numbers(self,original_text, processed_text):
        # 第一步：恢复拼音后的音调数字（1-4）
        # 建立中文数字到阿拉伯数字的映射
        chinese_to_num = {'一': '1', '二': '2', '三': '3', '四': '4'}

        # 使用正则表达式找到拼音+中文数字的组合（如 "xuan四"）
        def replace_tone(match):
            pinyin = match.group(1)  # 拼音部分
            chinese_num = match.group(2)  # 中文数字部分
            # 将中文数字转换为阿拉伯数字
            num = chinese_to_num.get(chinese_num, chinese_num)
            return f"{pinyin}{num}"

        # 匹配拼音后跟中文数字（一、二、三、四）的情况
        pattern = r'([a-zA-Z]+)([一二三四])'
        restored_text = re.sub(pattern, replace_tone, processed_text)
        restored_text = restored_text.lower()
        restored_text = self.pinyin_match(restored_text)

        return restored_text


if __name__ == '__main__':
    # 测试程序
    text_normalizer = TextNormalizer()
    print(text_normalizer.infer("2.5平方电线"))