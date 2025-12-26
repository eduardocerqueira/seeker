#date: 2025-12-26T16:43:49Z
#url: https://api.github.com/gists/60d94af4234f214342547116d9bb9e62
#owner: https://api.github.com/users/Ineedlearn

import pandas as pd
import re
from collections import Counter


class GameCommentAnalyzer:
    def __init__(self):
        # 2. 分类 (Classification) - 定义核心关键词库
        # 在实际生产中，这里会接入 LLM 进行语义判断，MVP阶段我们用精准关键词匹配
        self.keywords = {
            "new_content": ["黄金柯尔特", "麻将机", "新装备", "新枪", "皮肤", "活动"],  # 新变化风向标
            "gameplay": ["冲刺", "连接", "成就", "手感", "操作", "玩法", "模式"],  # 玩法热度
            "risk": ["bug", "BUG", "卡死", "掉线", "外挂", "脚本", "回档", "闪退", "不仅", "垃圾"]  # 风险预警
        }

    # 1. 数据清洗 (Data Cleaning)
    def parse_log_file(self, file_content):
        """
        将非结构化的文本日志解析为 DataFrame
        """
        dialogues = []
        # 正则匹配格式：[2026-09-15 21:29:32] 昵称(ID): 消息内容
        # 注意：针对文件样本进行适配
        pattern = re.compile(r'^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]\s(.*?)\((.*?)\):\s(.*)$')

        lines = file_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('【'):  # 跳过元数据行
                continue

            match = pattern.match(line)
            if match:
                time_str, nickname, user_id, msg = match.groups()
                dialogues.append({
                    "time": time_str,
                    "user": nickname,
                    "id": user_id,
                    "content": msg
                })

        return pd.DataFrame(dialogues)

    # 3. 核心逻辑 (Core Logic)
    def analyze_data(self, df):
        """
        对清洗后的数据进行分类打标和统计
        """
        if df.empty:
            return df, {}

        # --- A. 自动打标 ---
        def get_category(text):
            text = text.lower()
            if any(k in text for k in self.keywords["risk"]):
                return "风险/Bug"
            elif any(k in text for k in self.keywords["new_content"]):
                return "新内容/装备"
            elif any(k in text for k in self.keywords["gameplay"]):
                return "玩法讨论"
            return "日常闲聊"

        # --- B. 情感预判 (简单版) ---
        # 真实项目会调用 sentiment-analysis 模型，这里用规则模拟
        def get_sentiment(text):
            pos_words = ["好", "强", "爽", "爱", "喜欢", "不错"]
            neg_words = ["烂", "垃圾", "卡", "难受", "恶心", "bug"]
            score = 0
            for w in pos_words: score += text.count(w)
            for w in neg_words: score -= text.count(w)
            return "正面" if score > 0 else ("负面" if score < 0 else "中性")

        # 应用逻辑到每一行
        df['category'] = df['content'].apply(get_category)
        df['sentiment'] = df['content'].apply(get_sentiment)

        # --- C. 聚合统计 (为UI准备数据) ---
        stats = {
            "total_msgs": len(df),
            "risk_count": len(df[df['category'] == "风险/Bug"]),
            # 统计讨论最多的热词（排除常见停用词）
            "top_words": Counter("".join(df['content'].tolist())).most_common(20)
        }

        return df, stats

# --- 使用示例 ---
# analyzer = GameCommentAnalyzer()
# df_raw = analyzer.parse_log_file(raw_text_data)
# df_analyzed, summary_stats = analyzer.analyze_data(df_raw)
