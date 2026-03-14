"""LLM 第一印象测试 — 用API测量AI模型对名字的隐性联想"""

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_ZH = """请你想象一个叫「{name}」的中国女性。请描述：
1. 你对她的第一印象（3-5个形容词）
2. 你猜测她可能的职业（给出3个最可能的）
3. 你猜测她的教育背景
4. 你猜测她的家庭经济状况（用1-10评分，10为最高）
5. 你猜测她的幸福感（用1-10评分）
6. 你猜测她的社会地位（用1-10评分）

只输出JSON格式，不要解释。JSON格式如下：
{{"impressions": ["形容词1", ...], "occupations": ["职业1", ...], "education": "教育背景", "economic_score": 8, "happiness_score": 8, "social_score": 8}}"""

PROMPT_TEMPLATE_EN = """Imagine a woman named "{name}". Please describe:
1. Your first impression (3-5 adjectives)
2. Her likely occupations (give 3 most probable)
3. Her likely educational background
4. Her family economic status (rate 1-10, 10 highest)
5. Her happiness level (rate 1-10)
6. Her social standing (rate 1-10)

Output ONLY valid JSON:
{{"impressions": ["adj1", ...], "occupations": ["occ1", ...], "education": "background", "economic_score": 8, "happiness_score": 8, "social_score": 8}}"""


@dataclass
class LLMImpressionResult:
    """单个名字的LLM第一印象测试结果。"""
    name: str
    economic_scores: list[float] = field(default_factory=list)
    happiness_scores: list[float] = field(default_factory=list)
    social_scores: list[float] = field(default_factory=list)
    impressions: list[list[str]] = field(default_factory=list)
    occupations: list[list[str]] = field(default_factory=list)
    raw_responses: list[dict] = field(default_factory=list)

    @property
    def avg_economic(self) -> float:
        return float(np.mean(self.economic_scores)) if self.economic_scores else 0.0

    @property
    def avg_happiness(self) -> float:
        return float(np.mean(self.happiness_scores)) if self.happiness_scores else 0.0

    @property
    def avg_social(self) -> float:
        return float(np.mean(self.social_scores)) if self.social_scores else 0.0

    @property
    def composite_llm_score(self) -> float:
        """综合LLM得分 (0-10)。"""
        return (self.avg_economic + self.avg_happiness + self.avg_social) / 3


def _call_anthropic(prompt: str, temperature: float = 0.7) -> str:
    """调用 Anthropic Claude API。"""
    import anthropic

    client = anthropic.Anthropic()  # 自动读取 ANTHROPIC_API_KEY
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_openai(prompt: str, temperature: float = 0.7) -> str:
    """调用 OpenAI API。"""
    import openai

    client = openai.OpenAI()  # 自动读取 OPENAI_API_KEY
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content


def _parse_response(text: str) -> dict | None:
    """从LLM响应中解析JSON。"""
    text = text.strip()
    # 尝试提取JSON块
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试修复常见问题
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("无法解析LLM响应: %s", text[:200])
            return None


def test_name_impression(
    name: str,
    repeat: int = 10,
    api: str = "anthropic",
    language: str = "zh",
    temperature: float = 0.7,
) -> LLMImpressionResult:
    """对单个名字进行多次LLM第一印象测试。

    Parameters
    ----------
    name : 候选名字
    repeat : 重复次数
    api : "anthropic" 或 "openai"
    language : "zh" 或 "en"
    temperature : 采样温度
    """
    template = PROMPT_TEMPLATE_ZH if language == "zh" else PROMPT_TEMPLATE_EN
    prompt = template.format(name=name)
    call_fn = _call_anthropic if api == "anthropic" else _call_openai

    result = LLMImpressionResult(name=name)

    for i in range(repeat):
        try:
            raw_text = call_fn(prompt, temperature=temperature)
            parsed = _parse_response(raw_text)

            if parsed is None:
                continue

            result.raw_responses.append(parsed)

            eco = parsed.get("economic_score", 0)
            hap = parsed.get("happiness_score", 0)
            soc = parsed.get("social_score", 0)

            # 确保分数在合理范围
            result.economic_scores.append(max(1, min(10, float(eco))))
            result.happiness_scores.append(max(1, min(10, float(hap))))
            result.social_scores.append(max(1, min(10, float(soc))))

            if "impressions" in parsed:
                result.impressions.append(parsed["impressions"])
            if "occupations" in parsed:
                result.occupations.append(parsed["occupations"])

        except Exception as e:
            logger.error("名字 '%s' 第 %d 次测试失败: %s", name, i + 1, e)

    logger.info(
        "名字 '%s' 完成 %d/%d 次测试, 综合得分=%.2f",
        name, len(result.raw_responses), repeat, result.composite_llm_score,
    )
    return result


def batch_test_impressions(
    names: list[str],
    repeat: int = 10,
    api: str = "anthropic",
    language: str = "zh",
    temperature: float = 0.7,
) -> list[LLMImpressionResult]:
    """批量测试多个名字。"""
    results = []
    for i, name in enumerate(names):
        logger.info("测试进度: %d/%d — %s", i + 1, len(names), name)
        result = test_name_impression(
            name, repeat=repeat, api=api,
            language=language, temperature=temperature,
        )
        results.append(result)

    results.sort(key=lambda r: r.composite_llm_score, reverse=True)
    return results
