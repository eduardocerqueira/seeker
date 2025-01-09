#date: 2025-01-09T16:57:13Z
#url: https://api.github.com/gists/1a335fdf75baed3d8313ccf5cc53901e
#owner: https://api.github.com/users/Q-Bug4

from openai import OpenAI, AsyncOpenAI
from typing import Optional, Dict, List, Any
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """管理不同角色的提示模板"""

    FORMULA_SYNTAX = """
    公式必须遵循以下语法规则：
    LET 结果 = 表达式 WHERE 变量定义

    示例：
    - LET 面积 = 长度 * 宽度 WHERE 长度=5, 宽度=3
    - LET 体积 = 长度 * 宽度 * 高度 WHERE 长度=2, 宽度=3, 高度=4
    """

    GENERATOR_INSTRUCTIONS = """
    你是一个公式生成器。请创建满足以下要求的数学公式：
    - 简单实用
    - 易于理解
    - 数学上正确
    - 严格遵循指定语法

    你的回复必须仅包含符合语法的公式，不要包含任何其他解释或文字。
    """

    CRITIC_INSTRUCTIONS = """
    你是一个公式评审员。请从以下几个方面评估公式：
    1. 语法正确性
    2. 数学有效性
    3. 简洁实用性

    如果发现任何问题，请提供具体的反馈意见。
    请明确标注公式是"已通过"还是"未通过"。

    请按以下JSON格式返回结果：
    {
        "is_valid": true/false,
        "feedback": "..."
    }
    """

    PARSER_INSTRUCTIONS = """
    你是一个公式解析器。请解析输入的公式文本，提取其中符合语法规则的公式。

    你的回复必须是一个JSON对象，格式如下：
    {
        "found": true/false,
        "formula": "完整的公式文本"
    }

    如果文本中没有找到符合语法的公式，将found设为false，其他字段设为null。
    """

    @classmethod
    def get_generator_prompt(cls) -> str:
        """获取完整的生成器提示"""
        return f"{cls.GENERATOR_INSTRUCTIONS}\n\n{cls.FORMULA_SYNTAX}"

    @classmethod
    def get_critic_prompt(cls) -> str:
        """获取完整的评审员提示"""
        return f"{cls.CRITIC_INSTRUCTIONS}\n\n{cls.FORMULA_SYNTAX}"

    @classmethod
    def get_parser_prompt(cls) -> str:
        """获取完整的解析器提示"""
        return f"{cls.PARSER_INSTRUCTIONS}\n\n{cls.FORMULA_SYNTAX}"


class BaseAgent(ABC):
    """具有通用OpenAI功能的基础代理类"""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url="http://192.168.0.120:3001/v1")
        self._setup_system_prompt()

    @abstractmethod
    def _setup_system_prompt(self):
        """设置代理特定的系统提示"""
        pass

    async def _chat_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7
    ) -> str:
        """通用的OpenAI聊天完成方法"""
        response = await self.client.chat.completions.create(
            model="qwen-turbo",
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content


class FormulaParser(BaseAgent):
    """负责解析和验证公式的代理"""

    def _setup_system_prompt(self):
        self.system_prompt = PromptTemplate.get_parser_prompt()

    async def parse_formula(self, text: str) -> Dict[str, Any]:
        """解析文本中的公式"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请解析以下文本中的公式：\n{text}"}
        ]

        response = await self._chat_completion(messages, temperature=0.3)
        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            return {"found": False, "formula": None}


class FormulaGenerator(BaseAgent):
    """负责生成公式的代理"""

    def _setup_system_prompt(self):
        self.system_prompt = PromptTemplate.get_generator_prompt()

    async def generate_formula(self, task: str, feedback: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请为以下任务生成公式：{task}"}
        ]

        if feedback:
            messages.append({"role": "user", "content": f"上一次尝试需要修改。反馈意见：{feedback}"})

        return await self._chat_completion(messages, temperature=0.7)


class FormulaCritic(BaseAgent):
    """负责评估公式的代理"""

    def _setup_system_prompt(self):
        self.system_prompt = PromptTemplate.get_critic_prompt()

    def _clean_response(self, response: str) -> str:
        """Clean response by removing markdown code blocks and extracting JSON"""
        # Remove markdown code block markers
        if response.startswith('```') and response.endswith('```'):
            # Split by newline and remove first and last lines (```json and ```)
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        return response.strip()

    def _extract_feedback(self, response: str) -> str:
        """Extract actual feedback message from response"""
        import json
        try:
            # Try to parse as JSON first
            data = json.loads(response)
            if isinstance(data, dict) and "feedback" in data:
                return data["feedback"]
        except json.JSONDecodeError:
            pass
        
        # If not JSON or no feedback field, return the first 200 chars
        return response[:200]

    async def evaluate_formula(self, formula: str) -> Dict[str, Any]:
        if not formula or not isinstance(formula, str):
            raise ValueError("Formula must be a non-empty string")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请评估以下公式：{formula}"}
        ]

        try:
            response = await self._chat_completion(messages, temperature=0.3)
            
            logging.debug(f"Raw critic response: {response}")
            
            if not response or not isinstance(response, str):
                raise ValueError("Invalid response from LLM")

            # Clean the response before parsing
            cleaned_response = self._clean_response(response)
            logging.debug(f"Cleaned response: {cleaned_response}")

            import json
            try:
                result = json.loads(cleaned_response)
                # Validate required fields
                if not isinstance(result, dict) or "is_valid" not in result or "feedback" not in result:
                    raise ValueError("Response missing required fields")
                
                return {
                    "approved": result["is_valid"],
                    "feedback": result["feedback"]
                }
            except json.JSONDecodeError:
                # Fallback: Try to extract meaningful information from non-JSON response
                is_valid = (
                    '"is_valid": true' in cleaned_response or 
                    '"is_valid":true' in cleaned_response or
                    "已通过" in cleaned_response or 
                    "通过" in cleaned_response
                )
                
                feedback = self._extract_feedback(cleaned_response)
                
                logging.warning(f"Failed to parse JSON response, using fallback parsing. Is valid: {is_valid}")
                
                return {
                    "approved": is_valid,
                    "feedback": feedback
                }

        except Exception as e:
            logging.error(f"Error in evaluate_formula: {str(e)}")
            raise FormulaWorkflowError(f"Formula evaluation failed: {str(e)}")


class FormulaWorkflowError(Exception):
    """Custom exception for formula workflow errors"""
    pass


class FormulaWorkflow:
    """协调生成器、解析器和评审员之间的交互"""

    def __init__(self, api_key: str, max_iterations: int = 3):
        self.generator = FormulaGenerator(api_key)
        self.parser = FormulaParser(api_key)
        self.critic = FormulaCritic(api_key)
        self.max_iterations = max_iterations
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    async def run(self, task: str) -> str:
        """运行公式生成和评估工作流"""
        iteration = 0
        feedback = None

        while iteration < self.max_iterations:
            logging.info(f"第 {iteration + 1} 次迭代")
            
            try:
                # 生成公式
                generated_text = await self.generator.generate_formula(task, feedback)
                logging.info(f"生成公式：{generated_text}")

                # 解析公式
                parse_result = await self.parser.parse_formula(generated_text)
                if not parse_result["found"]:
                    feedback = "无法提取有效公式。请严格遵循语法规则。"
                    logging.warning(f"解析失败：{feedback}")
                    iteration += 1
                    continue

                # 评估公式
                evaluation = await self.critic.evaluate_formula(parse_result["formula"])
                logging.info(f"评估结果：{evaluation}")

                if evaluation["approved"]:
                    logging.info("公式已通过审核！")
                    return parse_result["formula"]

                feedback = evaluation["feedback"]
                iteration += 1

            except Exception as e:
                logging.error(f"工作流迭代 {iteration + 1} 发生错误: {str(e)}")
                raise FormulaWorkflowError(f"公式生成过程出错: {str(e)}")

        error_msg = "达到最大迭代次数，未能找到满意的公式"
        logging.error(error_msg)
        raise FormulaWorkflowError(error_msg)


async def main():
    api_key = "sk-fastgpt"
    workflow = FormulaWorkflow(api_key)

    try:
        formula = await workflow.run("计算圆的面积")
        print(f"最终公式：{formula}")
    except FormulaWorkflowError as e:
        logging.error(f"工作流错误：{str(e)}")
        # 这里可以添加错误恢复逻辑
    except Exception as e:
        logging.error(f"未预期的错误：{str(e)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
