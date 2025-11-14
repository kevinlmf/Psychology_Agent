"""
Critic Agent - 质量评估和置信度路由
评估回应质量，决定是否需要重新生成或调整策略
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from models import (
    get_orchestrator,
    ModelRouter,
    TaskType,
    SystemPrompts,
)
from agent.mental_state_interpreter import MentalStateVector


@dataclass
class ResponseQualityScore:
    """回应质量评分"""
    
    # 核心评分维度 (0.0-1.0)
    relevance: float = 0.0  # 相关性：回应是否与用户状态和需求相关
    safety: float = 1.0  # 安全性：是否安全、无有害内容
    effectiveness: float = 0.0  # 有效性：是否可能有效改善用户状态
    empathy: float = 0.0  # 同理心：是否温暖、理解用户
    clarity: float = 0.0  # 清晰度：是否清晰易懂
    actionability: float = 0.0  # 可执行性：是否提供可执行的建议
    
    # 总体评分
    overall_score: float = 0.0  # 加权总分
    confidence: float = 0.0  # 置信度：对评估的置信度
    
    # 问题识别
    issues: List[str] = None  # 识别到的问题列表
    strengths: List[str] = None  # 识别到的优点列表
    
    # 建议
    refinement_needed: bool = False  # 是否需要优化
    refinement_suggestions: List[str] = None  # 优化建议
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.strengths is None:
            self.strengths = []
        if self.refinement_suggestions is None:
            self.refinement_suggestions = []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "relevance": self.relevance,
            "safety": self.safety,
            "effectiveness": self.effectiveness,
            "empathy": self.empathy,
            "clarity": self.clarity,
            "actionability": self.actionability,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "issues": self.issues,
            "strengths": self.strengths,
            "refinement_needed": self.refinement_needed,
            "refinement_suggestions": self.refinement_suggestions,
        }


class CriticAgent:
    """
    Critic Agent
    评估回应质量，提供置信度路由决策
    """
    
    # 评分权重配置
    QUALITY_WEIGHTS = {
        "relevance": 0.25,
        "safety": 0.30,  # 安全性最重要
        "effectiveness": 0.20,
        "empathy": 0.10,
        "clarity": 0.10,
        "actionability": 0.05,
    }
    
    # 置信度阈值
    HIGH_CONFIDENCE_THRESHOLD = 0.8  # 高置信度，直接使用
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6  # 中等置信度，轻微调整
    LOW_CONFIDENCE_THRESHOLD = 0.4  # 低置信度，需要重新生成
    
    def __init__(self):
        self.llm = get_orchestrator()
    
    async def evaluate_response(
        self,
        response: str,
        user_message: str,
        mental_state: MentalStateVector,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> ResponseQualityScore:
        """
        评估回应质量
        
        Args:
            response: Agent生成的回应
            user_message: 用户原始消息
            mental_state: 用户当前心理状态
            user_profile: 用户画像
        
        Returns:
            ResponseQualityScore: 质量评分
        """
        # 构建评估提示
        prompt = self._build_evaluation_prompt(
            response=response,
            user_message=user_message,
            mental_state=mental_state,
            user_profile=user_profile,
        )
        
        config = ModelRouter.get_model_config(TaskType.RISK_ASSESSMENT)
        
        # 使用LLM进行结构化评估
        result = await self.llm.generate_structured(
            prompt=prompt,
            config=config,
            system_prompt=self._get_critic_system_prompt(),
        )
        
        # 解析结果
        score = ResponseQualityScore(
            relevance=float(result.get("relevance", 0.5)),
            safety=float(result.get("safety", 1.0)),
            effectiveness=float(result.get("effectiveness", 0.5)),
            empathy=float(result.get("empathy", 0.5)),
            clarity=float(result.get("clarity", 0.5)),
            actionability=float(result.get("actionability", 0.5)),
            confidence=float(result.get("confidence", 0.5)),
            issues=result.get("issues", []),
            strengths=result.get("strengths", []),
            refinement_needed=result.get("refinement_needed", False),
            refinement_suggestions=result.get("refinement_suggestions", []),
        )
        
        # 计算加权总分
        score.overall_score = self._calculate_weighted_score(score)
        
        return score
    
    def _calculate_weighted_score(self, score: ResponseQualityScore) -> float:
        """计算加权总分"""
        weighted_sum = (
            score.relevance * self.QUALITY_WEIGHTS["relevance"] +
            score.safety * self.QUALITY_WEIGHTS["safety"] +
            score.effectiveness * self.QUALITY_WEIGHTS["effectiveness"] +
            score.empathy * self.QUALITY_WEIGHTS["empathy"] +
            score.clarity * self.QUALITY_WEIGHTS["clarity"] +
            score.actionability * self.QUALITY_WEIGHTS["actionability"]
        )
        return weighted_sum
    
    def decide_routing(
        self,
        score: ResponseQualityScore,
    ) -> Dict[str, Any]:
        """
        根据评分决定路由策略
        
        Returns:
            {
                "action": "use" | "refine" | "regenerate",
                "reason": str,
                "priority": "high" | "medium" | "low",
            }
        """
        # 安全性检查：如果安全性低，必须重新生成
        if score.safety < 0.7:
            return {
                "action": "regenerate",
                "reason": "安全性不足，需要重新生成",
                "priority": "high",
            }
        
        # 根据置信度决定
        if score.confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            if score.overall_score >= 0.7:
                return {
                    "action": "use",
                    "reason": "高质量回应，置信度高",
                    "priority": "low",
                }
            else:
                return {
                    "action": "refine",
                    "reason": "置信度高但质量中等，需要优化",
                    "priority": "medium",
                }
        
        elif score.confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return {
                "action": "refine",
                "reason": "中等置信度，建议优化",
                "priority": "medium",
            }
        
        else:
            return {
                "action": "regenerate",
                "reason": "低置信度，需要重新生成",
                "priority": "high",
            }
    
    def _build_evaluation_prompt(
        self,
        response: str,
        user_message: str,
        mental_state: MentalStateVector,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建评估提示"""
        
        state_summary = f"""
情绪状态: {mental_state.mood_label}
焦虑: {mental_state.anxiety:.2f}, 抑郁: {mental_state.depression:.2f}, 压力: {mental_state.stress:.2f}
核心需求: {mental_state.primary_need or '未知'}
认知模式: {', '.join(mental_state.cognitive_patterns) if mental_state.cognitive_patterns else '无'}
"""
        
        profile_text = ""
        if user_profile:
            profile_text = f"\n用户画像: {user_profile.get('summary', '无')}"
        
        prompt = f"""
请作为专业的心理健康回应质量评估专家，评估以下Agent回应的质量。

用户消息: "{user_message}"

用户当前心理状态:
{state_summary}
{profile_text}

Agent回应:
"{response}"

请从以下维度进行评估，输出结构化评分：

1. **相关性 (relevance, 0.0-1.0)**
   - 回应是否与用户的消息和状态相关？
   - 是否回应了用户的核心需求？

2. **安全性 (safety, 0.0-1.0)**
   - 是否安全、无有害内容？
   - 是否避免了可能加重用户症状的建议？
   - 是否在危机情况下提供了适当的资源？

3. **有效性 (effectiveness, 0.0-1.0)**
   - 回应是否可能有效改善用户状态？
   - 是否基于CBT/ACT等科学原理？
   - 是否提供了有用的策略？

4. **同理心 (empathy, 0.0-1.0)**
   - 是否温暖、理解用户？
   - 是否避免了评判性语言？

5. **清晰度 (clarity, 0.0-1.0)**
   - 回应是否清晰易懂？
   - 是否避免了过于专业或模糊的语言？

6. **可执行性 (actionability, 0.0-1.0)**
   - 是否提供了具体可执行的建议？
   - 建议是否现实可行？

请严格按照以下JSON格式输出：

{{
    "relevance": 0.0-1.0,
    "safety": 0.0-1.0,
    "effectiveness": 0.0-1.0,
    "empathy": 0.0-1.0,
    "clarity": 0.0-1.0,
    "actionability": 0.0-1.0,
    "confidence": 0.0-1.0,
    "issues": ["问题1", "问题2"],
    "strengths": ["优点1", "优点2"],
    "refinement_needed": true/false,
    "refinement_suggestions": ["建议1", "建议2"]
}}
"""
        return prompt
    
    def _get_critic_system_prompt(self) -> str:
        """获取Critic Agent的系统提示"""
        return """You are a professional quality assessment expert for mental health AI responses.

Your role is to:
1. Objectively evaluate response quality across multiple dimensions
2. Identify strengths and weaknesses
3. Provide actionable feedback for improvement
4. Ensure safety and effectiveness

Be thorough, fair, and constructive in your evaluation."""


