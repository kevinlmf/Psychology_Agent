"""
Refiner Agent - 迭代优化回应生成
根据Critic反馈，迭代优化或生成多个候选回应
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
from agent.critic_agent import ResponseQualityScore
from rlhf.personalized_reward_model import PersonalizedRewardWeights


@dataclass
class RefinedResponse:
    """优化后的回应"""
    response: str
    quality_score: ResponseQualityScore
    refinement_iterations: int = 0
    strategy_used: str = ""  # CBT, ACT, problem-solving, etc.
    tone: str = ""  # gentle, supportive, direct, etc.


class RefinerAgent:
    """
    Refiner Agent
    迭代优化回应，生成多个候选并选择最佳
    """
    
    MAX_REFINEMENT_ITERATIONS = 3  # 最大优化迭代次数
    
    def __init__(self):
        self.llm = get_orchestrator()
    
    async def refine_response(
        self,
        original_response: str,
        user_message: str,
        mental_state: MentalStateVector,
        critic_feedback: ResponseQualityScore,
        reward_weights: PersonalizedRewardWeights,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> RefinedResponse:
        """
        根据Critic反馈优化回应
        
        Args:
            original_response: 原始回应
            critic_feedback: Critic的质量评分
            mental_state: 用户心理状态
            reward_weights: 个性化奖励权重
            conversation_history: 对话历史
            user_profile: 用户画像
        
        Returns:
            RefinedResponse: 优化后的回应
        """
        # 构建优化提示
        prompt = self._build_refinement_prompt(
            original_response=original_response,
            user_message=user_message,
            mental_state=mental_state,
            critic_feedback=critic_feedback,
            reward_weights=reward_weights,
            conversation_history=conversation_history,
            user_profile=user_profile,
        )
        
        config = ModelRouter.get_model_config(TaskType.INTERVENTION_PLANNING)
        
        # 生成优化后的回应
        refined_response = await self.llm.generate(
            prompt=prompt,
            config=config,
            system_prompt=SystemPrompts.THERAPIST_BASE,
        )
        
        return RefinedResponse(
            response=refined_response,
            quality_score=critic_feedback,
            refinement_iterations=1,
            strategy_used=self._infer_strategy(mental_state, reward_weights),
            tone=self._infer_tone(reward_weights),
        )
    
    async def generate_multiple_candidates(
        self,
        user_message: str,
        mental_state: MentalStateVector,
        reward_weights: PersonalizedRewardWeights,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        num_candidates: int = 3,
    ) -> List[RefinedResponse]:
        """
        生成多个候选回应（不同策略和风格）
        
        Args:
            user_message: 用户消息
            mental_state: 心理状态
            reward_weights: 奖励权重
            num_candidates: 生成候选数量
        
        Returns:
            List[RefinedResponse]: 候选回应列表
        """
        candidates = []
        
        # 定义不同的策略组合
        strategies = [
            {"strategy": "CBT", "tone": "gentle", "focus": "cognitive_reframing"},
            {"strategy": "ACT", "tone": "supportive", "focus": "acceptance"},
            {"strategy": "problem-solving", "tone": "direct", "focus": "actionable_steps"},
        ]
        
        for i, strategy_config in enumerate(strategies[:num_candidates]):
            prompt = self._build_candidate_prompt(
                user_message=user_message,
                mental_state=mental_state,
                reward_weights=reward_weights,
                strategy_config=strategy_config,
                conversation_history=conversation_history,
                user_profile=user_profile,
            )
            
            config = ModelRouter.get_model_config(TaskType.INTERVENTION_PLANNING)
            
            response_text = await self.llm.generate(
                prompt=prompt,
                config=config,
                system_prompt=SystemPrompts.THERAPIST_BASE,
            )
            
            candidates.append(RefinedResponse(
                response=response_text,
                quality_score=None,  # 需要Critic评估
                refinement_iterations=0,
                strategy_used=strategy_config["strategy"],
                tone=strategy_config["tone"],
            ))
        
        return candidates
    
    def _build_refinement_prompt(
        self,
        original_response: str,
        user_message: str,
        mental_state: MentalStateVector,
        critic_feedback: ResponseQualityScore,
        reward_weights: PersonalizedRewardWeights,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建优化提示"""
        
        state_summary = f"""
情绪状态: {mental_state.mood_label}
核心需求: {mental_state.primary_need or '未知'}
认知模式: {', '.join(mental_state.cognitive_patterns) if mental_state.cognitive_patterns else '无'}
"""
        
        feedback_summary = f"""
质量评分:
- 相关性: {critic_feedback.relevance:.2f}
- 安全性: {critic_feedback.safety:.2f}
- 有效性: {critic_feedback.effectiveness:.2f}
- 同理心: {critic_feedback.empathy:.2f}
- 清晰度: {critic_feedback.clarity:.2f}
- 可执行性: {critic_feedback.actionability:.2f}

识别的问题: {', '.join(critic_feedback.issues) if critic_feedback.issues else '无'}
优点: {', '.join(critic_feedback.strengths) if critic_feedback.strengths else '无'}
优化建议: {', '.join(critic_feedback.refinement_suggestions) if critic_feedback.refinement_suggestions else '无'}
"""
        
        prompt = f"""
请优化以下心理健康回应。

用户消息: "{user_message}"

用户当前心理状态:
{state_summary}

原始回应:
"{original_response}"

质量评估反馈:
{feedback_summary}

请根据反馈优化回应，确保：
1. 解决识别到的问题
2. 保持原有的优点
3. 遵循优化建议
4. 提升整体质量

优化后的回应应该：
- 更相关、更有效
- 更温暖、更有同理心
- 更清晰、更可执行
- 保持安全性

请生成优化后的回应（3-5段，自然流畅）：
"""
        return prompt
    
    def _build_candidate_prompt(
        self,
        user_message: str,
        mental_state: MentalStateVector,
        reward_weights: PersonalizedRewardWeights,
        strategy_config: Dict[str, str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建候选回应生成提示"""
        
        state_summary = f"""
情绪状态: {mental_state.mood_label}
核心需求: {mental_state.primary_need or '未知'}
"""
        
        strategy_guidance = {
            "CBT": "使用认知行为疗法：识别认知偏差，帮助区分想法和事实",
            "ACT": "使用接纳承诺疗法：帮助接纳情绪，关注价值观驱动的行动",
            "problem-solving": "使用问题解决疗法：提供具体、可执行的步骤",
        }.get(strategy_config["strategy"], "")
        
        tone_guidance = {
            "gentle": "语气温和、耐心",
            "supportive": "语气支持、鼓励",
            "direct": "语气直接、清晰",
        }.get(strategy_config["tone"], "")
        
        prompt = f"""
请生成一个心理健康回应。

用户消息: "{user_message}"

用户当前心理状态:
{state_summary}

策略: {strategy_config["strategy"]} - {strategy_guidance}
语气: {tone_guidance}
重点: {strategy_config["focus"]}

请生成回应（3-5段，自然流畅）：
"""
        return prompt
    
    def _infer_strategy(
        self,
        mental_state: MentalStateVector,
        reward_weights: PersonalizedRewardWeights,
    ) -> str:
        """推断应该使用的策略"""
        # 根据心理状态和权重选择策略
        if mental_state.anxiety > 0.7:
            return "CBT"  # 高焦虑用CBT认知调节
        elif mental_state.motivation < 0.3:
            return "ACT"  # 低动机用ACT接纳
        else:
            return "problem-solving"  # 其他情况用问题解决
    
    def _infer_tone(self, reward_weights: PersonalizedRewardWeights) -> str:
        """推断应该使用的语气"""
        weights_dict = reward_weights.to_dict()
        if weights_dict.get("compassion", 0) > 0.2:
            return "gentle"
        elif weights_dict.get("cognitive_clarity", 0) > 0.2:
            return "direct"
        else:
            return "supportive"


