"""
性格分析器 (Personality Analyzer)
基于对话内容分析用户性格特征，用于个性化RLHF
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from models import (
    get_orchestrator,
    ModelRouter,
    TaskType,
    SystemPrompts,
)
from agent.mental_state_interpreter import MentalStateVector


@dataclass
class PersonalityProfile:
    """用户性格画像"""
    
    # Big Five 人格特质 (0.0-1.0)
    openness: float = 0.5  # 开放性：对新体验的开放程度
    conscientiousness: float = 0.5  # 尽责性：组织性和可靠性
    extraversion: float = 0.5  # 外向性：社交性和活力
    agreeableness: float = 0.5  # 宜人性：合作性和信任度
    neuroticism: float = 0.5  # 神经质：情绪稳定性（越高越不稳定）
    
    # 沟通风格偏好
    communication_style: str = "balanced"  # direct, gentle, balanced, analytical
    response_preference: str = "balanced"  # emotional_support, practical_advice, cognitive_reframing, balanced
    
    # 学习/改变偏好
    learning_style: str = "balanced"  # visual, auditory, kinesthetic, reading, balanced
    change_approach: str = "gradual"  # gradual, immediate, structured, flexible
    
    # 应对机制偏好
    coping_style: str = "balanced"  # problem_focused, emotion_focused, avoidance, balanced
    
    # 元数据
    confidence: float = 0.5  # 性格评估的置信度
    assessment_date: datetime = field(default_factory=datetime.now)
    assessment_method: str = "conversation_analysis"  # conversation_analysis, questionnaire, hybrid
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "big_five": {
                "openness": self.openness,
                "conscientiousness": self.conscientiousness,
                "extraversion": self.extraversion,
                "agreeableness": self.agreeableness,
                "neuroticism": self.neuroticism,
            },
            "communication_style": self.communication_style,
            "response_preference": self.response_preference,
            "learning_style": self.learning_style,
            "change_approach": self.change_approach,
            "coping_style": self.coping_style,
            "confidence": self.confidence,
            "assessment_date": self.assessment_date.isoformat(),
            "assessment_method": self.assessment_method,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityProfile':
        """从字典创建"""
        big_five = data.get("big_five", {})
        return cls(
            openness=big_five.get("openness", 0.5),
            conscientiousness=big_five.get("conscientiousness", 0.5),
            extraversion=big_five.get("extraversion", 0.5),
            agreeableness=big_five.get("agreeableness", 0.5),
            neuroticism=big_five.get("neuroticism", 0.5),
            communication_style=data.get("communication_style", "balanced"),
            response_preference=data.get("response_preference", "balanced"),
            learning_style=data.get("learning_style", "balanced"),
            change_approach=data.get("change_approach", "gradual"),
            coping_style=data.get("coping_style", "balanced"),
            confidence=data.get("confidence", 0.5),
            assessment_date=datetime.fromisoformat(data.get("assessment_date", datetime.now().isoformat())),
            assessment_method=data.get("assessment_method", "conversation_analysis"),
        )
    
    def get_personality_summary(self) -> str:
        """获取性格摘要"""
        traits = []
        if self.openness > 0.6:
            traits.append("高开放性（喜欢新体验）")
        elif self.openness < 0.4:
            traits.append("低开放性（偏好稳定）")
        
        if self.conscientiousness > 0.6:
            traits.append("高尽责性（组织性强）")
        elif self.conscientiousness < 0.4:
            traits.append("低尽责性（灵活随性）")
        
        if self.extraversion > 0.6:
            traits.append("外向型（社交活跃）")
        elif self.extraversion < 0.4:
            traits.append("内向型（偏好独处）")
        
        if self.neuroticism > 0.6:
            traits.append("高神经质（情绪敏感）")
        elif self.neuroticism < 0.4:
            traits.append("低神经质（情绪稳定）")
        
        return ", ".join(traits) if traits else "平衡型"


class PersonalityAnalyzer:
    """
    性格分析器
    基于对话内容分析用户性格特征
    """
    
    def __init__(self):
        self.llm = get_orchestrator()
    
    async def analyze_from_conversation(
        self,
        conversation_history: List[Dict[str, str]],
        mental_states: Optional[List[MentalStateVector]] = None,
    ) -> PersonalityProfile:
        """
        从对话历史分析性格特征
        
        Args:
            conversation_history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            mental_states: 历史心理状态（可选）
        
        Returns:
            PersonalityProfile: 性格画像
        """
        # 构建分析提示
        prompt = self._build_analysis_prompt(conversation_history, mental_states)
        
        config = ModelRouter.get_model_config(TaskType.BEHAVIOR_ANALYSIS)
        
        # 使用LLM进行结构化分析
        result = await self.llm.generate_structured(
            prompt=prompt,
            config=config,
            system_prompt=self._get_personality_system_prompt(),
        )
        
        # 解析结果
        big_five = result.get("big_five", {})
        
        return PersonalityProfile(
            openness=float(big_five.get("openness", 0.5)),
            conscientiousness=float(big_five.get("conscientiousness", 0.5)),
            extraversion=float(big_five.get("extraversion", 0.5)),
            agreeableness=float(big_five.get("agreeableness", 0.5)),
            neuroticism=float(big_five.get("neuroticism", 0.5)),
            communication_style=result.get("communication_style", "balanced"),
            response_preference=result.get("response_preference", "balanced"),
            learning_style=result.get("learning_style", "balanced"),
            change_approach=result.get("change_approach", "gradual"),
            coping_style=result.get("coping_style", "balanced"),
            confidence=float(result.get("confidence", 0.5)),
            assessment_method="conversation_analysis",
        )
    
    async def quick_assessment(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> PersonalityProfile:
        """
        快速性格评估（基于单条消息或简短对话）
        
        Args:
            user_message: 用户消息
            conversation_history: 对话历史（可选）
        
        Returns:
            PersonalityProfile: 初步性格画像
        """
        history = conversation_history or []
        history.append({"role": "user", "content": user_message})
        
        return await self.analyze_from_conversation(history)
    
    def _build_analysis_prompt(
        self,
        conversation_history: List[Dict[str, str]],
        mental_states: Optional[List[MentalStateVector]] = None,
    ) -> str:
        """构建性格分析提示"""
        
        # 格式化对话历史
        history_text = "\n".join([
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in conversation_history[-10:]  # 最近10轮对话
        ])
        
        # 格式化心理状态历史
        states_text = ""
        if mental_states:
            states_summary = []
            for state in mental_states[-5:]:  # 最近5个状态
                states_summary.append(
                    f"情绪: {state.mood_label}, "
                    f"焦虑: {state.anxiety:.2f}, "
                    f"压力: {state.stress:.2f}, "
                    f"认知模式: {', '.join(state.cognitive_patterns) if state.cognitive_patterns else '无'}"
                )
            states_text = "\n".join(states_summary)
        
        # 构建心理状态部分
        states_section = ""
        if states_text:
            states_section = f"心理状态历史:\n{states_text}\n\n"
        
        prompt = f"""
请作为专业的性格分析专家，基于以下对话内容分析用户的性格特征。

对话历史:
{history_text}

{states_section}请从以下维度进行分析：

1. **Big Five 人格特质** (0.0-1.0)
   - openness (开放性): 对新体验、新想法的开放程度
   - conscientiousness (尽责性): 组织性、可靠性、自律性
   - extraversion (外向性): 社交性、活力、外向程度
   - agreeableness (宜人性): 合作性、信任度、同理心
   - neuroticism (神经质): 情绪不稳定性（越高越不稳定）

2. **沟通风格偏好**
   - direct: 直接、明确
   - gentle: 温和、耐心
   - balanced: 平衡
   - analytical: 分析性、逻辑性

3. **回应偏好**
   - emotional_support: 情感支持
   - practical_advice: 实用建议
   - cognitive_reframing: 认知重构
   - balanced: 平衡

4. **学习风格**
   - visual: 视觉学习
   - auditory: 听觉学习
   - kinesthetic: 动觉学习
   - reading: 阅读学习
   - balanced: 平衡

5. **改变方式偏好**
   - gradual: 渐进式
   - immediate: 立即行动
   - structured: 结构化
   - flexible: 灵活

6. **应对机制偏好**
   - problem_focused: 问题导向
   - emotion_focused: 情绪导向
   - avoidance: 回避型
   - balanced: 平衡

请严格按照以下JSON格式输出：

{{
    "big_five": {{
        "openness": 0.0-1.0,
        "conscientiousness": 0.0-1.0,
        "extraversion": 0.0-1.0,
        "agreeableness": 0.0-1.0,
        "neuroticism": 0.0-1.0
    }},
    "communication_style": "direct/gentle/balanced/analytical",
    "response_preference": "emotional_support/practical_advice/cognitive_reframing/balanced",
    "learning_style": "visual/auditory/kinesthetic/reading/balanced",
    "change_approach": "gradual/immediate/structured/flexible",
    "coping_style": "problem_focused/emotion_focused/avoidance/balanced",
    "confidence": 0.0-1.0
}}
"""
        return prompt
    
    def _get_personality_system_prompt(self) -> str:
        """获取性格分析的系统提示"""
        return """You are a professional personality psychologist specializing in Big Five personality traits and communication styles.

Your role is to:
1. Analyze user personality traits from conversation patterns
2. Identify communication preferences and learning styles
3. Provide accurate, evidence-based personality assessments
4. Be objective and avoid stereotypes

Focus on observable patterns in language, emotional expression, and interaction style."""

