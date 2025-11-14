"""
心理状态解释器 (Mental State Interpreter)
LLM层：理解用户当前心理状态，输出结构化心理状态向量
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from models import (
    get_orchestrator,
    ModelRouter,
    TaskType,
    SystemPrompts,
)


@dataclass
class MentalStateVector:
    """心理状态向量 - RL的状态空间"""
    
    # 核心情绪指标 (0.0-1.0)
    anxiety: float = 0.0
    depression: float = 0.0
    stress: float = 0.0
    motivation: float = 0.5
    self_efficacy: float = 0.5  # 自我效能感
    
    # 情绪标签
    mood_label: str = "neutral"  # anxious, depressed, stressed, stable, positive
    
    # 需求识别
    needs: List[str] = field(default_factory=list)  # reassurance, structure, direction, validation, etc.
    
    # 认知模式
    cognitive_patterns: List[str] = field(default_factory=list)  # overthinking, catastrophizing, perfectionism, etc.
    
    # 压力源
    stressors: List[str] = field(default_factory=list)  # work, relationships, health, etc.
    
    # 身体信号
    physical_signals: List[str] = field(default_factory=list)  # sleep_issues, fatigue, etc.
    
    # 当前需求类型
    primary_need: Optional[str] = None  # comfort, structure, validation, problem_solving
    
    # 元数据
    confidence: float = 0.5  # 状态评估的置信度
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "anxiety": self.anxiety,
            "depression": self.depression,
            "stress": self.stress,
            "motivation": self.motivation,
            "self_efficacy": self.self_efficacy,
            "mood_label": self.mood_label,
            "needs": self.needs,
            "cognitive_patterns": self.cognitive_patterns,
            "stressors": self.stressors,
            "physical_signals": self.physical_signals,
            "primary_need": self.primary_need,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class MentalStateInterpreter:
    """
    心理状态解释器
    使用LLM理解用户输入，输出结构化心理状态向量
    """
    
    def __init__(self):
        self.llm = get_orchestrator()
    
    async def interpret(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> MentalStateVector:
        """
        解释用户当前心理状态
        
        Args:
            user_message: 用户当前消息
            conversation_history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            user_profile: 用户画像信息
        
        Returns:
            MentalStateVector: 结构化心理状态向量
        """
        # 构建上下文
        context_parts = []
        
        if user_profile:
            context_parts.append(f"用户画像: {user_profile}")
        
        if conversation_history:
            history_text = "\n".join([
                f"{turn.get('role', 'user')}: {turn.get('content', '')}"
                for turn in conversation_history[-5:]  # 最近5轮对话
            ])
            context_parts.append(f"最近对话历史:\n{history_text}")
        
        context_text = "\n\n".join(context_parts) if context_parts else "无历史上下文"
        
        # 构建分析提示
        prompt = f"""
请作为专业的心理状态评估专家，分析用户的当前心理状态。

用户当前消息: "{user_message}"

{context_text}

请从以下维度进行深度分析，输出结构化的心理状态向量：

1. **核心情绪指标** (0.0-1.0，数值越高表示程度越高)
   - anxiety: 焦虑程度
   - depression: 抑郁程度
   - stress: 压力程度
   - motivation: 动机水平
   - self_efficacy: 自我效能感（"我可以做到"的信念强度）

2. **情绪标签**
   - mood_label: 主要情绪状态 (anxious, depressed, stressed, stable, positive, mixed_anxiety等)

3. **需求识别**
   - needs: 用户当前的心理需求列表，可能包括：
     * reassurance: 需要安慰和安全感
     * structure: 需要结构和方向
     * validation: 需要被理解和认可
     * direction: 需要具体指导
     * comfort: 需要情感支持
     * problem_solving: 需要解决问题的方法
   - primary_need: 最核心的需求（从needs中选择一个）

4. **认知模式**
   - cognitive_patterns: 识别到的认知偏差或思维模式，例如：
     * overthinking: 过度思考
     * catastrophizing: 灾难化思维
     * perfectionism: 完美主义
     * black_white_thinking: 非黑即白思维
     * personalization: 个人化归因

5. **压力源**
   - stressors: 识别到的压力来源（work, relationships, health, financial, academic等）

6. **身体信号**
   - physical_signals: 提到的身体症状（sleep_issues, fatigue, headache, palpitations等）

7. **评估置信度**
   - confidence: 对此次评估的置信度 (0.0-1.0)

请严格按照以下JSON格式输出，不要添加任何其他文字：

{{
    "anxiety": 0.0-1.0,
    "depression": 0.0-1.0,
    "stress": 0.0-1.0,
    "motivation": 0.0-1.0,
    "self_efficacy": 0.0-1.0,
    "mood_label": "情绪标签",
    "needs": ["需求1", "需求2"],
    "cognitive_patterns": ["模式1", "模式2"],
    "stressors": ["压力源1", "压力源2"],
    "physical_signals": ["信号1", "信号2"],
    "primary_need": "核心需求",
    "confidence": 0.0-1.0
}}
"""
        
        config = ModelRouter.get_model_config(TaskType.BEHAVIOR_ANALYSIS)
        
        result = await self.llm.generate_structured(
            prompt=prompt,
            config=config,
            system_prompt=SystemPrompts.BEHAVIOR_ANALYST,
        )
        
        # 解析结果并创建MentalStateVector
        return MentalStateVector(
            anxiety=float(result.get("anxiety", 0.0)),
            depression=float(result.get("depression", 0.0)),
            stress=float(result.get("stress", 0.0)),
            motivation=float(result.get("motivation", 0.5)),
            self_efficacy=float(result.get("self_efficacy", 0.5)),
            mood_label=result.get("mood_label", "neutral"),
            needs=result.get("needs", []),
            cognitive_patterns=result.get("cognitive_patterns", []),
            stressors=result.get("stressors", []),
            physical_signals=result.get("physical_signals", []),
            primary_need=result.get("primary_need"),
            confidence=float(result.get("confidence", 0.5)),
        )
    
    def get_state_summary(self, state: MentalStateVector) -> str:
        """获取状态摘要（用于日志和调试）"""
        summary_parts = [
            f"情绪状态: {state.mood_label}",
            f"焦虑: {state.anxiety:.2f}, 压力: {state.stress:.2f}, 动机: {state.motivation:.2f}",
        ]
        
        if state.needs:
            summary_parts.append(f"核心需求: {state.primary_need or state.needs[0]}")
        
        if state.cognitive_patterns:
            summary_parts.append(f"认知模式: {', '.join(state.cognitive_patterns)}")
        
        return " | ".join(summary_parts)

