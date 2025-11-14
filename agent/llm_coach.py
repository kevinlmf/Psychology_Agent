"""
LLM-Coach (认知行为调节器)
根据心理状态和奖励函数，生成CBT/ACT风格的认知行为支持回应
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from models import (
    get_orchestrator,
    ModelRouter,
    TaskType,
    SystemPrompts,
)
from agent.mental_state_interpreter import MentalStateVector
from rlhf.personalized_reward_model import PersonalizedRewardWeights


class LLMCoach:
    """
    LLM-Coach
    核心任务：
    1. 解释情绪
    2. 做轻量CBT/ACT式认知调节
    3. 给出micro action
    4. 给出grounding技巧
    5. 强化自我效能
    6. 帮用户重建安全感
    7. 提供结构与方向（not just empathy）
    """
    
    def __init__(self):
        self.llm = get_orchestrator()
    
    async def generate_coaching_response(
        self,
        user_message: str,
        mental_state: MentalStateVector,
        reward_weights: PersonalizedRewardWeights,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        生成认知行为支持回应
        
        Args:
            user_message: 用户消息
            mental_state: 当前心理状态向量
            reward_weights: 个性化奖励权重（用于指导回应风格）
            conversation_history: 对话历史
            user_profile: 用户画像
        
        Returns:
            教练式回应文本
        """
        # 构建上下文
        context_parts = []
        
        if user_profile:
            context_parts.append(f"用户画像: {user_profile}")
        
        if conversation_history:
            history_text = "\n".join([
                f"{turn.get('role', 'user')}: {turn.get('content', '')}"
                for turn in conversation_history[-3:]
            ])
            context_parts.append(f"最近对话:\n{history_text}")
        
        # 构建心理状态摘要
        state_summary = self._format_mental_state(mental_state)
        
        # 构建奖励权重指导（告诉模型应该强调哪些方面）
        reward_guidance = self._format_reward_guidance(reward_weights)
        
        # 构建提示
        prompt = f"""
你是一位专业的心理健康教练，使用CBT（认知行为疗法）和ACT（接纳承诺疗法）的原则。

用户当前消息: "{user_message}"

用户当前心理状态:
{state_summary}

个性化回应指导（根据用户历史反馈优化）:
{reward_guidance}

{chr(10).join(context_parts) if context_parts else ""}

请生成一个温暖、专业、有效的回应，遵循以下原则：

**核心任务：**

1. **解释情绪** (Emotion Explanation)
   - 帮助用户理解他们的情绪是正常的、有意义的
   - 解释情绪背后的功能（例如：焦虑是大脑的保护机制）
   - 不要只是说"我理解你"，而是真正解释情绪的本质

2. **认知调节** (Cognitive Reframing)
   - 如果识别到认知偏差（如灾难化思维、过度思考），温和地引导用户意识到
   - 使用CBT技术：帮助用户区分"想法"和"事实"
   - 使用ACT技术：帮助用户接纳情绪，而不是对抗情绪

3. **提供Micro Action** (具体可执行的小步骤)
   - 给出1-2个非常简单、立刻可以做的行动
   - 例如：呼吸练习、写下三个想法、5分钟散步
   - 确保行动具体、可测量、可达成

4. **Grounding技巧** (接地技巧)
   - 如果用户焦虑或压力高，提供grounding技巧
   - 例如：5-4-3-2-1感官练习、呼吸练习、身体扫描

5. **强化自我效能** (Self-Efficacy Boost)
   - 帮助用户看到他们已经做到的、有能力做到的
   - 使用具体的例子和证据
   - 避免空洞的鼓励，而是基于事实的肯定

6. **重建安全感** (Safety Rebuilding)
   - 如果用户感到不安全或失控，帮助重建安全感
   - 提供结构和方向，而不是只给同理心
   - 帮助用户看到：即使现在困难，也有路径可以走

7. **提供结构与方向** (Structure & Direction)
   - 不只是共情，还要给出清晰的下一步
   - 帮助用户看到：问题是可以解决的，有方法可以尝试

**回应风格要求：**

- 温暖但不过度：专业、有边界
- 具体而非抽象：给出具体的方法和例子
- 平衡同理心和行动：既要理解，也要引导
- 符合用户偏好：根据reward_weights调整风格
  * 如果compassion权重高 → 更温柔、更多理解
  * 如果cognitive_clarity权重高 → 更多认知调节
  * 如果self_efficacy权重高 → 更多能力强化

**回应格式：**

1. 开头：简短的情绪确认和理解（1-2句）
2. 主体：认知调节 + 具体方法（2-3段）
3. 结尾：Micro Action + 鼓励（1段）

请生成回应（3-5段，自然流畅）：
"""
        
        config = ModelRouter.get_model_config(TaskType.INTERVENTION_PLANNING)
        
        response = await self.llm.generate(
            prompt=prompt,
            config=config,
            system_prompt=SystemPrompts.THERAPIST_BASE,
        )
        
        return response
    
    def _format_mental_state(self, state: MentalStateVector) -> str:
        """格式化心理状态为文本"""
        parts = [
            f"- 情绪标签: {state.mood_label}",
            f"- 焦虑: {state.anxiety:.2f}, 抑郁: {state.depression:.2f}, 压力: {state.stress:.2f}",
            f"- 动机: {state.motivation:.2f}, 自我效能: {state.self_efficacy:.2f}",
        ]
        
        if state.needs:
            parts.append(f"- 核心需求: {state.primary_need or state.needs[0]}")
            parts.append(f"- 其他需求: {', '.join(state.needs)}")
        
        if state.cognitive_patterns:
            parts.append(f"- 认知模式: {', '.join(state.cognitive_patterns)}")
        
        if state.stressors:
            parts.append(f"- 压力源: {', '.join(state.stressors)}")
        
        if state.physical_signals:
            parts.append(f"- 身体信号: {', '.join(state.physical_signals)}")
        
        return "\n".join(parts)
    
    def _format_reward_guidance(self, weights: PersonalizedRewardWeights) -> str:
        """格式化奖励权重为指导文本"""
        guidance_parts = []
        
        # 找出权重最高的几个维度
        weight_dict = weights.to_dict()
        sorted_weights = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)
        
        top_weights = sorted_weights[:3]  # 前3个最高权重
        
        guidance_parts.append("根据用户历史反馈，以下方面对用户最有效：")
        
        for dimension, weight in top_weights:
            if weight > 0.15:  # 如果权重显著高于平均
                dimension_name = {
                    "emotional_stability": "情绪稳定性支持",
                    "stress_reduction": "压力缓解技巧",
                    "self_efficacy": "自我效能感强化",
                    "cognitive_clarity": "认知清晰度引导",
                    "behavioral_consistency": "行为一致性鼓励",
                    "compassion": "同理心和情感支持",
                }.get(dimension, dimension)
                
                guidance_parts.append(f"- {dimension_name} (权重: {weight:.2f})")
        
        return "\n".join(guidance_parts)
    
    async def generate_crisis_response(
        self,
        user_message: str,
        mental_state: MentalStateVector,
    ) -> str:
        """
        生成危机干预回应（当检测到高风险时）
        """
        prompt = f"""
用户当前处于高风险状态。

用户消息: "{user_message}"

心理状态:
- 情绪: {mental_state.mood_label}
- 焦虑: {mental_state.anxiety:.2f}, 抑郁: {mental_state.depression:.2f}
- 压力源: {', '.join(mental_state.stressors) if mental_state.stressors else '未知'}

请生成紧急危机干预回应：

1. 立即表达关心和支持
2. 确认用户当前安全状况
3. 提供紧急资源（心理危机热线、急诊等）
4. 询问是否有支持系统（家人、朋友）
5. 温和但坚定地建议寻求专业帮助

语气：温暖、坚定、非评判性
长度：2-3段
"""
        
        config = ModelRouter.get_model_config(TaskType.CRISIS_DETECTION)
        
        response = await self.llm.generate(
            prompt=prompt,
            config=config,
            system_prompt=SystemPrompts.CRISIS_DETECTOR,
        )
        
        # 添加紧急资源信息
        response += "\n\n---\n🆘 紧急资源:\n"
        response += "24小时心理危机热线: 400-161-9995\n"
        response += "如有立即危险，请拨打120或前往最近急诊科"
        
        return response

