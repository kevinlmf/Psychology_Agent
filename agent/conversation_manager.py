"""
Conversation Manager (重构版)
整合新流程：User Input → State Interpreter → Reward Model → LLM-Coach → Response → Feedback → Update Weights
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from models import (
    get_orchestrator,
    ModelRouter,
    TaskType,
    SystemPrompts,
)
from agent.memory_system import (
    get_memory_system,
    Session,
    ConversationTurn,
)
from agent.mental_state_interpreter import MentalStateInterpreter, MentalStateVector
from agent.llm_coach import LLMCoach
from agent.critic_agent import CriticAgent, ResponseQualityScore
from agent.refiner_agent import RefinerAgent, RefinedResponse
from agent.personality_analyzer import PersonalityAnalyzer, PersonalityProfile
from rlhf.personalized_reward_model import (
    get_personalized_reward_model,
    PersonalizedMentalRewardModel,
    MentalRewardFeedback,
)


class ConversationState:
    """Conversation state"""

    def __init__(self):
        self.current_topic: Optional[str] = None
        self.detected_emotions: list = []
        self.risk_flags: list = []
        self.intervention_mode: Optional[str] = None  # 'crisis', 'assessment', 'therapy', 'casual'
        self.current_mental_state: Optional[MentalStateVector] = None  # 当前心理状态
        self.previous_mental_state: Optional[MentalStateVector] = None  # 上一轮心理状态


class ConversationManager:
    """
    Conversation Manager
    Handles overall Conversation Flow orchestration
    """

    def __init__(self, user_id: str, session_id: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        self.state = ConversationState()

        # Initialize dependencies
        self.llm = get_orchestrator()
        self.memory = get_memory_system()
        
        # 新架构的核心组件
        self.state_interpreter = MentalStateInterpreter()
        self.llm_coach = LLMCoach()
        self.reward_model = get_personalized_reward_model()
        
        # 增强架构组件（Phase 1）
        self.critic = CriticAgent()
        self.refiner = RefinerAgent()
        
        # 性格分析器（用于个性化RLHF）
        self.personality_analyzer = PersonalityAnalyzer()

        # Create or load conversation
        self.session = Session(
            session_id=self.session_id,
            user_id=user_id,
            start_time=datetime.now(),
        )

        # Load User profile
        self.user_profile = self.memory.get_or_create_profile(user_id)

    async def process_message(self, user_message: str) -> str:
        """
        处理用户消息 - 新架构流程：
        User Input → State Interpreter → Reward Model → LLM-Coach → Response
        
        Args:
            user_message: user input

        Returns:
            agent response
        """
        # 1. 快速危机检测（最高优先级）
        risk_assessment = await self._quick_risk_check(user_message)

        if risk_assessment['risk_level'] == 'high':
            # 危机模式：立即响应
            response = await self._handle_crisis(user_message, risk_assessment)
            self.state.intervention_mode = 'crisis'
        else:
            # 2. 正常流程：新架构
            try:
                response = await self._process_with_new_architecture(user_message)
            except RuntimeError as e:
                # API错误（余额不足、密钥无效等）
                error_msg = str(e)
                if "余额不足" in error_msg or "insufficient" in error_msg.lower():
                    return (
                        "抱歉，系统暂时无法处理您的请求。\n\n"
                        "API余额不足。请前往以下链接充值：\n"
                        "https://console.anthropic.com/settings/billing\n\n"
                        "Sorry, the system is temporarily unable to process your request.\n"
                        "API credits insufficient. Please add credits at the link above."
                    )
                elif "API密钥" in error_msg or "API key" in error_msg.lower():
                    return (
                        "抱歉，系统配置错误。\n\n"
                        "API密钥无效或未设置。请检查 .env 文件中的 ANTHROPIC_API_KEY。\n\n"
                        "Sorry, system configuration error.\n"
                        "Invalid or missing API key. Please check ANTHROPIC_API_KEY in .env file."
                    )
                else:
                    raise

        # 3. 记录对话
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            agent_response=response,
            risk_level=risk_assessment.get('risk_level'),
        )
        self.session.turns.append(turn)

        # 4. 保存会话（定期）
        if len(self.session.turns) % 5 == 0:
            self.memory.save_session(self.session)

        return response
    
    async def _process_with_new_architecture(self, user_message: str) -> str:
        """
        增强架构处理流程：
        1. 心理状态解释器 → State_t
        2. RLHF心理奖励模型 → 获取个性化权重
        3. LLM-Coach → 生成初始回应
        4. Critic Agent → 评估回应质量
        5. Refiner Agent → 根据评估优化回应（如需要）
        """
        # 准备对话历史
        conversation_history = self._get_conversation_history_for_llm()
        
        # 准备用户画像
        user_profile_dict = {
            "user_id": self.user_id,
            "summary": self.memory.get_user_summary(self.user_id),
            "recent_context": self.memory.get_recent_context(self.user_id, days=7),
        }
        
        # Step 1: 心理状态解释器 → State_t
        mental_state = await self.state_interpreter.interpret(
            user_message=user_message,
            conversation_history=conversation_history,
            user_profile=user_profile_dict,
        )
        
        # 保存状态（用于后续反馈）
        self.state.previous_mental_state = self.state.current_mental_state
        self.state.current_mental_state = mental_state
        
        # Step 2: 获取或分析性格特征（用于个性化RLHF）
        personality_profile = await self._get_or_analyze_personality(
            user_message=user_message,
            conversation_history=conversation_history,
        )
        
        # Step 3: RLHF心理奖励模型 → 获取个性化权重（基于性格特征）
        reward_weights = self.reward_model.get_user_weights(
            self.user_id,
            personality_profile=personality_profile,
        )
        
        # Step 4: LLM-Coach → 生成初始回应
        initial_response = await self.llm_coach.generate_coaching_response(
            user_message=user_message,
            mental_state=mental_state,
            reward_weights=reward_weights,
            conversation_history=conversation_history,
            user_profile=user_profile_dict,
        )
        
        # Step 5: Critic Agent → 评估回应质量
        quality_score = await self.critic.evaluate_response(
            response=initial_response,
            user_message=user_message,
            mental_state=mental_state,
            user_profile=user_profile_dict,
        )
        
        # Step 6: 根据评估结果决定路由
        routing_decision = self.critic.decide_routing(quality_score)
        
        if routing_decision["action"] == "use":
            # 高质量回应，直接使用
            final_response = initial_response
        elif routing_decision["action"] == "refine":
            # 需要优化
            refined = await self.refiner.refine_response(
                original_response=initial_response,
                user_message=user_message,
                mental_state=mental_state,
                critic_feedback=quality_score,
                reward_weights=reward_weights,
                conversation_history=conversation_history,
                user_profile=user_profile_dict,
            )
            final_response = refined.response
        else:  # regenerate
            # 低质量或低置信度，重新生成
            # 可以尝试生成多个候选，选择最佳
            candidates = await self.refiner.generate_multiple_candidates(
                user_message=user_message,
                mental_state=mental_state,
                reward_weights=reward_weights,
                conversation_history=conversation_history,
                user_profile=user_profile_dict,
                num_candidates=2,
            )
            
            # 评估候选，选择最佳
            best_candidate = None
            best_score = 0.0
            for candidate in candidates:
                candidate_score_obj = await self.critic.evaluate_response(
                    response=candidate.response,
                    user_message=user_message,
                    mental_state=mental_state,
                    user_profile=user_profile_dict,
                )
                if candidate_score_obj.overall_score > best_score:
                    best_score = candidate_score_obj.overall_score
                    best_candidate = candidate.response
            
            final_response = best_candidate or initial_response  # 如果都失败，使用原始回应
        
        return final_response
    
    async def _get_or_analyze_personality(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
    ) -> Optional[PersonalityProfile]:
        """
        获取或分析用户性格特征
        
        Args:
            user_message: 当前用户消息
            conversation_history: 对话历史
        
        Returns:
            PersonalityProfile: 性格画像（如果已存在或分析成功）
        """
        # 检查用户画像中是否已有性格分析
        if self.user_profile.personality_profile:
            return self.user_profile.personality_profile
        
        # 如果没有，进行性格分析
        # 需要至少3-5轮对话才能进行初步分析
        if len(conversation_history) >= 3:
            try:
                # 收集历史心理状态
                mental_states = []
                if self.state.current_mental_state:
                    mental_states.append(self.state.current_mental_state)
                if self.state.previous_mental_state:
                    mental_states.append(self.state.previous_mental_state)
                
                # 分析性格
                personality = await self.personality_analyzer.analyze_from_conversation(
                    conversation_history=conversation_history,
                    mental_states=mental_states if mental_states else None,
                )
                
                # 保存到用户画像
                self.user_profile.personality_profile = personality
                self.memory.save_profile(self.user_profile)
                
                return personality
            except Exception as e:
                print(f"[Warning] Personality analysis failed: {e}")
                return None
        
        # 对话轮数不足，返回None（使用默认权重）
        return None
    
    def _get_conversation_history_for_llm(self) -> List[Dict[str, str]]:
        """获取对话历史（LLM格式）"""
        recent_turns = self.session.turns[-5:]  # 最近5轮
        history = []
        for turn in recent_turns:
            history.append({"role": "user", "content": turn.user_message})
            history.append({"role": "assistant", "content": turn.agent_response})
        return history

    async def _quick_risk_check(self, message: str) -> Dict[str, Any]:
        """
        Quick Risk detection using Keywords + LLM dual detection
        """
        # First layer: Keyword Rapid screening
        high_risk_keywords = [
            'suicide', '死', 'end生命', 'not想活', '轻生', '伤害自己',
            'self-harm', '割腕', '跳楼',
        ]

        has_risk_keyword = any(kw in message for kw in high_risk_keywords)

        if has_risk_keyword:
            # Second layer: LLM precise determination
            from safety.crisis_detection import CrisisDetector
            detector = CrisisDetector()
            return await detector.assess_risk(message, self.user_id)

        # No obvious Risk
        return {'risk_level': 'low', 'signals': []}

    async def _handle_crisis(self, message: str, risk_info: Dict[str, Any]) -> str:
        """
        危机处理（使用LLM-Coach的危机响应）
        """
        # 先解释心理状态（即使危机情况也需要理解状态）
        mental_state = await self.state_interpreter.interpret(
            user_message=message,
            conversation_history=self._get_conversation_history_for_llm(),
        )
        
        # 使用LLM-Coach生成危机响应
        response = await self.llm_coach.generate_crisis_response(
            user_message=message,
            mental_state=mental_state,
        )
        
        return response

    def collect_feedback(
        self,
        explicit_rating: Optional[int] = None,
        feedback_text: Optional[str] = None,
        continued_conversation: bool = True,
    ):
        """
        收集用户反馈并更新RLHF权重
        
        Args:
            explicit_rating: 显式评分（1-5）
            feedback_text: 反馈文本
            continued_conversation: 是否继续对话
        """
        if not self.state.current_mental_state:
            return  # 没有当前状态，无法更新
        
        # 创建反馈记录
        feedback = MentalRewardFeedback(
            user_id=self.user_id,
            timestamp=datetime.now(),
            state_before=self.state.previous_mental_state or self.state.current_mental_state,
            state_after=self.state.current_mental_state,
            explicit_rating=explicit_rating,
            feedback_text=feedback_text,
            continued_conversation=continued_conversation,
        )
        
        # 计算奖励
        reward = self.reward_model.calculate_reward(
            user_id=self.user_id,
            state_before=feedback.state_before,
            state_after=feedback.state_after,
            explicit_rating=explicit_rating,
            continued_conversation=continued_conversation,
        )
        feedback.calculated_reward = reward
        
        # 更新权重
        self.reward_model.update_weights_from_feedback(
            user_id=self.user_id,
            feedback=feedback,
        )

    async def end_session(self, user_satisfaction: Optional[int] = None) -> str:
        """
        End conversation, generate Summary
        """
        self.session.end_time = datetime.now()
        self.session.user_satisfaction = user_satisfaction

        # Generate conversation Summary
        summary = await self._generate_session_summary()
        self.session.summary = summary

        # Save session
        self.memory.save_session(self.session)

        # Update User profile
        self.memory.update_profile_from_session(self.user_id, self.session)

        return summary

    async def _generate_session_summary(self) -> str:
        """Use LLM to generate conversation Summary"""
        if not self.session.turns:
            return "No Conversation content"

        # Extract all conversations
        all_messages = "\n\n".join([
            f"User: {turn.user_message}\nAssistant: {turn.agent_response}"
            for turn in self.session.turns
        ])

        config = ModelRouter.get_model_config(TaskType.BEHAVIOR_ANALYSIS)

        prompt = f"""
Please summarize the following psychological counseling conversation:

{all_messages}

Please provide:
1. Main discussion Topics (tag form)
2. User Emotional state
3. Identified Cognitive patterns or distress
4. Intervention strategies used
5. User response and progress

Format: concise JSON
"""

        result = await self.llm.generate_structured(
            prompt=prompt,
            config=config,
        )

        # Extract Topics
        if 'themes' in result or 'topics' in result:
            themes = result.get('themes') or result.get('topics', [])
            self.session.identified_themes = themes if isinstance(themes, list) else [themes]

        return str(result)

    def get_conversation_history(self, last_n: int = 5) -> str:
        """Get recent N turns conversation (used for display)"""
        recent = self.session.turns[-last_n:]
        return "\n\n".join([
            f"**User**: {turn.user_message}\n**Assistant**: {turn.agent_response}"
            for turn in recent
        ])
