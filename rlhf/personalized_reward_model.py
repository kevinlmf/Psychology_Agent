"""
个性化心理奖励模型 (Personalized Mental Reward Function)
RLHF = 心理奖励函数建构，根据用户反馈调整奖励权重
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
from pathlib import Path

from agent.mental_state_interpreter import MentalStateVector
from agent.personality_analyzer import PersonalityProfile


@dataclass
class PersonalizedRewardWeights:
    """个性化奖励权重配置"""
    
    # 核心心理健康奖励维度
    emotional_stability: float = 0.30  # 情绪回到安全范围
    stress_reduction: float = 0.25      # 焦虑下降
    self_efficacy: float = 0.10        # "我可以做到"的信念增强
    cognitive_clarity: float = 0.15    # 不再过度思考
    behavioral_consistency: float = 0.10  # 健康行为坚持度
    compassion: float = 0.10           # 对自己温柔、支持
    
    def normalize(self):
        """归一化权重，确保总和为1.0"""
        total = sum([
            self.emotional_stability,
            self.stress_reduction,
            self.self_efficacy,
            self.cognitive_clarity,
            self.behavioral_consistency,
            self.compassion,
        ])
        if total > 0:
            self.emotional_stability /= total
            self.stress_reduction /= total
            self.self_efficacy /= total
            self.cognitive_clarity /= total
            self.behavioral_consistency /= total
            self.compassion /= total
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "emotional_stability": self.emotional_stability,
            "stress_reduction": self.stress_reduction,
            "self_efficacy": self.self_efficacy,
            "cognitive_clarity": self.cognitive_clarity,
            "behavioral_consistency": self.behavioral_consistency,
            "compassion": self.compassion,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PersonalizedRewardWeights':
        """从字典创建"""
        return cls(**data)


@dataclass
class MentalRewardFeedback:
    """心理奖励反馈记录"""
    user_id: str
    timestamp: datetime
    
    # 状态变化
    state_before: MentalStateVector
    state_after: Optional[MentalStateVector] = None
    
    # 用户反馈
    explicit_rating: Optional[int] = None  # 1-5
    feedback_text: Optional[str] = None
    
    # 行为指标
    continued_conversation: bool = True
    response_helpful: Optional[bool] = None
    
    # 奖励值
    calculated_reward: Optional[float] = None


class PersonalizedMentalRewardModel:
    """
    个性化心理奖励模型
    根据用户历史反馈调整奖励权重，形成个性化奖励函数
    """
    
    def __init__(self, storage_dir: str = "psychology_agent/data/rlhf"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 用户个性化权重缓存
        self.user_weights: Dict[str, PersonalizedRewardWeights] = {}
        
        # 反馈历史
        self.feedback_history: Dict[str, List[MentalRewardFeedback]] = {}
    
    def get_user_weights(
        self, 
        user_id: str,
        personality_profile: Optional[PersonalityProfile] = None,
    ) -> PersonalizedRewardWeights:
        """
        获取用户的个性化权重
        
        Args:
            user_id: 用户ID
            personality_profile: 性格画像（如果提供，用于初始化权重）
        
        Returns:
            PersonalizedRewardWeights: 个性化权重
        """
        if user_id not in self.user_weights:
            # 尝试从文件加载
            weights = self._load_user_weights(user_id)
            if weights is None:
                # 如果没有保存的权重，根据性格特征初始化
                if personality_profile:
                    weights = self._initialize_weights_from_personality(personality_profile)
                else:
                    weights = PersonalizedRewardWeights()  # 默认权重
            self.user_weights[user_id] = weights
        
        return self.user_weights[user_id]
    
    def _initialize_weights_from_personality(
        self,
        personality: PersonalityProfile,
    ) -> PersonalizedRewardWeights:
        """
        根据性格特征初始化奖励权重
        
        性格特征 → RLHF权重映射规则：
        - 高神经质 → 增加 emotional_stability 和 stress_reduction 权重
        - 高外向性 → 增加 compassion 权重
        - 高尽责性 → 增加 behavioral_consistency 权重
        - 高开放性 → 增加 cognitive_clarity 权重
        - 高宜人性 → 增加 compassion 权重
        - 问题导向应对 → 增加 self_efficacy 权重
        - 情绪导向应对 → 增加 compassion 权重
        """
        weights = PersonalizedRewardWeights()
        
        # 基础权重
        base_weights = {
            "emotional_stability": 0.30,
            "stress_reduction": 0.25,
            "self_efficacy": 0.10,
            "cognitive_clarity": 0.15,
            "behavioral_consistency": 0.10,
            "compassion": 0.10,
        }
        
        # 根据Big Five调整
        # 高神经质 → 需要更多情绪稳定支持
        if personality.neuroticism > 0.6:
            base_weights["emotional_stability"] += 0.10
            base_weights["stress_reduction"] += 0.10
            base_weights["compassion"] += 0.05
        
        # 低神经质 → 可以更多关注认知和行为
        elif personality.neuroticism < 0.4:
            base_weights["self_efficacy"] += 0.05
            base_weights["cognitive_clarity"] += 0.05
            base_weights["behavioral_consistency"] += 0.05
        
        # 高外向性 → 更多社交支持
        if personality.extraversion > 0.6:
            base_weights["compassion"] += 0.05
            base_weights["behavioral_consistency"] += 0.03
        
        # 低外向性（内向） → 更多自我效能
        elif personality.extraversion < 0.4:
            base_weights["self_efficacy"] += 0.05
            base_weights["cognitive_clarity"] += 0.05
        
        # 高尽责性 → 更多行为一致性支持
        if personality.conscientiousness > 0.6:
            base_weights["behavioral_consistency"] += 0.05
            base_weights["self_efficacy"] += 0.03
        
        # 高开放性 → 更多认知清晰度支持
        if personality.openness > 0.6:
            base_weights["cognitive_clarity"] += 0.05
        
        # 高宜人性 → 更多同理心支持
        if personality.agreeableness > 0.6:
            base_weights["compassion"] += 0.05
        
        # 根据应对机制调整
        if personality.coping_style == "problem_focused":
            base_weights["self_efficacy"] += 0.05
            base_weights["cognitive_clarity"] += 0.03
        elif personality.coping_style == "emotion_focused":
            base_weights["compassion"] += 0.05
            base_weights["emotional_stability"] += 0.03
        
        # 根据回应偏好调整
        if personality.response_preference == "emotional_support":
            base_weights["compassion"] += 0.05
            base_weights["emotional_stability"] += 0.03
        elif personality.response_preference == "practical_advice":
            base_weights["self_efficacy"] += 0.05
            base_weights["behavioral_consistency"] += 0.03
        elif personality.response_preference == "cognitive_reframing":
            base_weights["cognitive_clarity"] += 0.05
            base_weights["self_efficacy"] += 0.03
        
        # 应用权重
        weights.emotional_stability = base_weights["emotional_stability"]
        weights.stress_reduction = base_weights["stress_reduction"]
        weights.self_efficacy = base_weights["self_efficacy"]
        weights.cognitive_clarity = base_weights["cognitive_clarity"]
        weights.behavioral_consistency = base_weights["behavioral_consistency"]
        weights.compassion = base_weights["compassion"]
        
        # 归一化
        weights.normalize()
        
        return weights
    
    def calculate_reward(
        self,
        user_id: str,
        state_before: MentalStateVector,
        state_after: Optional[MentalStateVector] = None,
        explicit_rating: Optional[int] = None,
        continued_conversation: bool = True,
    ) -> float:
        """
        计算个性化心理奖励
        
        Args:
            user_id: 用户ID
            state_before: 交互前的心理状态
            state_after: 交互后的心理状态（可选）
            explicit_rating: 用户显式评分（1-5）
            continued_conversation: 是否继续对话
        
        Returns:
            奖励值 (-1.0 到 1.0)
        """
        weights = self.get_user_weights(user_id)
        
        reward_components = {}
        
        # 1. Emotional Stability Reward (情绪稳定性)
        if state_after:
            # 计算情绪改善：焦虑、抑郁、压力下降
            anxiety_improvement = max(0, state_before.anxiety - state_after.anxiety)
            depression_improvement = max(0, state_before.depression - state_after.depression)
            stress_improvement = max(0, state_before.stress - state_after.stress)
            
            emotional_stability_score = (
                anxiety_improvement * 0.4 +
                depression_improvement * 0.3 +
                stress_improvement * 0.3
            )
            reward_components['emotional_stability'] = emotional_stability_score
        else:
            reward_components['emotional_stability'] = 0.0
        
        # 2. Stress Reduction Reward (压力减少)
        if state_after:
            stress_reduction_score = max(0, state_before.stress - state_after.stress)
            reward_components['stress_reduction'] = stress_reduction_score
        else:
            reward_components['stress_reduction'] = 0.0
        
        # 3. Self-Efficacy Reward (自我效能感增强)
        if state_after:
            efficacy_improvement = max(0, state_after.self_efficacy - state_before.self_efficacy)
            reward_components['self_efficacy'] = efficacy_improvement
        else:
            reward_components['self_efficacy'] = 0.0
        
        # 4. Cognitive Clarity Reward (认知清晰度)
        if state_after:
            # 认知模式减少（过度思考等减少）
            patterns_before = len(state_before.cognitive_patterns)
            patterns_after = len(state_after.cognitive_patterns)
            clarity_score = max(0, (patterns_before - patterns_after) / max(patterns_before, 1))
            reward_components['cognitive_clarity'] = clarity_score
        else:
            reward_components['cognitive_clarity'] = 0.0
        
        # 5. Behavioral Consistency Reward (行为一致性)
        # 基于用户是否继续对话
        reward_components['behavioral_consistency'] = 1.0 if continued_conversation else 0.0
        
        # 6. Compassion Reward (同理心/支持)
        # 基于显式评分（如果用户给出高评分，说明感受到了支持）
        if explicit_rating:
            compassion_score = (explicit_rating - 3) / 2.0  # 1-5映射到-1到1，然后归一化到0-1
            reward_components['compassion'] = max(0, (compassion_score + 1) / 2)
        else:
            reward_components['compassion'] = 0.5  # 中性值
        
        # 加权求和
        total_reward = sum(
            reward_components[key] * getattr(weights, key)
            for key in reward_components
        )
        
        return total_reward
    
    def update_weights_from_feedback(
        self,
        user_id: str,
        feedback: MentalRewardFeedback,
        learning_rate: float = 0.1,
    ):
        """
        根据用户反馈更新个性化权重
        
        Args:
            user_id: 用户ID
            feedback: 反馈记录
            learning_rate: 学习率（权重调整幅度）
        """
        weights = self.get_user_weights(user_id)
        
        # 如果用户给出了显式反馈，根据反馈调整权重
        if feedback.explicit_rating:
            rating = feedback.explicit_rating
            
            # 如果评分高（4-5），增强当前响应的奖励维度权重
            if rating >= 4:
                # 分析哪些维度可能贡献了高评分
                if feedback.state_after:
                    # 如果情绪改善明显，增强emotional_stability权重
                    if feedback.state_after.anxiety < feedback.state_before.anxiety:
                        weights.emotional_stability += learning_rate * 0.1
                        weights.stress_reduction += learning_rate * 0.1
                    
                    # 如果自我效能感提升，增强self_efficacy权重
                    if feedback.state_after.self_efficacy > feedback.state_before.self_efficacy:
                        weights.self_efficacy += learning_rate * 0.1
                    
                    # 如果认知模式减少，增强cognitive_clarity权重
                    if len(feedback.state_after.cognitive_patterns) < len(feedback.state_before.cognitive_patterns):
                        weights.cognitive_clarity += learning_rate * 0.1
                
                # 用户继续对话，说明行为一致性好
                if feedback.continued_conversation:
                    weights.behavioral_consistency += learning_rate * 0.05
                
                # 高评分通常意味着感受到了支持
                weights.compassion += learning_rate * 0.05
            
            # 如果评分低（1-2），降低相关权重
            elif rating <= 2:
                # 降低所有权重，但保持相对比例
                weights.emotional_stability *= (1 - learning_rate * 0.5)
                weights.stress_reduction *= (1 - learning_rate * 0.5)
                weights.self_efficacy *= (1 - learning_rate * 0.5)
                weights.cognitive_clarity *= (1 - learning_rate * 0.5)
                weights.behavioral_consistency *= (1 - learning_rate * 0.5)
                weights.compassion *= (1 - learning_rate * 0.5)
        
        # 归一化权重
        weights.normalize()
        
        # 保存更新后的权重
        self.user_weights[user_id] = weights
        self._save_user_weights(user_id, weights)
        
        # 记录反馈
        if user_id not in self.feedback_history:
            self.feedback_history[user_id] = []
        self.feedback_history[user_id].append(feedback)
    
    def _load_user_weights(self, user_id: str) -> Optional[PersonalizedRewardWeights]:
        """从文件加载用户权重"""
        weights_file = self.storage_dir / f"{user_id}_reward_weights.json"
        
        if weights_file.exists():
            try:
                with open(weights_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return PersonalizedRewardWeights.from_dict(data)
            except Exception as e:
                print(f"Error loading weights for {user_id}: {e}")
        
        return None
    
    def _save_user_weights(self, user_id: str, weights: PersonalizedRewardWeights):
        """保存用户权重到文件"""
        weights_file = self.storage_dir / f"{user_id}_reward_weights.json"
        
        try:
            with open(weights_file, 'w', encoding='utf-8') as f:
                json.dump(weights.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving weights for {user_id}: {e}")


# Global singleton
_personalized_reward_model = None


def get_personalized_reward_model() -> PersonalizedMentalRewardModel:
    """获取全局个性化奖励模型实例"""
    global _personalized_reward_model
    if _personalized_reward_model is None:
        _personalized_reward_model = PersonalizedMentalRewardModel()
    return _personalized_reward_model

