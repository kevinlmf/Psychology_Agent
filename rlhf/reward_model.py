"""
RLHF Reward Model
Multimodal Reward function, synthesizing human feedback and behavior data
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class Interaction:
    """Single interaction record"""
    interaction_id: str
    user_id: str
    timestamp: datetime

    # Conversation content
    user_message: str
    agent_response: str
    context: Dict[str, Any]  # User profile, historical etc

    # Human feedback
    explicit_rating: Optional[int] = None  # 1-5 points
    feedback_text: Optional[str] = None

    # Behavior feedback
    continued_conversation: bool = True
    session_length: int = 0  # Follow-up conversation turns
    user_satisfaction_indicators: Dict[str, Any] = None

    # Clinical metrics
    risk_level_before: Optional[str] = None
    risk_level_after: Optional[str] = None
    emotion_before: Optional[str] = None
    emotion_after: Optional[str] = None

    # Safety
    safety_violation: bool = False
    ethical_concern: bool = False


@dataclass
class PreferenceComparison:
    """Preference comparison (used for Training)"""
    context: str  # Same user input and background
    response_a: str
    response_b: str
    preference: str  # 'A', 'B', 'Equal'
    annotator_id: str
    confidence: float  # Annotator confidence
    reasoning: Optional[str] = None


class MultiModalRewardModel:
    """
    Multimodal Reward Model
    Synthesizing Explicit feedback, behavior data, clinical metrics
    """

    # Reward weight configuration
    REWARD_WEIGHTS = {
        'explicit_feedback': 0.3,  # User explicit rating
        'behavioral': 0.25,  # Behavior metrics (continued conversation, conversation length)
        'clinical': 0.25,  # Clinical metrics (Emotion improvement, Risk lower)
        'safety': 0.15,  # Safety (No violations)
        'engagement': 0.05,  # Engagement level
    }

    def __init__(self, storage_dir: str = "psychology_agent/data/rlhf"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Storage Interaction data and Preference comparison
        self.interactions: List[Interaction] = []
        self.preferences: List[PreferenceComparison] = []

    def calculate_reward(self, interaction: Interaction) -> float:
        """
        Calculate comprehensive Reward points

        Returns:
            Reward points (-1.0 to 1.0)
        """
        rewards = {}

        # 1. Explicit feedback Reward
        rewards['explicit_feedback'] = self._explicit_feedback_reward(interaction)

        # 2. Behavior Reward
        rewards['behavioral'] = self._behavioral_reward(interaction)

        # 3. Clinical metric Reward
        rewards['clinical'] = self._clinical_reward(interaction)

        # 4. Safety Reward
        rewards['safety'] = self._safety_reward(interaction)

        # 5. Engagement Reward
        rewards['engagement'] = self._engagement_reward(interaction)

        # Weighted sum
        total_reward = sum(
            rewards[key] * self.REWARD_WEIGHTS[key]
            for key in rewards
        )

        return total_reward

    def _explicit_feedback_reward(self, interaction: Interaction) -> float:
        """Explicit feedback Reward (user rating)"""
        if interaction.explicit_rating is None:
            return 0.0  # No rating

        # 1-5 points map to -1 to 1
        # 1 points -> -1, 3 points -> 0, 5 points -> 1
        normalized = (interaction.explicit_rating - 3) / 2
        return normalized

    def _behavioral_reward(self, interaction: Interaction) -> float:
        """Behavior metric Reward"""
        score = 0.0

        # User continued conversation (positive signal)
        if interaction.continued_conversation:
            score += 0.5

        # Conversation length (longer indicates user more engaged)
        # Assume ideal conversation length is 5-15 turns
        if interaction.session_length > 0:
            if 5 <= interaction.session_length <= 15:
                score += 0.5
            elif interaction.session_length > 15:
                score += 0.3  # Too long may be inefficient
            else:
                score += 0.2  # Too short not deep enough

        return min(score, 1.0)

    def _clinical_reward(self, interaction: Interaction) -> float:
        """Clinical metric Reward (Emotion improvement, Risk lower)"""
        score = 0.0

        # Risk level change
        risk_change = self._assess_risk_change(
            interaction.risk_level_before,
            interaction.risk_level_after
        )
        score += risk_change * 0.5

        # Emotion change
        emotion_change = self._assess_emotion_change(
            interaction.emotion_before,
            interaction.emotion_after
        )
        score += emotion_change * 0.5

        return score

    def _assess_risk_change(
        self,
        before: Optional[str],
        after: Optional[str]
    ) -> float:
        """Assess risk level change"""
        if before is None or after is None:
            return 0.0

        risk_scores = {'low': 0, 'medium': 1, 'high': 2}
        before_score = risk_scores.get(before, 0)
        after_score = risk_scores.get(after, 0)

        # Risk lower: positive Reward
        if after_score < before_score:
            return 1.0
        # Risk increase: negative Reward
        elif after_score > before_score:
            return -1.0
        # No change
        return 0.0

    def _assess_emotion_change(
        self,
        before: Optional[str],
        after: Optional[str]
    ) -> float:
        """Evaluate Emotion change"""
        if before is None or after is None:
            return 0.0

        # Simplified Emotion points
        emotion_scores = {
            'crisis': -2,
            'depressed': -1,
            'anxious': -0.5,
            'stressed': -0.5,
            'neutral': 0,
            'stable': 0.5,
            'positive': 1,
        }

        before_score = emotion_scores.get(before, 0)
        after_score = emotion_scores.get(after, 0)

        # Emotion improvement
        change = after_score - before_score
        # Normalize to -1 to 1
        return max(min(change / 2, 1.0), -1.0)

    def _safety_reward(self, interaction: Interaction) -> float:
        """Safety Reward"""
        if interaction.safety_violation:
            return -1.0
        if interaction.ethical_concern:
            return -0.5
        # No safety issues
        return 1.0

    def _engagement_reward(self, interaction: Interaction) -> float:
        """Engagement Reward"""
        if interaction.user_satisfaction_indicators is None:
            return 0.0

        indicators = interaction.user_satisfaction_indicators
        score = 0.0

        # User actively shares (deep engagement)
        if indicators.get('shared_personal_info'):
            score += 0.5

        # User asks questions (positive exploration)
        if indicators.get('asked_questions'):
            score += 0.3

        # User expresses thanks
        if indicators.get('expressed_gratitude'):
            score += 0.2

        return min(score, 1.0)

    def add_interaction(self, interaction: Interaction):
        """Add interaction record"""
        self.interactions.append(interaction)

        # Periodically save
        if len(self.interactions) % 10 == 0:
            self.save_interactions()

    def add_preference(self, preference: PreferenceComparison):
        """Add Preference comparison data"""
        self.preferences.append(preference)

        if len(self.preferences) % 10 == 0:
            self.save_preferences()

    def save_interactions(self):
        """Save Interaction data"""
        file_path = self.storage_dir / "interactions.jsonl"

        with open(file_path, 'a', encoding='utf-8') as f:
            for interaction in self.interactions:
                data = {
                    'interaction_id': interaction.interaction_id,
                    'user_id': interaction.user_id,
                    'timestamp': interaction.timestamp.isoformat(),
                    'user_message': interaction.user_message,
                    'agent_response': interaction.agent_response,
                    'explicit_rating': interaction.explicit_rating,
                    'reward': self.calculate_reward(interaction),
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        # Clear memory
        self.interactions = []

    def save_preferences(self):
        """Save Preference comparison data"""
        file_path = self.storage_dir / "preferences.jsonl"

        with open(file_path, 'a', encoding='utf-8') as f:
            for pref in self.preferences:
                data = {
                    'context': pref.context,
                    'response_a': pref.response_a,
                    'response_b': pref.response_b,
                    'preference': pref.preference,
                    'annotator_id': pref.annotator_id,
                    'confidence': pref.confidence,
                    'reasoning': pref.reasoning,
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        self.preferences = []

    def get_training_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Get Training data

        Returns:
            (interactions_data, preferences_data)
        """
        interactions_file = self.storage_dir / "interactions.jsonl"
        preferences_file = self.storage_dir / "preferences.jsonl"

        interactions_data = []
        if interactions_file.exists():
            with open(interactions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    interactions_data.append(json.loads(line))

        preferences_data = []
        if preferences_file.exists():
            with open(preferences_file, 'r', encoding='utf-8') as f:
                for line in f:
                    preferences_data.append(json.loads(line))

        return interactions_data, preferences_data

    def generate_statistics(self) -> Dict[str, Any]:
        """Generate Training data statistics"""
        interactions_data, preferences_data = self.get_training_data()

        if not interactions_data:
            return {'error': 'No Training data'}

        # Calculate average Reward
        rewards = [d['reward'] for d in interactions_data if 'reward' in d]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0

        # Rating distribution
        ratings = [d['explicit_rating'] for d in interactions_data if d.get('explicit_rating')]
        rating_dist = {}
        for r in ratings:
            rating_dist[r] = rating_dist.get(r, 0) + 1

        return {
            'total_interactions': len(interactions_data),
            'total_preferences': len(preferences_data),
            'average_reward': avg_reward,
            'rating_distribution': rating_dist,
            'reward_range': (min(rewards), max(rewards)) if rewards else (0, 0),
        }


# Global singleton
_reward_model = None


def get_reward_model() -> MultiModalRewardModel:
    """Get global Reward Model instance"""
    global _reward_model
    if _reward_model is None:
        _reward_model = MultiModalRewardModel()
    return _reward_model
