"""
User Behavior pattern analysis
Use LLM to analyze multimodal user data, identify mental health metrics
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from models import (
    get_orchestrator,
    ModelRouter,
    TaskType,
    SystemPrompts,
)


@dataclass
class BehaviorPattern:
    """Behavior pattern analysis result"""
    timestamp: datetime
    user_id: str

    # Emotion metrics
    emotional_state: str  # 'anxious', 'depressed', 'stressed', 'stable'
    emotion_confidence: float

    # Identified topics
    identified_themes: List[str]

    # Behavior changes
    behavior_changes: Dict[str, str]  # {'sleep': 'deteriorating', 'social': 'improving'}

    # Risk factors
    risk_factors: List[str]
    protective_factors: List[str]

    # Original data sources
    data_sources: List[str]

    # LLM analysis result
    raw_analysis: Dict[str, Any]


class BehaviorAnalyzer:
    """
    Behavior pattern analyzer
    Integrates multi-source data, uses LLM for in-depth analysis
    """

    def __init__(self):
        self.llm = get_orchestrator()

    async def analyze_recent_activity(
        self,
        user_id: str,
        search_history: Optional[List[str]] = None,
        app_usage: Optional[Dict[str, Any]] = None,
        conversation_summary: Optional[str] = None,
        days: int = 7,
    ) -> BehaviorPattern:
        """
        Analyze user recent activity pattern

        Args:
            user_id: User ID
            search_history: Search records
            app_usage: App usage data
            conversation_summary: Conversation summary
            days: Analyze recent days
        """
        # Build analysis prompt
        data_parts = []
        data_sources = []

        if search_history:
            data_parts.append(f"Search records (recent {len(search_history)} items):\n" + "\n".join(search_history))
            data_sources.append("search_history")

        if app_usage:
            data_parts.append(f"App usage pattern:\n{self._format_app_usage(app_usage)}")
            data_sources.append("app_usage")

        if conversation_summary:
            data_parts.append(f"Conversation summary:\n{conversation_summary}")
            data_sources.append("conversation")

        if not data_parts:
            # No data, return default value
            return self._create_default_pattern(user_id)

        analysis_prompt = f"""
Analyze user recent {days} days behavior data:

{chr(10).join(data_parts)}

Please analyze from mental health expert perspective:

1. **Emotional state evaluation**
   - Main emotion (anxiety/depression/stress/stable)
   - Confidence (0.0-1.0)

2. **Identify psychological topics**
   - List topics user is concerned about
   - For example: insomnia, social anxiety, work stress, interpersonal relationships

3. **Behavior pattern changes**
   - Sleep quality (improving/stable/deteriorating)
   - Social activity (increasing/stable/decreasing)
   - Self-care (improving/stable/deteriorating)

4. **Risk factors**
   - Factors that may worsen psychological distress

5. **Protective factors**
   - Factors beneficial to mental health

Output strict JSON format:
{{
    "emotional_state": "anxious|depressed|stressed|stable",
    "emotion_confidence": 0.0-1.0,
    "identified_themes": ["topic1", "topic2"],
    "behavior_changes": {{
        "sleep": "improving|stable|deteriorating",
        "social": "increasing|stable|decreasing",
        "self_care": "improving|stable|deteriorating"
    }},
    "risk_factors": ["factor1", "factor2"],
    "protective_factors": ["factor1", "factor2"],
    "key_insights": "brief analysis summary"
}}
"""

        config = ModelRouter.get_model_config(TaskType.BEHAVIOR_ANALYSIS)

        result = await self.llm.generate_structured(
            prompt=analysis_prompt,
            config=config,
            system_prompt=SystemPrompts.BEHAVIOR_ANALYST,
        )

        # Parse result
        return BehaviorPattern(
            timestamp=datetime.now(),
            user_id=user_id,
            emotional_state=result.get('emotional_state', 'stable'),
            emotion_confidence=result.get('emotion_confidence', 0.5),
            identified_themes=result.get('identified_themes', []),
            behavior_changes=result.get('behavior_changes', {}),
            risk_factors=result.get('risk_factors', []),
            protective_factors=result.get('protective_factors', []),
            data_sources=data_sources,
            raw_analysis=result,
        )

    async def compare_patterns(
        self,
        current: BehaviorPattern,
        previous: BehaviorPattern
    ) -> Dict[str, Any]:
        """
        Compare two timepoint Behavior patterns, identify change trend

        Args:
            current: Current pattern
            previous: Previous pattern
        """
        comparison_prompt = f"""
Compare user Behavior pattern changes:

**Before ({previous.timestamp.strftime('%Y-%m-%d')})**
- Emotional state: {previous.emotional_state}
- Topics: {', '.join(previous.identified_themes)}
- Risk factors: {', '.join(previous.risk_factors)}

**Current ({current.timestamp.strftime('%Y-%m-%d')})**
- Emotional state: {current.emotional_state}
- Topics: {', '.join(current.identified_themes)}
- Risk factors: {', '.join(current.risk_factors)}

Please analyze:
1. Overall trend (improving/stable/declining)
2. Key change points
3. New risks requiring attention
4. Positive progress

Output JSON format.
"""

        config = ModelRouter.get_model_config(TaskType.BEHAVIOR_ANALYSIS)

        result = await self.llm.generate_structured(
            prompt=comparison_prompt,
            config=config,
        )

        return result

    def _format_app_usage(self, app_usage: Dict[str, Any]) -> str:
        """Format App usage data"""
        parts = []

        if 'screen_time' in app_usage:
            parts.append(f"Screen time: {app_usage['screen_time']} hours/day")

        if 'social_media_time' in app_usage:
            parts.append(f"Social media: {app_usage['social_media_time']} hours/day")

        if 'sleep_tracking' in app_usage:
            parts.append(f"Sleep: {app_usage['sleep_tracking']} hours/day")

        if 'exercise' in app_usage:
            parts.append(f"Exercise: {app_usage['exercise']} minutes/day")

        return "\n".join(parts)

    def _create_default_pattern(self, user_id: str) -> BehaviorPattern:
        """Create default pattern (when no data)"""
        return BehaviorPattern(
            timestamp=datetime.now(),
            user_id=user_id,
            emotional_state='unknown',
            emotion_confidence=0.0,
            identified_themes=[],
            behavior_changes={},
            risk_factors=[],
            protective_factors=[],
            data_sources=[],
            raw_analysis={},
        )

    async def generate_personalized_insights(self, pattern: BehaviorPattern) -> str:
        """
        Generate Personalized insights and recommendations based on Behavior pattern
        """
        insights_prompt = f"""
Based on user Behavior pattern analysis:

Emotional state: {pattern.emotional_state} (confidence: {pattern.emotion_confidence})
Topics of concern: {', '.join(pattern.identified_themes)}
Behavior changes: {pattern.behavior_changes}
Risk factors: {', '.join(pattern.risk_factors)}
Protective factors: {', '.join(pattern.protective_factors)}

Please generate:
1. Warm, empathetic feedback on user current state
2. 2-3 practical, specific recommendations
3. Small exercises or techniques to try

Tone: Supportive, Non-judgmental, encouraging
Length: 3-4 paragraphs
"""

        config = ModelRouter.get_model_config(TaskType.INTERVENTION_PLANNING)

        insights = await self.llm.generate(
            prompt=insights_prompt,
            config=config,
            system_prompt=SystemPrompts.THERAPIST_BASE,
        )

        return insights
