"""
Feedback Collector
Used to collect human feedback (ratings, Preference comparison)
"""

from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from .reward_model import (
    get_reward_model,
    Interaction,
    PreferenceComparison,
)


class FeedbackCollector:
    """
    Feedback Collector
    Can integrate into Web UI or Command Line interface
    """

    def __init__(self):
        self.reward_model = get_reward_model()

    def collect_rating(
        self,
        user_id: str,
        user_message: str,
        agent_response: str,
        rating: int,
        feedback_text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Collect user rating feedback

        Args:
            user_id: User ID
            user_message: user message
            agent_response: agent Response
            rating: Rating (1-5)
            feedback_text: Optional text feedback
            context: Conversation context

        Returns:
            interaction_id
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1-5")

        interaction = Interaction(
            interaction_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime.now(),
            user_message=user_message,
            agent_response=agent_response,
            context=context or {},
            explicit_rating=rating,
            feedback_text=feedback_text,
        )

        self.reward_model.add_interaction(interaction)

        return interaction.interaction_id

    def collect_comparison(
        self,
        context: str,
        response_a: str,
        response_b: str,
        preference: str,
        annotator_id: str,
        confidence: float = 1.0,
        reasoning: Optional[str] = None,
    ):
        """
        Collect Preference comparison data

        Args:
            context: Same user input
            response_a: Response A
            response_b: Response B
            preference: 'A' / 'B' / 'Equal'
            annotator_id: Annotator ID
            confidence: Annotation confidence (0-1)
            reasoning: Selection reasoning
        """
        if preference not in ['A', 'B', 'Equal']:
            raise ValueError("Preference must be 'A', 'B', or 'Equal'")

        comparison = PreferenceComparison(
            context=context,
            response_a=response_a,
            response_b=response_b,
            preference=preference,
            annotator_id=annotator_id,
            confidence=confidence,
            reasoning=reasoning,
        )

        self.reward_model.add_preference(comparison)

    def update_behavioral_feedback(
        self,
        interaction_id: str,
        continued: bool,
        session_length: int,
    ):
        """
        Update behavior feedback
        Call after user ends conversation
        """
        # TODO: implementation to find and update existing interaction
        pass

    def prompt_for_rating(self, agent_response: str) -> str:
        """
        Generate collect rating prompt info
        """
        return f"""
---
Please rate this Response (1-5):
1 = very poor, not helpful
2 = poor, help limited
3 = average, somewhat helpful
4 = good, quite helpful
5 = very good, extremely helpful

Input rating or continue conversation directly.
"""

    def create_comparison_task(
        self,
        user_message: str,
        responses: list
    ) -> Dict[str, Any]:
        """
        Create comparison annotation task

        Args:
            user_message: user message
            responses: Multiple candidate Responses

        Returns:
            Annotation task data
        """
        return {
            'task_id': str(uuid.uuid4()),
            'context': user_message,
            'responses': responses,
            'instructions': """
Please compare these Responses, select better one:

Evaluation dimensions:
1. Empathy and understanding
2. Professionalism and accuracy
3. Practicality and feasibility
4. Safety
5. Naturalness

Select A, B or Equal, and explain reasoning.
""",
        }


# Used for Command Line simple Feedback collection
def collect_rating_cli(
    user_message: str,
    agent_response: str,
    user_id: str = "cli_user"
) -> Optional[int]:
    """
    Command Line version rating collection
    """
    print("\n" + "=" * 50)
    print("💬 Conversation review:")
    print(f"User: {user_message}")
    print(f"Assistant: {agent_response[:200]}...")
    print("=" * 50)

    print("\nPlease rate this Response (1-5, or press Enter to Skip):")
    print("1=very poor 2=poor 3=average 4=good 5=very good")

    try:
        user_input = input("Rating: ").strip()

        if not user_input:
            return None

        rating = int(user_input)

        if 1 <= rating <= 5:
            collector = FeedbackCollector()
            collector.collect_rating(
                user_id=user_id,
                user_message=user_message,
                agent_response=agent_response,
                rating=rating,
            )
            print(f"✓ Recorded rating: {rating}")
            return rating
        else:
            print("⚠ Rating must be between 1-5")
            return None

    except ValueError:
        print("⚠ Invalid input")
        return None
    except KeyboardInterrupt:
        print("\nCanceled")
        return None
