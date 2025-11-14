"""
User memory and profile management system
Store user history, preferences, and treatment progress
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

from agent.personality_analyzer import PersonalityProfile


@dataclass
class UserProfile:
    """User profile"""
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)

    # Basic information
    age: Optional[int] = None
    gender: Optional[str] = None

    # Mental health status
    main_concerns: List[str] = field(default_factory=list)  # Main concerns
    diagnosed_conditions: List[str] = field(default_factory=list)  # Diagnosed conditions

    # Therapy preferences
    preferred_therapy_style: Optional[str] = None  # CBT, DBT, ACT etc
    communication_preference: str = "balanced"  # direct, gentle, balanced

    # Historically effective strategies
    effective_strategies: List[str] = field(default_factory=list)
    ineffective_strategies: List[str] = field(default_factory=list)

    # Risk assessment history
    risk_history: List[Dict[str, Any]] = field(default_factory=list)

    # Treatment goals
    goals: List[str] = field(default_factory=list)
    
    # Personality profile
    personality_profile: Optional[PersonalityProfile] = None

    # Metadata
    total_sessions: int = 0
    last_session_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)"""
        data = asdict(self)
        # Convert datetime to string
        data['created_at'] = self.created_at.isoformat()
        if self.last_session_date:
            data['last_session_date'] = self.last_session_date.isoformat()
        # Convert personality profile
        if self.personality_profile:
            data['personality_profile'] = self.personality_profile.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create from dictionary"""
        # Convert string to datetime
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_session_date'):
            data['last_session_date'] = datetime.fromisoformat(data['last_session_date'])
        # Convert personality profile
        if 'personality_profile' in data and data['personality_profile']:
            data['personality_profile'] = PersonalityProfile.from_dict(data['personality_profile'])
        return cls(**data)


@dataclass
class ConversationTurn:
    """Single conversation turn"""
    timestamp: datetime
    user_message: str
    agent_response: str
    detected_emotion: Optional[str] = None
    risk_level: Optional[str] = None
    intervention_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class Session:
    """A session"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    summary: Optional[str] = None
    identified_themes: List[str] = field(default_factory=list)
    user_satisfaction: Optional[int] = None  # 1-5 points

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        data['turns'] = [turn.to_dict() for turn in self.turns]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        data['turns'] = [ConversationTurn.from_dict(t) for t in data.get('turns', [])]
        return cls(**data)


class MemorySystem:
    """
    Memory management system
    Responsible for storage and retrieval of user profiles and session history
    """

    def __init__(self, storage_dir: str = "data/user_profiles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._profiles: Dict[str, UserProfile] = {}
        self._sessions: Dict[str, List[Session]] = {}

    def _get_profile_path(self, user_id: str) -> Path:
        """Get user profile file path"""
        return self.storage_dir / f"{user_id}_profile.json"

    def _get_sessions_path(self, user_id: str) -> Path:
        """Get session history file path"""
        return self.storage_dir / f"{user_id}_sessions.json"

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id in self._profiles:
            return self._profiles[user_id]

        profile_path = self._get_profile_path(user_id)

        if profile_path.exists():
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                profile = UserProfile.from_dict(data)
        else:
            profile = UserProfile(user_id=user_id)

        self._profiles[user_id] = profile
        return profile

    def save_profile(self, profile: UserProfile):
        """Save user profile"""
        self._profiles[profile.user_id] = profile
        profile_path = self._get_profile_path(profile.user_id)

        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)

    def get_sessions(self, user_id: str, recent_n: Optional[int] = None) -> List[Session]:
        """Get user session history"""
        if user_id not in self._sessions:
            sessions_path = self._get_sessions_path(user_id)

            if sessions_path.exists():
                with open(sessions_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._sessions[user_id] = [Session.from_dict(s) for s in data]
            else:
                self._sessions[user_id] = []

        sessions = self._sessions[user_id]
        if recent_n:
            return sessions[-recent_n:]
        return sessions

    def save_session(self, session: Session):
        """Save session"""
        if session.user_id not in self._sessions:
            self._sessions[session.user_id] = []

        # Add or update session
        existing = [s for s in self._sessions[session.user_id] if s.session_id == session.session_id]

        if existing:
            idx = self._sessions[session.user_id].index(existing[0])
            self._sessions[session.user_id][idx] = session
        else:
            self._sessions[session.user_id].append(session)

        # Save to file
        sessions_path = self._get_sessions_path(session.user_id)
        with open(sessions_path, 'w', encoding='utf-8') as f:
            data = [s.to_dict() for s in self._sessions[session.user_id]]
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_recent_context(self, user_id: str, days: int = 7) -> str:
        """
        Get recent conversation context summary (for LLM prompt)

        Args:
            user_id: User ID
            days: Data from recent days
        """
        sessions = self.get_sessions(user_id)
        cutoff_time = datetime.now() - timedelta(days=days)

        recent_sessions = [s for s in sessions if s.start_time > cutoff_time]

        if not recent_sessions:
            return "No recent conversation history"

        context_parts = []
        for session in recent_sessions[-3:]:  # At most 3 recent sessions
            date_str = session.start_time.strftime("%Y-%m-%d")
            context_parts.append(f"[{date_str}] Topics: {', '.join(session.identified_themes)}")
            if session.summary:
                context_parts.append(f"  Summary: {session.summary}")

        return "\n".join(context_parts)

    def update_profile_from_session(self, user_id: str, session: Session):
        """Update user profile from session"""
        profile = self.get_or_create_profile(user_id)

        # Update session statistics
        profile.total_sessions += 1
        profile.last_session_date = session.end_time or session.start_time

        # Update identified Topics
        for theme in session.identified_themes:
            if theme not in profile.main_concerns:
                profile.main_concerns.append(theme)

        self.save_profile(profile)

    def get_user_summary(self, user_id: str) -> str:
        """Generate User profile Summary (used in LLM prompt)"""
        profile = self.get_or_create_profile(user_id)

        parts = [
            f"User ID: {profile.user_id}",
            f"Total sessions: {profile.total_sessions}",
        ]

        if profile.main_concerns:
            parts.append(f"Main concerns: {', '.join(profile.main_concerns)}")

        if profile.goals:
            parts.append(f"Treatment goals: {', '.join(profile.goals)}")

        if profile.effective_strategies:
            parts.append(f"Effective strategies: {', '.join(profile.effective_strategies)}")

        if profile.communication_preference:
            parts.append(f"Communication preference: {profile.communication_preference}")

        return "\n".join(parts)


# Global singleton
_memory_system = None


def get_memory_system(storage_dir: str = "psychology_agent/data/user_profiles") -> MemorySystem:
    """Get global memory system instance"""
    global _memory_system
    if _memory_system is None:
        _memory_system = MemorySystem(storage_dir)
    return _memory_system
