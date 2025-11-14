"""
Search Log Processor
Extract, filter, analyze user Search history
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import re


@dataclass
class SearchEntry:
    """Single item Search record"""
    timestamp: datetime
    query: str
    category: Optional[str] = None  # 'mental_health', 'medical', 'other'
    sentiment: Optional[str] = None  # 'crisis', 'concerning', 'neutral', 'positive'
    anonymized_query: Optional[str] = None


class SearchLogProcessor:
    """
    Search log handler
    Identify mental health related searches, protect privacy
    """

    # Mental health keyword classification
    MENTAL_HEALTH_KEYWORDS = {
        'crisis': [
            'suicide', '轻生', '不想活', '结束生命', 'self-harm', '割腕',
            '安眠药', '跳楼', '上吊',
        ],
        'depression': [
            'depression', '抑郁症', '情绪低落', '无望', '绝望', '悲伤',
            '失去兴趣', '抗抑郁药', 'feeling bad',
        ],
        'anxiety': [
            'anxiety', '焦虑症', '恐慌', '惊恐发作', '紧张', '担心',
            '失眠', '睡不着', '心慌', '社交恐惧',
        ],
        'stress': [
            'stress', '压力大', '疲惫', '倦怠', '工作压力', '学业压力',
            '职业倦怠', 'burnout',
        ],
        'therapy': [
            '心理咨询', '心理治疗', '心理医生', 'CBT', 'DBT',
            'Cognitive Behavioral Therapy', '正念', '冥想',
        ],
        'relationship': [
            '人际关系', '社交困难', '孤独', '分手', '失恋',
            '家庭矛盾', '亲子关系',
        ],
    }

    # Personal information patterns to be removed
    PI_PATTERNS = [
        (r'\b\d{11}\b', '[phone]'),  # phone number
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[email]'),  # email
        (r'\b\d{17}[\dXx]\b', '[id_card]'),  # ID card number
    ]

    def __init__(self):
        self.category_keywords = self._build_keyword_index()

    def _build_keyword_index(self) -> Dict[str, List[str]]:
        """Build keyword index"""
        return self.MENTAL_HEALTH_KEYWORDS

    def process_search_history(
        self,
        raw_searches: List[Dict[str, Any]],
        days: int = 7
    ) -> List[SearchEntry]:
        """
        Process Search history

        Args:
            raw_searches: Original Search records [{'timestamp': ..., 'query': ...}, ...]
            days: Process Data from recent days
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        processed = []

        for search in raw_searches:
            # Parse time
            if isinstance(search['timestamp'], str):
                timestamp = datetime.fromisoformat(search['timestamp'])
            else:
                timestamp = search['timestamp']

            if timestamp < cutoff_time:
                continue

            query = search['query']

            # Filter: only keep mental health related searches
            if not self.is_mental_health_related(query):
                continue

            # Anonymization
            anonymized = self.anonymize(query)

            # Classification
            category = self.categorize(query)

            # Sentiment evaluation
            sentiment = self.assess_sentiment(query)

            entry = SearchEntry(
                timestamp=timestamp,
                query=query,  # Original query only used for analysis, not stored
                category=category,
                sentiment=sentiment,
                anonymized_query=anonymized,
            )

            processed.append(entry)

        return processed

    def is_mental_health_related(self, query: str) -> bool:
        """Determine if search is related to mental health"""
        query_lower = query.lower()

        for category, keywords in self.category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return True

        return False

    def categorize(self, query: str) -> str:
        """Classify search progressively"""
        query_lower = query.lower()

        for category, keywords in self.category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return category

        return 'other'

    def assess_sentiment(self, query: str) -> str:
        """
        Evaluate search Sentiment tendency
        crisis > concerning > neutral > positive
        """
        query_lower = query.lower()

        # Crisis signal
        if any(kw in query_lower for kw in self.category_keywords['crisis']):
            return 'crisis'

        # Concerning search
        concerning_patterns = [
            '怎么办', '无法', '无力', '失败', '痛苦', '难受',
            '崩溃', '撑不下去',
        ]
        if any(pattern in query_lower for pattern in concerning_patterns):
            return 'concerning'

        # Positive seeking help
        positive_patterns = [
            '如何改善', '治疗方法', '如何应对', '缓解', '帮助',
            '咨询', '康复',
        ]
        if any(pattern in query_lower for pattern in positive_patterns):
            return 'positive'

        return 'neutral'

    def anonymize(self, query: str) -> str:
        """
        Anonymize search content
        Remove personally identifiable information, keep semantics
        """
        anonymized = query

        # Remove PI
        for pattern, replacement in self.PI_PATTERNS:
            anonymized = re.sub(pattern, replacement, anonymized)

        # Remove specific people names, place names (simplified version)
        # Actual should use NER model
        anonymized = re.sub(r'我(的)?(朋友|同事|老板|父母|孩子)', '[关系人]', anonymized)

        return anonymized

    def generate_summary(self, entries: List[SearchEntry]) -> Dict[str, Any]:
        """
        Generate Search history Summary

        Returns:
            Statistical info
        """
        if not entries:
            return {
                'total_searches': 0,
                'categories': {},
                'sentiments': {},
                'crisis_count': 0,
            }

        # By classification statistics
        category_counts = {}
        for entry in entries:
            cat = entry.category or 'other'
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # By Sentiment statistics
        sentiment_counts = {}
        crisis_count = 0
        for entry in entries:
            sent = entry.sentiment or 'neutral'
            sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
            if sent == 'crisis':
                crisis_count += 1

        return {
            'total_searches': len(entries),
            'categories': category_counts,
            'sentiments': sentiment_counts,
            'crisis_count': crisis_count,
            'date_range': {
                'start': min(e.timestamp for e in entries).isoformat(),
                'end': max(e.timestamp for e in entries).isoformat(),
            },
        }

    def get_top_concerns(self, entries: List[SearchEntry], top_n: int = 5) -> List[str]:
        """Extract most concerning topics"""
        category_counts = {}
        for entry in entries:
            cat = entry.category or 'other'
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Sort
        sorted_categories = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [cat for cat, _ in sorted_categories[:top_n]]


# Example: simulate search data
def generate_mock_search_data() -> List[Dict[str, Any]]:
    """Generate simulated search data (used for testing)"""
    mock_searches = [
        {'timestamp': datetime.now() - timedelta(hours=2), 'query': '如何应对焦虑症状'},
        {'timestamp': datetime.now() - timedelta(hours=5), 'query': '失眠怎么办'},
        {'timestamp': datetime.now() - timedelta(days=1), 'query': '工作压力大如何缓解'},
        {'timestamp': datetime.now() - timedelta(days=1), 'query': '心理咨询一次多少钱'},
        {'timestamp': datetime.now() - timedelta(days=2), 'query': '抑郁症症状'},
        {'timestamp': datetime.now() - timedelta(days=3), 'query': '社交恐惧症怎么治疗'},
        {'timestamp': datetime.now() - timedelta(days=4), 'query': '如何提高自信'},
        {'timestamp': datetime.now() - timedelta(days=5), 'query': '正念冥想方法'},
        # Not mental health related (will be filtered)
        {'timestamp': datetime.now() - timedelta(hours=1), 'query': '今天天气怎么样'},
        {'timestamp': datetime.now() - timedelta(days=1), 'query': 'python教程'},
    ]

    return mock_searches
