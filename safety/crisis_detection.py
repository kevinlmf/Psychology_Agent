"""
Crisis Detection system
Quickly identify suicide Risk, Self-harm behavior etc emergency situations
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from models import (
    get_orchestrator,
    ModelRouter,
    TaskType,
    SystemPrompts,
)


@dataclass
class CrisisAssessment:
    """Crisis evaluation result"""
    risk_level: str  # 'high', 'medium', 'low'
    confidence: float  # 0.0-1.0
    detected_signals: List[str]
    immediate_actions: List[str]
    reasoning: str
    timestamp: datetime


class CrisisDetector:
    """
    Crisis Detector
    Multi-layer detection mechanism: Keywords + rule + LLM
    """

    # Crisis keyword dictionary
    CRISIS_KEYWORDS = {
        'suicide': {
            'keywords': [
                'suicide', '轻生', '不想活', '活着没意思', '结束生命',
                '自杀', '一了百了', '解脱', '死了算了',
            ],
            'severity': 'high',
        },
        'self_harm': {
            'keywords': [
                'self-harm', '割腕', '伤害自己', '划伤', '烫伤自己',
                '撞墙', '打自己',
            ],
            'severity': 'high',
        },
        'method_planning': {
            'keywords': [
                '跳楼', '上吊', '安眠药', '割腕刀', '煤气',
                '最痛苦死法', '怎么死', '自杀方法',
            ],
            'severity': 'high',
        },
        'hopelessness': {
            'keywords': [
                '没有希望', '绝望', '无法忍受', '痛不欲生', '走投无路',
                '无能为力', '撑不下去',
            ],
            'severity': 'medium',
        },
        'severe_depression': {
            'keywords': [
                '每天都想死', '活着太累', '生命毫无意义', '留恋',
                '空虚', '麻木', '失去感觉',
            ],
            'severity': 'medium',
        },
    }

    # Protective factors (lower Risk)
    PROTECTIVE_FACTORS = [
        '家人', '朋友', '孩子', '宠物', '责任', '寻求帮助',
        '咨询', '治疗', '药物',
    ]

    def __init__(self):
        self.llm = get_orchestrator()

    async def assess_risk(self, message: str, user_id: str) -> Dict[str, Any]:
        """
        Evaluate user message crisis Risk

        Args:
            message: user message
            user_id: User ID

        Returns:
            Crisis evaluation result
        """
        # First layer: Keyword Rapid screening
        keyword_result = self._keyword_screening(message)

        # Second layer: LLM depth analysis
        if keyword_result['initial_risk'] in ['high', 'medium']:
            llm_result = await self._llm_assessment(message, keyword_result)
            assessment = self._combine_assessments(keyword_result, llm_result)
        else:
            # Low Risk, Skip LLM analysis to save cost
            assessment = {
                'risk_level': 'low',
                'confidence': keyword_result['confidence'],
                'signals': keyword_result['detected_keywords'],
                'immediate_action': 'continue_monitoring',
            }

        # Record evaluation result (for later analysis)
        self._log_assessment(user_id, message, assessment)

        return assessment

    def _keyword_screening(self, message: str) -> Dict[str, Any]:
        """
        Keyword screening (First layer)
        Fast, low cost initial screening
        """
        message_lower = message.lower()
        detected = []
        max_severity = 'low'

        # Detect crisis Keywords
        for category, info in self.CRISIS_KEYWORDS.items():
            for keyword in info['keywords']:
                if keyword in message_lower:
                    detected.append({
                        'keyword': keyword,
                        'category': category,
                        'severity': info['severity'],
                    })
                    if info['severity'] == 'high':
                        max_severity = 'high'
                    elif info['severity'] == 'medium' and max_severity != 'high':
                        max_severity = 'medium'

        # Detect Protective factors
        protective_present = any(factor in message_lower for factor in self.PROTECTIVE_FACTORS)

        # Calculate confidence
        confidence = len(detected) * 0.2 if detected else 0.1
        confidence = min(confidence, 0.9)  # Keyword at most 0.9 confidence

        # If have Protective factors, lower Risk level
        if protective_present and max_severity == 'high':
            max_severity = 'medium'

        return {
            'initial_risk': max_severity,
            'detected_keywords': detected,
            'protective_factors': protective_present,
            'confidence': confidence,
        }

    async def _llm_assessment(
        self,
        message: str,
        keyword_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM depth evaluation (Second layer)
        Understanding context and intent
        """
        detected_kw = ', '.join([d['keyword'] for d in keyword_result['detected_keywords']])

        prompt = f"""
User message: "{message}"

Initial detected Risk signals: {detected_kw}

Please as psychological crisis evaluation expert, depth analysis:

1. **True actual Risk level** (consider context)
   - high: Immediate suicide/self-harm Risk, have specific plan or intent
   - medium: Severe suicidal ideation but no specific plan, or severe Emotional distress
   - low: General emotional expression, no clear Risk

2. **Detected Specific signals**
   - List key Risk metrics

3. **Immediate action recommendation**
   - high Risk: immediate intervention, provide hotline, recommend emergency
   - medium Risk: Strongly recommend professional help, close monitoring
   - low Risk: continued supportive conversation

4. **Evaluation reasoning**
   - Brief explanation of determination basis

Output JSON format:
{{
    "risk_level": "high|medium|low",
    "confidence": 0.0-1.0,
    "detected_signals": ["signal1", "signal2"],
    "immediate_actions": ["action1", "action2"],
    "reasoning": "evaluation reasoning",
    "context_notes": "context explanation"
}}
"""

        config = ModelRouter.get_model_config(TaskType.CRISIS_DETECTION)

        result = await self.llm.generate_structured(
            prompt=prompt,
            config=config,
            system_prompt=SystemPrompts.CRISIS_DETECTOR,
        )

        return result

    def _combine_assessments(
        self,
        keyword_result: Dict[str, Any],
        llm_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize Keyword and LLM evaluation results
        """
        # LLM result as main, but not lower than Keyword evaluation
        risk_hierarchy = {'low': 0, 'medium': 1, 'high': 2}

        keyword_risk = keyword_result['initial_risk']
        llm_risk = llm_result.get('risk_level', 'medium')

        # Take relatively high Risk level (Safety first)
        if risk_hierarchy[llm_risk] >= risk_hierarchy[keyword_risk]:
            final_risk = llm_risk
        else:
            final_risk = keyword_risk

        return {
            'risk_level': final_risk,
            'confidence': llm_result.get('confidence', 0.7),
            'signals': llm_result.get('detected_signals', []),
            'immediate_action': llm_result.get('immediate_actions', []),
            'reasoning': llm_result.get('reasoning', ''),
            'keyword_screening': keyword_result,
            'llm_analysis': llm_result,
        }

    def _log_assessment(self, user_id: str, message: str, assessment: Dict[str, Any]):
        """
        Record evaluation result (used for audit and improvement)
        Actual should write to database
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'message_preview': message[:50] + '...' if len(message) > 50 else message,
            'risk_level': assessment['risk_level'],
            'confidence': assessment.get('confidence'),
        }

        # TODO: write log database
        print(f"[CRISIS LOG] {log_entry}")

    async def generate_crisis_response(self, assessment: Dict[str, Any]) -> str:
        """
        Generate Crisis intervention Response according to evaluation result
        """
        if assessment['risk_level'] == 'high':
            return self._high_risk_response()
        elif assessment['risk_level'] == 'medium':
            return await self._medium_risk_response(assessment)
        else:
            return ""  # Low Risk no special Response required

    def _high_risk_response(self) -> str:
        """High Risk standard Response"""
        return """
I notice you are currently dealing with very great suffering. Your life is very important, I hope you can get professional help.

**Please immediately take the following actions:**

**24-hour psychological crisis hotline**
- National psychological aid hotline: 400-161-9995
- Beijing psychological Crisis intervention hotline: 010-82951332
- Shanghai psychological aid hotline: 021-62785488

**If there is immediate danger**
- Call 120 emergency
- Go to nearest hospital emergency
- Contact family members or trusted friends

I will always be here to accompany you, but professional psychological counselors and doctors can give you better help. Please tell me, are you safe now? Is there someone around you who can help?
"""

    async def _medium_risk_response(self, assessment: Dict[str, Any]) -> str:
        """Medium Risk personalized Response"""
        config = ModelRouter.get_model_config(TaskType.CRISIS_DETECTION)

        prompt = f"""
User is in medium psychological Risk state:

Risk signals: {', '.join(assessment.get('signals', []))}

Please generate Warm, supportive Response:
1. Confirm and validate user feelings
2. Express concern and support
3. Warmly but firmly recommend seeking professional help
4. Provide psychological counseling resources
5. Ask about current support system (family, friends)

Tone: Warm, empathy, Non-judgmental, stable
Length: 2-3 paragraphs
"""

        response = await self.llm.generate(prompt=prompt, config=config)

        # Add resource info
        response += "\n\n**Mental health resources**\n"
        response += "- Psychological counseling appointment: can search local psychological counseling institutions\n"
        response += "- Psychological aid hotline: 400-161-9995 (24 hours)"

        return response
