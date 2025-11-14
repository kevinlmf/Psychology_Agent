# Psychology Agent
A LLM-based mental health assistant integrating **Agent architecture** and **RLHF (Reinforcement Learning from Human Feedback)**, continuously optimized through multimodal data analysis, personality-based personalization, and human feedback.

## LLM Method Improvements & Technical Innovations



## Architecture

### Core Components

```
User Input
  ↓
1. Mental State Interpreter → Understand user's emotional state
  ↓
2. Personality Analyzer → Analyze personality traits (after 3+ conversations)
  ↓
3. RLHF Reward Model → Get personalized weights (based on personality)
  ↓
4. LLM-Coach → Generate initial response
  ↓
5. Critic Agent → Evaluate response quality
  ↓
6. Refiner Agent → Optimize response (if needed)
  ↓
Final Response (High-quality, Personalized)
```

### Key Modules

1. **Mental State Interpreter**: Analyzes user's current psychological state
2. **Personality Analyzer**: Extracts Big Five traits and preferences from conversations
3. **RLHF Reward Model**: Personalized reward weights based on personality + feedback
4. **LLM-Coach**: Generates CBT/ACT-style therapeutic responses
5. **Critic Agent**: Quality assessment and routing decisions
6. **Refiner Agent**: Iterative response optimization

## Key Features Explained

### Quality Assurance Pipeline

Every response goes through:
1. **Initial Generation**: LLM-Coach generates response
2. **Quality Assessment**: Critic Agent evaluates quality
3. **Smart Routing**: 
   - High quality → Use directly
   - Medium quality → Refine
   - Low quality → Regenerate with multiple candidates
4. **Final Response**: Optimized, personalized, high-quality


### Personality-Based RLHF

The system automatically analyzes user personality after 3+ conversations and initializes personalized reward weights:

- **High Neuroticism** → More emotional stability support
- **Introverted** → More self-efficacy reinforcement
- **Problem-focused** → More practical advice
- **Emotion-focused** → More compassion and support




### Multi-Agent LLM Architecture Improvements

This system implements a multi-agent architecture following the **Record-Understand-Decide-Execute** paradigm:

**1. Record (Memory)**: Graph-Based State & Reasoning History
- Stores conversation history and mental state transitions
- Tracks user personality profile and preferences
- Maintains long-term memory for personalized responses

**2. Understand (Reasoning Core)**: Reasoner Agent (LLM-powered)
- Mental State Interpreter: Analyzes user's psychological state
- Personality Analyzer: Extracts personality traits from conversations
- Multi-step reasoning to understand user needs and context

**3. Decide (Controller)**: Critic Agent (Confidence-based routing)
- Evaluates response quality across 6 dimensions
- Makes routing decisions: use / refine / regenerate
- Confidence-based decision making for optimal response selection

**4. Execute (Tool Layer)**: Refiner Agent (Answer synthesis)
- Synthesizes final response based on Critic feedback
- Generates multiple candidate responses with different strategies
- Iteratively optimizes response quality

**Workflow:**
```
User Input
  → Record: Store in memory & retrieve context
  → Understand: Mental State + Personality Analysis (Reasoner)
  → Decide: Quality Assessment & Routing (Critic)
  → Execute: Response Generation & Refinement (Refiner)
  → Final Response
```

 



## Project Structure

```
Agent_psychology_assistant/
├── agent/                          # Agent core modules
│   ├── conversation_manager.py     # Main conversation orchestrator
│   ├── mental_state_interpreter.py # Psychological state analysis
│   ├── llm_coach.py                # CBT/ACT response generation
│   ├── personality_analyzer.py     # Personality analysis
│   ├── critic_agent.py            # Quality assessment
│   ├── refiner_agent.py            # Response optimization
│   └── memory_system.py            # User profile & memory
├── models/                         # LLM wrapper
│   ├── llm_configs.py              # Model configuration
│   └── llm_orchestrator.py         # Unified LLM API
├── rlhf/                           # RLHF modules
│   ├── personalized_reward_model.py # Personality-based RLHF
│   ├── reward_model.py             # Base reward model
│   └── feedback_collector.py       # Feedback collection
├── safety/                         # Safety module
│   └── crisis_detection.py         # Crisis detection
├── llm_analysis/                   # Analysis modules
│   └── behavior_analyzer.py        # Behavior pattern analysis
├── data_collection/                # Data collection
│   └── search_log_processor.py    # Search log processing
├── main.py                         # Main application entry
└── requirements.txt                # Dependencies
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/kevinlmf/Psychology_Agent
cd Psychology_Agent

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

**Important: Never commit your `.env` file to git!**

Create a `.env` file and add your API keys:

```bash
# Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
# Optional:
# OPENAI_API_KEY=sk-your-openai-key-here
EOF
```

Or manually create `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

Get your Anthropic API key from: https://console.anthropic.com/settings/keys


### 4. Run Application

```bash
python main.py
```

Select mode:
- **Mode 1**: Interactive conversation (full experience with enhanced architecture)
- **Mode 2**: Demo basic conversation features
- **Mode 3**: Demo behavior analysis
- **Mode 4**: Demo crisis detection
- **Mode 5**: Demo how to be happier




## Configuration

### Model Configuration

Models are configured in `models/llm_configs.py`. Currently using:
- `claude-3-7-sonnet-20250219` for most tasks
- Configurable per task type (crisis detection, casual chat, behavior analysis, etc.)

### RLHF Configuration

RLHF weights are personalized per user and stored in:
- `psychology_agent/data/rlhf/{user_id}_reward_weights.json`

### Personality Analysis

Personality profiles are stored in user profiles:
- `psychology_agent/data/user_profiles/{user_id}_profile.json`
---
##  Disclaimer

This system is for **research and educational purposes only**.  
It is **not** a medical device and must not be used for diagnosis or treatment.  
Consult professionals for real medical issues.
---
Happy Happy Happy😊
