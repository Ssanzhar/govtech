Task: Design and Implement Real-Time Call Scam Detection Assistant
You are a principal mobile AI architect, security engineer, and product designer.

Your task is to design a production-grade system from scratch that assists users during potentially fraudulent phone calls.

Project Goal
Build an intelligent call assistant that listens to an ongoing phone conversation, analyzes it in real time, estimates scam probability, explains why the call is suspicious, and helps the user respond safely.

The assistant must act as a decision-support tool.

It must never automatically hang up, report, block, or take actions without explicit user consent.

Core User Experience
A user receives a phone call from an unknown number.

The assistant automatically activates (or activates according to configured rules).

During the call it continuously:

Listens to speech.
Generates live transcription.
Analyzes conversation context.
Detects social-engineering tactics.
Calculates a dynamic suspicion score.
Explains why the score changed.
Suggests safe responses or verification questions.
Offers a report flow after the call if scam indicators are strong.
The system should feel similar to a "fraud navigation assistant."

Main Feature Requirements
1. Real-Time Listening
The system must support realistic audio acquisition methods.

Evaluate and design:

Android speakerphone microphone capture
Android accessibility/telephony integrations
Secondary-device listening architecture
Future OS-level integrations
Any alternative architecture
For each approach explain:

feasibility
limitations
privacy implications
expected latency
MVP suitability
2. Live Transcription
The assistant must continuously convert speech to text.

Requirements:

streaming or pseudo-streaming
partial transcripts
speaker separation if possible
support for multilingual conversations
support for language switching
noisy environments
Explain:

architecture
models
inference pipeline
latency expectations
hardware requirements
3. Scam Detection Engine
The assistant must continuously estimate scam probability.

Output:

Suspicion Score: 0–100
The score must evolve during the conversation.

The model should detect:

urgency
authority impersonation
bank impersonation
police impersonation
government impersonation
OTP requests
verification code requests
money transfer requests
account recovery manipulation
secrecy requests
emotional pressure
investment scams
job scams
parcel scams
romance scams
technical-support scams
Design:

classification architecture
streaming inference strategy
rolling context handling
confidence calibration
thresholding strategy
false-positive mitigation
4. Suspicion Meter
Design a live risk meter.

Requirements:

Low Risk (0–30)
Display:

No obvious scam indicators.
Medium Risk (31–60)
Display:

Caution.
Explain detected warning signs.
High Risk (61–80)
Display:

Strong warning.
Highlight suspicious statements.
Critical Risk (81–100)
Display:

Immediate warning.
Explain the highest-risk indicators.
The score should:

rise gradually
avoid rapid oscillations
support hysteresis
support confidence weighting
support evidence accumulation
Explain the exact scoring methodology.

5. Explainability System
Every score increase must be explainable.

For every warning provide:

suspicious phrase
detected tactic
confidence
human-readable explanation
Examples:

Detected tactic:
Authority Impersonation

Evidence:
"Calling from bank security department"

Confidence:
87%
The explanation must be generated from actual model evidence.

No fabricated explanations.

6. User Guidance Engine
The assistant should actively help the user verify legitimacy.

Instead of only showing warnings, it should recommend actions.
Verification Questions
Ask for official case number.
Ask for employee ID.
Ask for callback through official website.
Ask them to send information through official channels.
Recommended Actions
End call and verify independently.
Contact organization directly.
Do not share OTP codes.
Do not install remote software.
Do not transfer funds.
Design a recommendation framework.

Explain:

recommendation generation
tactic-specific advice
confidence-based advice
localization support
7. Post-Call Analysis
When the call ends:

Generate a structured summary.

Example:

Final Suspicion Score: 84

Detected Tactics:
- Authority Impersonation
- Urgency
- OTP Request

Risk Level:
High

Recommended Action:
Do not engage further.
Contact organization through official channels.
8. Reporting Workflow
If suspicion exceeds a configurable threshold:

Offer the user a reporting option.

Requirements:

never automatic
user consent required
review before submission
editable report contents
Potential report contents:

number
transcript
suspicious phrases
detected tactics
timestamp
risk score
Design:

reporting UX
privacy model
consent flow
backend requirements
9. Privacy and Legal Requirements
The design must be legally realistic.

Explicitly analyze:

Android restrictions
iOS restrictions
recording consent laws
data retention
biometric implications
transcript storage
reporting requirements
Separate:

legally viable
legally risky
platform-impossible
10. Technical Architecture
Design a complete architecture.

Include:

Mobile Layer
audio capture
transcription
inference
UI
AI Layer
ASR
NLP
explainability
recommendation engine
Backend Layer
reporting
analytics
storage
Data Flow
Provide detailed flow diagrams.

11. MVP Definition
Assume a small team with 2–4 weeks.

Define:

Must Have
Minimum features required for a compelling demo.

Should Have
High-value improvements.

Nice to Have
Future roadmap items.

Expected Deliverables
Provide:

Executive Summary
Feasibility Assessment
Product Specification
System Architecture
AI Architecture
Scoring Framework
Recommendation Framework
Reporting Framework
Privacy & Legal Analysis
MVP Roadmap
Engineering Risks
Long-Term Roadmap
Critical Requirements
Be brutally realistic.

Do not ignore platform restrictions.

Distinguish clearly between:

possible today,
difficult but achievable,
unrealistic for MVP.
Optimize for real-world deployment, not research demos.

Assume this system may eventually be used by millions of users.