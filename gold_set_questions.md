## Direct-answer (no tool) cases

1. “Tell me a joke.”
2. “Who are you?”
3. “Say ‘good morning’ in French.”
4. “Explain what a speech-to-speech model is in simple termsUUsed
5. “Talk to me like a friendly bank teller.”
6. “Repeat after me: I will finish this demo.”
7. “What can you do?”
8. “Summarize our last conversation in one sentence.”

These should generally yield `tool_required: false`, `tool: null`, `response_mode: "direct"`.

## Tool-required cases (clear)

9. “What’s today’s date?”
10. “What time is it right now?”
11. “What’s the weather in Abuja at the moment?”
12. “What’s my current account balance?”
13. “Convert 100 US dollars to naira.”
14. “Set a reminder for tomorrow at 9 AM to call my bank.”
15. “Check the exchange rate for euro to naira today.”

These should generally yield `tool_required: true`, specific `tool` names (e.g. `"get_current_date"`, `"get_weather"`, `"get_account_balance"`), non-empty `args`, and `response_mode: "hold_then_answer"`.

## Borderline / context-dependent cases

16. “I think I’m broke, how much do I have in my account?”
17. “Is it raining today?”
18. “Can you help me plan my budget for this month?”
19. “Remind me next week to pay my electricity bill.”
20. “If I save 50,000 naira every month, how much will I have in a year?”
