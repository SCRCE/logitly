from logitly import DecisionModel

# Uses the existing pinned LFM alias; a HF ID or local directory also works.
with DecisionModel.from_pretrained("lfm") as engine:
    result = engine.choice(
        {"amount": 9700, "device": "unknown"},
        "What action is appropriate?",
        {"close": "Close as benign", "review": "Request analyst review", "block": "Block immediately"},
    )
    print(result)
