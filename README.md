# GraphRAG

Graph-enhanced retrieval-augmented generation prototype for higher-context QA.

## Demo
![Demo](https://raw.githubusercontent.com/Pancake2021/Pancake2021/main/assets/cloud-drift.gif)

## How to run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Results
- Better multi-hop retrieval behavior versus naive chunk retrieval.
- Cleaner source grounding through graph connections.
- Modular pipeline suitable for experimentation.
