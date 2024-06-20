from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from civirank import analyzers, parsers, rankers
from ranking_challenge.request import RankingRequest
from ranking_challenge.response import RankingResponse
import argparse
import uvicorn

app = FastAPI(
    title="Ranking Challenge",
    description="Ranks input using a local ranker.",
    version="0.1",
)

# Initialize the ranker instance
ranker = rankers.LocalRanker()

@app.post('/rank')
def rank(ranking_request: RankingRequest) -> RankingResponse:
    ranked_results, new_items = ranker.rank(ranking_request, batch_size=args.batch_size, scroll_warning_limit=args.scroll_warning_limit)
    return {"ranked_ids": ranked_results, "new_items": new_items}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranking Challenge')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--scroll_warning_limit', type=float, default=-0.1, help='Scroll warning limit')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    uvicorn.run(app, host='0.0.0.0', port=args.port)
