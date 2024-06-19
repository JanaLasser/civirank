from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from civirank import analyzers, parsers, rankers
from ranking_challenge.request import RankingRequest
from ranking_challenge.response import RankingResponse

app = FastAPI(
    title="Ranking Challenge",
    description="Ranks input using a local ranker.",
    version="0.1.0",
)

# Initialize the ranker instance
ranker = rankers.LocalRanker()

@app.post('/rank')
def rank(ranking_request: RankingRequest) -> RankingResponse:
    
    ranked_results, new_items = ranker.rank(ranking_request)
    return {"ranked_ids": ranked_results, "new_items": new_items}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
