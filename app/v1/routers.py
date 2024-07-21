import structlog
from fastapi import FastAPI

from app.v1 import llm
from app.v1 import schemas

router = FastAPI()
logger = structlog.getLogger("router")


@router.post("/embed/")
async def text_to_embedding(data: schemas.EmbedderRequest):
    embedding = llm.model.encode(data.text, return_sparse=True)
    return schemas.EmbedderResponse(
        embedding=embedding["dense_vecs"],
        lexical_weights=llm.model.convert_id_to_token(embedding['lexical_weights'])
    )
