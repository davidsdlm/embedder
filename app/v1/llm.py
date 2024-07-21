import pathlib
import os

from FlagEmbedding import BGEM3FlagModel

parent_directory = pathlib.Path(__file__).parent.resolve().parent.parent
llm_path = os.path.join(parent_directory, "llm_slim")

model = BGEM3FlagModel(llm_path)
