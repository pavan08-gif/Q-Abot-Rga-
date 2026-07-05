import json
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import heapq
with open("chunks_data.json","r") as p:
    y = json.load(p)
user = OpenAI()
user_input = input("Ask Away..:")
embedding_create = user.embeddings.create(
    input=user_input,
    model="text-embedding-3-small"
)
user_embedding = embedding_create.data[0].embedding
for x in y.values():
    x["score"] = cosine_similarity([x["embedding"]], [user_embedding])[0][0]
top_chunks = heapq.nlargest(3,y.values(),key = lambda x:x["score"])
statements = [x["statement"] for x in top_chunks]
print(statements)
response = user.responses.create(
    model="gpt-4o-mini",
    instructions="""You are a Q&A bot.
Answer ONLY using the provided context.
If the answer cannot be found in the context, reply:
I don't have enough information in the provided context.""",
    input=f"""
            context:
            {statements}
            question:
            {user_input}"""
)
print(response.output_text)


