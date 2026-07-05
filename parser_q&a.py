from openai import OpenAI
from collections import defaultdict
import json
with open("knowledge_base.txt","r") as f:
    knowledge_base = f.read()
#----------------------------------------
chunks = knowledge_base.split("==============================")
chunks_new = (chunks[2].strip()).split("----------------------------------------")
print(chunks_new)
with open("chunks.txt","w") as p:
    p.writelines(chunks_new)
with open("chunks.txt","r") as q:
    print(q.read()) 
d = defaultdict(dict)
i = 1
for x in chunks_new:
    user = OpenAI()
    response = user.embeddings.create(
    input = x,
    model="text-embedding-3-small"
    )
    d[f"chunk_{i}"]["statement"] = x
    d[f"chunk_{i}"]["embedding"] = response.data[0].embedding
    i += 1
embedding_data = dict(d)
with open("chunks_data.json","w") as f:
    json.dump(embedding_data,f,indent=4)




