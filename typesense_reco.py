import typesense

client= typesense.Client({
    'nodes':[{
        'host':'localhost',
        'port': '8108',
        'protocol':'http'
    }],
    'api_key':'xyz',
    'connection_timeout_seconds':2
})

COLLECTION_NAME='faces'

def create_collection():
    try:
        client.collections[COLLECTION_NAME].retrieve()
    except:
        client.collections.create({
            "name": COLLECTION_NAME,
            "fields":[
                {"name":"id","type":"string"},
                {"name":"name", "type":"string"},
                {"name":"embedding","type":"float[]","num_dim":128}
            ]
        })

def store_faces(id,name,embedding):
    document={
        "id":id,
        "name":name,
        "embedding":embedding
        
    }
    client.collections[COLLECTION_NAME].documents.upsert(document)

def predict_using_typesense(embedding, k=3):
    try:
        search_parameters = {
            "q": "*",
            "vector_query": f"embedding:([{','.join(map(str, embedding))}], k:{k})",
            "query_by": "name",
        }

        search_results = client.collections[COLLECTION_NAME].documents.search(search_parameters)
        print(search_results)
        hits = search_results.get('hits', [])
        if not hits:
            return "UNKNOWN", []

        top_prediction = hits[0]['document']['name']

        top_k_matches = [
            {
                "name": hit['document']['name'],
                "distance": hit.get('vector_distance', 1.0)  
            }
            for hit in hits
        ]

        return top_prediction, top_k_matches
    except Exception as e:
        print(f"Typesense prediction error: {e}")
        return "UNKNOWN", []
