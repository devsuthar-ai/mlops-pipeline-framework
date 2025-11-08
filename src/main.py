from fastapi import FastAPI
import uvicorn

app = FastAPI(title="MLOps Pipeline Framework", version="1.0.0")

@app.get("/")
async def root():
    return {"service": "MLOps Pipeline", "status": "operational"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/pipelines/create")
async def create_pipeline():
    return {"pipeline_id": "pipeline_001", "status": "created"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
