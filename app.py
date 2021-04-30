import uvicorn
import os

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=int(os.environ.get('PORT', 8000)), log_level="info", reload=True)