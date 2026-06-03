from fastapi import FastAPI

from app.database import (
    Base,
    engine
)

from app.routers import (
    users,
    violations,
    
    ai,
    challans
)

# CREATE TABLES

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Traffic Enforcement System"
)

# ROUTERS

app.include_router(users.router)

app.include_router(violations.router)

app.include_router(ai.router)

app.include_router(challans.router)


@app.get("/")
def home():

    return {

        "message":
        "AI Traffic Enforcement System Running Successfully"
    }