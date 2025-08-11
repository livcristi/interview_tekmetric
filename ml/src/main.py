import logging
from pathlib import Path

import uvicorn

from core.config import load_config
from src.app import create_app

if __name__ == "__main__":
    # Set-up logging
    logging.basicConfig(level=logging.INFO)

    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)

    # Start the FastAPI app
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
    )
