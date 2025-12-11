import os

import uvicorn
from agent import MASCoordinator
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response
from logging_config import get_logger, setup_logging

DIAL_ENDPOINT = os.getenv("DIAL_ENDPOINT", "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
UMS_AGENT_ENDPOINT = os.getenv("UMS_AGENT_ENDPOINT", "http://localhost:8042")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

setup_logging(log_level=LOG_LEVEL)
logger = get_logger(__name__)


class MASCoordinatorApplication(ChatCompletion):
    async def chat_completion(self, request: Request, response: Response) -> None:
        with response.create_single_choice() as choice:
            coordinator = MASCoordinator(
                endpoint=DIAL_ENDPOINT,
                deployment_name=DEPLOYMENT_NAME,
                ums_agent_endpoint=UMS_AGENT_ENDPOINT,
            )

            await coordinator.handle_request(choice=choice, request=request)


dial_app = DIALApp()
mas_coordinator = MASCoordinatorApplication()
dial_app.add_chat_completion(deployment_name="mas-coordinator", impl=mas_coordinator)


if __name__ == "__main__":
    uvicorn.run(dial_app, port=8055, host="0.0.0.0")
