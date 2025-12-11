import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.resources.chat.completions import AsyncIterable
from aidial_client.types.chat import ChatCompletionChunk, ChatCompletionResponse
from aidial_sdk.chat_completion import Choice, Message, Request, Role, Stage
from coordination.gpa import GPAGateway
from coordination.ums_agent import UMSAgentGateway
from logging_config import get_logger
from models import AgentName, CoordinationRequest
from prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:
    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version="2025-01-01-preview",
        )

        stage = StageProcessor.open_stage(choice, "Coordination request")

        coordination_request: CoordinationRequest = (
            await self.__prepare_coordination_request(client, request)
        )
        stage.append_content(f"**Coordination request**: {coordination_request}")

        StageProcessor.close_stage_safely(stage)

        stage = StageProcessor.open_stage(choice, coordination_request.agent_name)

        msg = await self.__handle_coordination_request(
            coordination_request, choice, stage, request
        )

        StageProcessor.close_stage_safely(stage)

        return await self.__final_response(client, choice, request, msg)

    async def __prepare_coordination_request(
        self, client: AsyncDial, request: Request
    ) -> CoordinationRequest:
        prep_messages = self.__prepare_messages(
            request, COORDINATION_REQUEST_SYSTEM_PROMPT
        )

        response: ChatCompletionResponse = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=prep_messages,
            stream=False,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequest.model_json_schema(),
                    },
                }
            },
        )

        content = json.loads(response.choices[0].message.content)

        return CoordinationRequest.model_validate(content)

    def __prepare_messages(
        self, request: Request, system_prompt: str
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": Role.SYSTEM, "content": system_prompt}
        ]

        for msg in request.messages:
            if msg.role == Role.USER:
                messages.append({"role": Role.USER, "content": msg.content})
            else:
                messages.append(msg.dict(exclude_none=True))

        return messages

    async def __handle_coordination_request(
        self,
        coordination_request: CoordinationRequest,
        choice: Choice,
        stage: Stage,
        request: Request,
    ) -> Message:
        if coordination_request.agent_name == AgentName.GPA:
            return await GPAGateway(self.endpoint).response(
                choice=choice,
                stage=stage,
                request=request,
                additional_instructions=coordination_request.additional_instructions,
            )
        else:
            return await UMSAgentGateway(self.ums_agent_endpoint).response(
                choice=choice,
                stage=stage,
                request=request,
                additional_instructions=coordination_request.additional_instructions,
            )

    async def __final_response(
        self,
        client: AsyncDial,
        choice: Choice,
        request: Request,
        agent_message: Message,
    ) -> Message:
        prep_messages = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)

        augmented_promt = f"**CONTEXT**: {agent_message}, **USER REQUEST**: {prep_messages[-1]['content']}"
        prep_messages[-1]["content"] = augmented_promt

        chunks: AsyncIterable[
            ChatCompletionChunk
        ] = await client.chat.completions.create(
            messages=prep_messages,
            deployment_name=self.deployment_name,
            stream=True,
        )

        content = ""

        async for chunk in chunks:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    content += delta.content
                    choice.append_content(delta.content)

        return Message(
            role=Role.ASSISTANT,
            content=content,
        )
