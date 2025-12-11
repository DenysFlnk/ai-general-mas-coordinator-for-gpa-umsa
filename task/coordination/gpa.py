from copy import deepcopy
from typing import Any, Optional

from aidial_client import AsyncDial
from aidial_client.resources.chat.completions import AsyncIterable
from aidial_client.types.chat.response import ChatCompletionChunk
from aidial_sdk.chat_completion import (
    Attachment,
    Choice,
    CustomContent,
    Message,
    Request,
    Role,
    Stage,
)
from stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"


class GPAGateway:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def response(
        self,
        choice: Choice,
        stage: Stage,
        request: Request,
        additional_instructions: Optional[str],
    ) -> Message:
        client = AsyncDial(
            api_version="2025-01-01-preview",
            api_key=request.api_key,
            base_url=self.endpoint,
        )

        response: AsyncIterable[
            ChatCompletionChunk
        ] = await client.chat.completions.create(
            deployment_name="general-purpose-agent",
            messages=self.__prepare_gpa_messages(
                request=request, additional_instructions=additional_instructions
            ),
            stream=True,
            extra_headers={
                "x-conversation-id": request.headers.get("x-conversation-id")
            },
        )

        content = ""
        custom_content: CustomContent = CustomContent()
        custom_content.attachments = []
        stages_map: dict[int, Stage] = {}

        async for chunk in response:
            if chunk.choices and chunk.choices[0]:
                delta = chunk.choices[0].delta
                print(f"GPA delta: {delta}")

                if delta.content:
                    stage.append_content(delta.content)
                    content += delta.content

                if delta_custom_content := delta.custom_content:
                    print(f"GPA custom content: {delta_custom_content}")

                    if delta_custom_content.attachments:
                        custom_content.attachments.extend(
                            delta_custom_content.attachments
                        )

                    if delta_custom_content.state:
                        custom_content.state = delta_custom_content.state

                    custom_content_dict = custom_content.dict(exclude_none=True)

                    if stages := custom_content_dict.get("stages"):
                        for stg in stages:
                            idx = stg["index"]

                            if existing_stage := stages_map.get(idx):
                                if stg.get("content"):
                                    existing_stage.append_content(stg["content"])

                                if attachments := stg.get("attachments"):
                                    for attachment in attachments:
                                        existing_stage.add_attachment(attachment)

                                if (
                                    stg.get("status")
                                    and stg.get("status") == "completed"
                                ):
                                    StageProcessor.close_stage_safely(existing_stage)
                            else:
                                stages_map[idx] = StageProcessor.open_stage(
                                    choice, stg.get("name")
                                )

        for stg in stages_map.values():
            StageProcessor.close_stage_safely(stg)

        for attachment in custom_content.attachments:
            choice.add_attachment(Attachment(**attachment.dict(exclude_none=True)))

        choice.set_state(
            {
                _IS_GPA: True,
                _GPA_MESSAGES: custom_content.state,
            }
        )

        return Message(
            role=Role.ASSISTANT,
            content=content,
        )

    def __prepare_gpa_messages(
        self, request: Request, additional_instructions: Optional[str]
    ) -> list[dict[str, Any]]:
        res_messages = []

        for idx in range(len(request.messages)):
            message = request.messages[idx]

            if message.role == Role.ASSISTANT:
                if message.custom_content and message.custom_content.state:
                    state = message.custom_content.state
                    if state.get(_IS_GPA):
                        res_messages.append(
                            request.messages[idx - 1].dict(exclude_none=True)
                        )
                        copy_msg = deepcopy(message)
                        copy_msg.custom_content.state = state.get(_GPA_MESSAGES)
                        res_messages.append(copy_msg.dict(exclude_none=True))

        user_msg = request.messages[-1]
        custom_content = user_msg.custom_content
        if additional_instructions:
            res_messages.append(
                {
                    "role": Role.USER,
                    "content": f"{user_msg.content}\n\n{additional_instructions}",
                    "custom_content": custom_content.dict(exclude_none=True)
                    if custom_content
                    else None,
                }
            )
        else:
            res_messages.append(user_msg.dict(exclude_none=True))

        return res_messages
