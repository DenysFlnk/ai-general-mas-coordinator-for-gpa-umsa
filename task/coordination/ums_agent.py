import json
from typing import Optional

import httpx
from aidial_sdk.chat_completion import Choice, Message, Request, Role, Stage

_UMS_CONVERSATION_ID = "ums_conversation_id"


class UMSAgentGateway:
    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint

    async def response(
        self,
        choice: Choice,
        stage: Stage,
        request: Request,
        additional_instructions: Optional[str],
    ) -> Message:
        ums_conversation_id = self.__get_ums_conversation_id(request)

        if not ums_conversation_id:
            ums_conversation_id = await self.__create_ums_conversation()
            choice.set_state({_UMS_CONVERSATION_ID: ums_conversation_id})

        user_message = request.messages[-1].content

        augmented_user_message = f"**ADDITIONAL INSTRUCTIONS**: {additional_instructions or ''}, **USER REQUEST**: {user_message}"

        assistant_response = await self.__call_ums_agent(
            conversation_id=ums_conversation_id,
            user_message=augmented_user_message,
            stage=stage,
        )

        return Message(role=Role.ASSISTANT, content=assistant_response)

    def __get_ums_conversation_id(self, request: Request) -> Optional[str]:
        """Extract UMS conversation ID from previous messages if it exists"""
        for msg in request.messages:
            custom_content = msg.custom_content
            if (
                custom_content
                and custom_content.state
                and custom_content.state.get(_UMS_CONVERSATION_ID)
            ):
                return custom_content.state.get(_UMS_CONVERSATION_ID)

        return None

    async def __create_ums_conversation(self) -> str:
        """Create a new conversation on UMS agent side"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=f"{self.ums_agent_endpoint}/conversations",
                json={"title": "UMS agent"},
            )
            response.raise_for_status()
            conversation_data = response.json()
            return conversation_data["id"]

    async def __call_ums_agent(
        self, conversation_id: str, user_message: str, stage: Stage
    ) -> str:
        """Call UMS agent and stream the response"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat",
                json={
                    "message": {"role": "user", "content": user_message},
                    "stream": True,
                },
            )
            response.raise_for_status()

            content = ""
            async for line in response.aiter_lines():
                if "[DONE]" in line:
                    break

                if line.startswith("data: "):
                    data = line[6:]

                    data_json = json.loads(data)

                    if "conversation_id" in data:
                        continue

                    if data_json.get("choices"):
                        delta = data_json["choices"][0].get("delta", {})
                        if delta_content := delta.get("content"):
                            stage.append_content(delta_content)
                            content += delta_content

            print(f"Content from UMS agent: {content}")

            return content
