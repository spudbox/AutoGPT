# Integrating ChatGPT Agent and Codex in Custom Blocks

This guide shows how to wrap OpenAI's ChatGPT Agent APIs and the Codex code-generation APIs inside a reusable AutoGPT block. The same pattern works whether you expose the models directly to end users or chain them between other role blocks (for example, routing Business Analyst output into a Codex code stubber and then handing that to a Developer block).

## Prerequisites
- AutoGPT Platform running locally or in the cloud with access to the block SDK.
- An OpenAI API key with access to both ChatGPT (Assistants / Responses APIs) and the `gpt-*` / `gpt-4o-mini` style Codex-compatible models.
- `openai` Python package available in your backend environment (installable through `poetry add openai` when developing backend blocks).

## 1. Define a Shared OpenAI Provider
Create `chatgpt_codex/_config.py` for your block package and register the OpenAI provider once so every block instance can reuse it:

```python
from backend.sdk import BlockCostType, ProviderBuilder

openai_provider = (
    ProviderBuilder("openai")
    .with_api_key("OPENAI_API_KEY", "OpenAI API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
```

This exposes an **API Key** credential field in the builder UI and makes the provider discoverable by other blocks in the same package.

## 2. Create the Block Skeleton
Start a new block file such as `chatgpt_codex/block.py` and wire up the credential + request configuration. The example below supports both conversational (ChatGPT) and code-generation (Codex) flows by flipping a mode input:

```python
import uuid
from typing import Literal

from openai import AsyncOpenAI

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import openai_provider


class ChatGPTCodexBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = openai_provider.credentials_field(
            description="OpenAI credentials"
        )
        mode: Literal["chat", "code"] = SchemaField(
            description="Choose ChatGPT (chat) or Codex (code) behaviour",
            default="chat",
        )
        system_prompt: str = SchemaField(
            description="System prompt to steer the model",
            default="You are a helpful assistant.",
        )
        user_prompt: str = SchemaField(description="Primary input text")
        temperature: float = SchemaField(default=0.2, ge=0, le=2)

    class Output(BlockSchema):
        response: str = SchemaField(description="Model output text")
        error: str = SchemaField(description="Failure reason, if any")

    def __init__(self) -> None:
        super().__init__(
            id=str(uuid.uuid4()),
            name="ChatGPT + Codex",
            description="Call ChatGPT agents or Codex models from one block",
            categories={BlockCategory.AI, BlockCategory.PRODUCTIVITY},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **_: object,
    ) -> BlockOutput:
        client = AsyncOpenAI(api_key=credentials.api_key.get_secret_value())

        try:
            if input_data.mode == "chat":
                result = await client.responses.create(
                    model="gpt-4.1-mini",  # ChatGPT Agent API
                    input=[
                        {
                            "role": "system",
                            "content": input_data.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": input_data.user_prompt,
                        },
                    ],
                    temperature=input_data.temperature,
                )
                yield "response", result.output_text  # condensed helper

            else:
                completion = await client.responses.create(
                    model="gpt-4o-mini",  # Codex-compatible code model
                    input=[
                        {
                            "role": "system",
                            "content": input_data.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": input_data.user_prompt,
                        },
                    ],
                    temperature=input_data.temperature,
                )
                yield "response", completion.output_text

        except Exception as exc:  # surface errors to downstream blocks
            yield "error", str(exc)
```

### Why use the Responses API?
The OpenAI Responses API works for both conversational agents (ChatGPT) and the current Codex-style models, simplifying maintenance and letting you share the same message formatting logic.

## 3. Publish the Block
1. Add an `__init__.py` that exposes the block class: `__all__ = ["ChatGPTCodexBlock"]`.
2. Run the backend in dev mode and ensure the package is discoverable under **Blocks → Custom**.
3. In the visual builder, drag the **ChatGPT + Codex** block into your workflow, supply the OpenAI API key, and choose the mode per role hand-off.

## 4. Chain Role Agents Together
- **Project Manager → ChatGPT:** Feed the PM charter brief into the block with `mode="chat"` to generate stakeholder updates.
- **Business Analyst → Codex:** Pass summarized requirements into the same block set to `mode="code"` to produce boilerplate test harnesses or schema definitions.
- **Developer QA loop:** Branch outputs into review blocks; you can even store both the raw response and the error field in a dictionary for auditing.

## 5. Testing Tips
- Use the block's `test_input` configuration to store sample prompts for each mode so collaborators can validate the integration quickly.
- Mock the `AsyncOpenAI` client in unit tests to avoid real API calls while verifying that the block yields the correct outputs for success and error paths.

With this pattern you can embed ChatGPT Agent reasoning and Codex-style code generation seamlessly inside larger AutoGPT automations, keeping the same provider credentials and block interface.
