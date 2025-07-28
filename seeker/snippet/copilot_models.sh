#date: 2025-07-28T17:06:24Z
#url: https://api.github.com/gists/c1462adde8aef23182075c96d0474271
#owner: https://api.github.com/users/sr-tream

#!/usr/bin/bash

if [ -n "$JSON_OUTPUT" ]; then
    PRINTER=(jq -r '
    .model_list | map({
        (.litellm_params.model): {
        litellm_params: {
            rpm: .litellm_params.rpm
        },
        model_info: (.model_info)
        }
    }) | add
    ')
else
    PRINTER=(yq -P)
    echo "For LiteLLM Proxy server you can define JSON_OUTPUT env to get simplified json outputs"
    echo ""
    echo ""
fi

if [ -z "$COPILOT_API_KEY" ]; then
    echo "Use COPILOT_API_KEY env to set API key"
    exit 1
fi

curl -s https://api.githubcopilot.com/models \
-H "Authorization: Bearer ${COPILOT_API_KEY}" \
-H "Content-Type: application/json" \
-H "Copilot-Integration-Id: vscode-chat" | \
jq -r '
{
credential_list: [{
  credential_name: "GH Copilot",
  credential_values: {
    api_key: "os.environ/COPILOT_API_KEY",
    api_base: "https://api.githubcopilot.com"
  },
  credential_info: {
    description: "Credential for models provided by Github Copilot chat."
}
}],
model_list: [
.data[] | select(.model_picker_enabled) | {
  model_name: (.id),
  litellm_params: {
    model: "openai/"+(.id),
    litellm_credential_name: "GH Copilot",
    custom_llm_provider: "openai",
    rpm: 14
  },
  model_info: (
    (.capabilities.limits | {max_input_tokens: "**********": .max_prompt_tokens}) +
    (.capabilities.supports | {supports_tool_choice: .tool_calls, supports_function_calling: .tool_calls, supports_vision: (.vision // false), supports_system_messages: true}) +
    (if (.id | contains("claude")) and (.id | contains("sonnet")) then {supports_computer_use: true} else {} end)
  )
}]}' | "${PRINTER[@]}"
rue} else {} end)
  )
}]}' | "${PRINTER[@]}"
