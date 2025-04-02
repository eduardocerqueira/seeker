#date: 2025-04-02T17:02:31Z
#url: https://api.github.com/gists/537a593aa592831e309112fa22cc85ec
#owner: https://api.github.com/users/av

curl --request POST \
  --url https://api.openai.com/v1/chat/completions \
  --header 'Accept: */*' \
  --header 'Authorization: Bearer <API KEY>' \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: httpyac' \
  --data '{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "Random nonce: 272465"
    },
    {
      "role": "user",
      "content": "Are candles getting taller or shorter when they burn?"
    },
    {
      "role": "assistant",
      "content": "When candles burn, they get **shorter** over time. This happens because the wax acts as fuel for the flame, and as the wax melts and vaporizes, the candle gradually decreases in height."
    },
    {
      "role": "user",
      "content": "Are you sure? Will you be able to recognize this fact in different circumstances?"
    },
    {
      "role": "assistant",
      "content": "Yes, I'\''m sure that candles get **shorter** as they burn. Here'\''s why this holds true universally, regardless of how the question is phrased or the circumstances described:\n\n- **How candles work:** The heat from the candle'\''s flame melts the solid wax near the wick, which then gets drawn up through the wick to fuel the flame. The burning process converts the wax into heat, light, water vapor, and carbon dioxide. As this happens, the candle'\''s solid wax diminishes, causing it to grow shorter.\n\nSo in all scenarios where a candle is burning normally, the height of the candle will reduce over time, and it won'\''t get taller. Even if the question is rephrased or asked indirectly, the underlying principle remains the same: burning a candle consumes its wax, making it shorter. Let me know if you'\''d like further clarification!"
    },
    {
      "role": "user",
      "content": "Now, consider what you said above and solve the following riddle: I'\''m tall when I'\''m young, and I'\''m taller when I'\''m old. What am I?"
    }
  ]
}'