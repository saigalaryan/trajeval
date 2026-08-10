# Judge clients

::: trajeval.judge.client.JudgeClient
::: trajeval.judge.client.AnthropicJudgeClient
::: trajeval.judge.client.OpenAIJudgeClient
::: trajeval.judge.client.FakeJudgeClient
::: trajeval.judge.client.JudgeParseError
::: trajeval.judge.client.extract_json

## Response caching

Judge responses are cached on disk by default
(`TrajevalConfig.judge_cache_path`) — plaintext, and gitignored for exactly
that reason. See [Configuration](config.md).

::: trajeval.judge.cache.JudgeCache
::: trajeval.judge.cache.cache_key
