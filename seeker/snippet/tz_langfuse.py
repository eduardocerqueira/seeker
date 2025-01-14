#date: 2025-01-14T16:56:31Z
#url: https://api.github.com/gists/d2243a658eadcd7c95cf10961c3ac09a
#owner: https://api.github.com/users/nikcaryo-super

def get_content(chunk: Any) -> str:
    if hasattr(chunk, "raw"):
        return chunk.raw
    elif hasattr(chunk, "content") and chunk.content:
        return chunk.content[0].text
    else:
        return ""
    
def get_env_tag():
    if os.environ.get("K_REVISION"):
        return "cloud-run"
    return os.environ.get("ENV", "unknown")

def get_inference_metadata(function_name: str, inference_response_or_chunk: Any, tz_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    tags = tz_kwargs.get("tags") or {}
    tags["env"] = get_env_tag()
    
    # TZ Tags are key, value pa 
    inference_metadata = {
        "episode_id": inference_response_or_chunk.episode_id if hasattr(inference_response_or_chunk, "episode_id") else None,
        "variant_name": inference_response_or_chunk.variant_name if hasattr(inference_response_or_chunk, "variant_name") else None,
        "inference_id": inference_response_or_chunk.inference_id if hasattr(inference_response_or_chunk, "inference_id") else None,
        "function_name": function_name,
        "tz_tags": tags,
    }
    for k, v in inference_metadata.items():
        if "_id" in k:
            inference_metadata[k] = str(v)
    return inference_metadata

def update_langfuse_observation_and_create_generation(
    function_name: str, 
    inference_response_or_chunk: Any, 
    tz_kwargs: Dict[str, Any], 
    t_start: datetime, 
    t_end: datetime, 
    t_first_chunk: datetime | None, 
    usage: Usage, 
    chunks: List | None, 
    content: str, 
    is_streaming: bool
):
    inference_metadata = get_inference_metadata(function_name, inference_response_or_chunk, tz_kwargs)
    output = chunks if is_streaming else content
    observation_id = langfuse_context.get_current_observation_id()
    inference_name = f"inference_{function_name}"
    generation_name = f"generation_{function_name}"
    
    # Langfuse tags are a list of strings
    langfuse_tags = [f"{k}:{v}" for k, v in inference_metadata["tz_tags"].items()]
    langfuse_tags.append("tensorzero")
    langfuse_tags.append(f"tz_function:{function_name}")
    
    # Normal observation - this lets us recreate the inference call, with the typed inputs to TZ
    # Goal - recreate gateway.inference(...)
    langfuse_context.update_current_observation(
        start_time=t_start,
        end_time=t_end,
        output=output,
        name=inference_name,
        metadata=inference_metadata,
        tags=langfuse_tags
    )
    
    # Generation - this lets us see what was actually sent to the llm and what the llm generated
    # Goal - be able to call llm.generate(messages) with any llm
    tz_input = tz_kwargs["input"]
    generation_input_messages = []
    if tz_input.get("system"):
        if isinstance(tz_input["system"], str):
            system_content = tz_input["system"]
        else:
            system_content = json.dumps(tz_input["system"], indent=2)
        generation_input_messages.append({"role": "system", "content": system_content})
    if tz_input.get("messages"):
        generation_input_messages.extend(tz_input["messages"])
        
    generation_input_messages = [{"role": m["role"], "content": m["content"]} for m in generation_input_messages]

    generation_kwargs = {
        "name": generation_name,
        "start_time": t_start,
        "end_time": t_end,
        "input": generation_input_messages,
        "output": content,
        "metadata": inference_metadata,
        "tags": langfuse_tags
    }
    if t_first_chunk:
        generation_kwargs["completion_start_time"] = t_first_chunk
    if usage:
        generation_kwargs["usage"] = {"promptTokens": "**********": usage.output_tokens}
    
    
    # TODO: Fetch this from clickhouse to see exactly what's going on instead.
    
    langfuse_client.generation(
        trace_id=inference_metadata["tz_tags"].get("trace_id"),
        id=inference_metadata["inference_id"],
        parent_observation_id=observation_id,
        **generation_kwargs
    )

async def inference_generator_with_langfuse(inference_response: AsyncGenerator[InferenceChunk, None], function_name: str, tz_kwargs: Dict[str, Any], t_start: datetime) -> AsyncGenerator[InferenceChunk, None]:
    chunks = []
    t_first_chunk = None
    usage = None
    content = None
    last_chunk = None
    
    async for chunk in inference_response:
        if t_first_chunk is None:
            t_first_chunk = datetime.now()
            
        # Yield chunks as we go, do langfuse stuff once chunks are all streamed
        yield chunk
        chunks.append(chunk)
    t_end = datetime.now()
    content = "".join([get_content(chunk) for chunk in chunks])
    last_chunk = chunks[-1] if chunks else None
    if last_chunk and last_chunk.usage:
        usage = last_chunk.usage

    try:
        # The langfuse client itself is threaded so this shouldn't add noticeable latency
        langfuse_update_start = datetime.now()
        update_langfuse_observation_and_create_generation(
            function_name, 
            last_chunk, 
            tz_kwargs, 
            t_start, 
            t_end, 
            t_first_chunk, 
            usage, 
            chunks, 
            content, 
            is_streaming=True
        )
        langfuse_update_end = datetime.now()
        log.debug("Langfuse update took %s seconds", (langfuse_update_end - langfuse_update_start).total_seconds())
    except Exception as e:
        log.exception("Error updating langfuse observation and creating generation: %s", str(e))
                    
class AsyncTensorZeroLangfuseGateway(AsyncTensorZeroGateway):
    
    @observe(capture_output=False)
    async def inference(self, *args, **kwargs) -> Union[InferenceResponse, AsyncGenerator[InferenceChunk, None]]:
        t_start = datetime.now()
        inference_response = await super().inference(*args, **kwargs)
        function_name = kwargs["function_name"]
        is_streaming = isinstance(inference_response, AsyncGenerator)
        if not is_streaming:
            content = get_content(inference_response)
            usage = inference_response.usage
            t_end = datetime.now()
            try:
                update_langfuse_observation_and_create_generation(
                    function_name, 
                    inference_response, 
                    kwargs, 
                    t_start, 
                    t_end, 
                    None, 
                    usage, 
                    None, 
                    content, 
                    is_streaming=False
                )
            except Exception as e:
                log.exception("Error updating langfuse observation and creating generation: %s", str(e))
            return inference_response
        else:
            # Return a new generator that yield chunks as they come in, and updates langfuse afterwards
            return inference_generator_with_langfuse(inference_response, function_name, kwargs, t_start)
        
kwargs, t_start)
        
