from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional


class State(BaseModel):
    language: Optional[str] = None
    messages: Optional[Annotated[Sequence[BaseMessage], add_messages]] = []  # UPDATES ARE APPENDED
    question: str
    documents: Optional[str] = None

    

app = FastAPI()
app.mount("/public", StaticFiles(directory="public"), name="public")
# Thread Pool for Async Execution
executor = ThreadPoolExecutor(max_workers=12)

@app.post("/chat/{thread_id}")
async def chat2(thread_id: str, state: State, background_tasks: BackgroundTasks,
                llm_app: StateGraph = Depends(get_llm_app)):
    logger.info(f"Received request for thread_id: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}
    response = await run_inference(llm_app, state, config)

    if "messages" in response and len(response["messages"]) > 0:
        latest_message = response["messages"][-1]

        if isinstance(latest_message, AIMessage):
            reasoning, answer = parse_response(latest_message.content)
            output = {"thought_process": reasoning, "answer": answer}
        else:
            output = {"thought_process": "", "answer": 'Error: No AI response found'}

    else:
        output = {"thought_process": "", "answer": 'Error: No message in response'}

    logger.debug(f"Response for thread_id {thread_id}: {response}")

    return output

uvicorn.run(app, host="0.0.0.0", port=8000) # run the app



"""

so i need to get the api to stream the response from the llm



"""