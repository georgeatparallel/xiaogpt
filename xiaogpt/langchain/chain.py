from __future__ import annotations

import os
from uuid import uuid4

from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import LLMMathChain
from langchain_classic.base_memory import BaseMemory
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper

from xiaogpt.langchain.parallel import ParallelSearch


def search_tools(
    provider: str,
    serpapi_api_key: str | None,
    session_id: str,
    proxy: str | None = None,
) -> list[Tool]:
    key = (
        os.getenv("SERPAPI_API_KEY", "") if serpapi_api_key is None else serpapi_api_key
    )
    if provider == "auto":
        provider = "serpapi" if key else "parallel"
    if provider == "none":
        return []
    if provider == "serpapi":
        if not key:
            raise ValueError(
                "SerpAPI search requires serpapi_api_key or SERPAPI_API_KEY"
            )
        search = SerpAPIWrapper(serpapi_api_key=key)
        return [
            Tool(
                name="Search",
                func=search.run,
                coroutine=getattr(search, "arun", None),
                description="如果你不知道或不确定答案，可以使用这个搜索引擎检索答案",
            )
        ]
    if provider != "parallel":
        raise ValueError(f"Unknown LangChain search provider: {provider}")
    search = ParallelSearch(session_id, proxy)
    return [
        Tool(
            name="Search",
            func=None,
            coroutine=search.search,
            description="Search the web for current information. Input a search query.",
            handle_tool_error=True,
        ),
        Tool(
            name="Fetch",
            func=None,
            coroutine=search.fetch,
            description="Read a web page. Input one HTTP or HTTPS URL.",
            handle_tool_error=True,
        ),
    ]


async def agent_search(
    query: str,
    memory: BaseMemory,
    callback: BaseCallbackHandler | None = None,
    *,
    search_provider: str = "auto",
    serpapi_api_key: str | None = None,
    session_id: str | None = None,
    proxy: str | None = None,
) -> str:
    llm = ChatOpenAI(
        streaming=True,
        temperature=0,
        model="gpt-3.5-turbo-0613",
    )

    # Initialization: search chain, mathematical calculation chain
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

    # Tool list: search, mathematical calculations
    tools = search_tools(
        search_provider, serpapi_api_key, session_id or str(uuid4()), proxy
    ) + [
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="在需要回答数学问题时非常有用",
        ),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant."),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = AgentExecutor(
        agent=create_openai_functions_agent(llm, tools, prompt),
        tools=tools,
        verbose=False,
        memory=memory,
    )
    callbacks = [callback] if callback else None
    # query eg：'杭州亚运会中国队获得了多少枚金牌？' // '计算 3 的 2 次方'
    result = await agent.ainvoke({"input": query}, config={"callbacks": callbacks})
    return result["output"]
