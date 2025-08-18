import enum
from livekit.agents.llm import function_tool
import logging
import os
import weaviate
import pandas as pd
import re
import pandas as pd
from livekit.agents import Agent, AgentSession, ModelSettings, stt, utils
import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, Coroutine
from livekit import rtc
from typing import Any
import duckdb
from unidecode import unidecode
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.chat_engine import SimpleChatEngine

from datetime import datetime
import time


def gemini_call(query: str, system_prompt: str = None):
    llm = GoogleGenAI(model="gemini-2.0-flash", system_prompt=system_prompt)
    ce = SimpleChatEngine(llm)
    return str(ce.chat(query))


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
car_parts = pd.read_csv(
    "/home/rejk/Documents/temppp/livekittest/backend/_test/lynkco_suzuki.csv"
)


import re


def extract_sql_code(text: str) -> str:
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else text


class CarPartAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
            instructions=(
                """\
You are Vietnamese auto part sale voicecall assistant with internal pricing information of Suzuki and Lynk&Co. 
To answer user question, you must search database to get relevant information.
You should use short and concise responses. DONOT produce unpronouncable punctuation.\
"""
            ),
        )

    @function_tool
    async def search_text(self, query: str):
        """
        Search for car part information in the db. 2 car provider including Lynk&co and Suzuki.
        Returns top 5 car parts related to queries.
        """
        print(query)
        client = weaviate.Client("http://localhost:9086")
        SEP = "\t"
        context = ""
        results = (
            client.query.get(class_name="CarPart", properties=["content"])
            .with_bm25(query="replacement part", properties=["content"])
            .with_limit(5)
            .do()
            .get("data", {})
            .get("Get", {})
            .get("CarPart", [])
        )
        if results:
            context = re.sub(r"\n+", r"\n", str(results)) + "\n\n"
        print(context)
        return "Một số bộ phận liên quan kèm ID : " + str(context)

    async def _query_sql(self, query: str):
        preset_allow_interruptions = self._allow_interruptions
        self._allow_interruptions = False
        print("run query" * 20, "\n", str(query))

        tries = 3
        current_query = query
        try:
            while True:
                try:
                    result = duckdb.query(query).to_df()
                    print("result" * 20, result)
                    self._allow_interruptions = preset_allow_interruptions
                    await self.session.generate_reply(
                        instructions=f"Answer user question based on {str(result.to_dict(orient='records'))}"
                    )
                    return str(result.to_dict(orient="records"))
                except Exception as e:
                    print(e)
                    current_query = gemini_call(
                        f"""
The sql query:
```sql
{current_query}
```
Caused error: {e}
Fix it. Code only. No comments.
You might need to consider tiếng việt không dấu.
""",
                        "You are a data architect engineer",
                    )
                    current_query = extract_sql_code(current_query)
                    if (tries := tries - 1) > 0:
                        continue
                    raise e
        except Exception as e:
            asyncio.get_running_loop().create_task(
                self.session.generate_reply(
                    instructions=f"Some error occured in db {str(e)}. Excuse yourself."
                )
            )
            return str(e)
        finally:
            self._allow_interruptions = preset_allow_interruptions

    @function_tool
    async def query_sql(self, query: str):
        """
                Run SQL queries on car part information in the database using DuckDB.
                Supports two car providers: Lynk&Co and Suzuki.
                Available columns:
        part_id: unique string id
        part_name: unnormalized,can be uppercase, tiếng Việt không dấu, viết liền,...
        car_name: brandname in lowercase with model name, some time it can be stacked like lynk&co 01&05&09
        quantity: quanity per car part
        VAT: tax
        distributor_price_vnd
        retail_price_vnd
                Example query:
                SELECT * FROM car_parts WHERE LOWER(car_part) LIKE LOWER('%bánh xe%') OR LOWER(car_part) LIKE LOWER('%banh xe%') OR LOWER(car_part) LIKE LOWER('%banhxe%');
        """
        asyncio.get_running_loop().create_task(self._query_sql(query))

        return "Tool response is pending. Ask user to wait for a moment as you check the db creatively."
