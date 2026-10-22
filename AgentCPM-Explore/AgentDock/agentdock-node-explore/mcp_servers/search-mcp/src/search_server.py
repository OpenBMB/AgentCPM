import os
import json
import http.client
import time
from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import tiktoken
import asyncio
import httpx
import sys

# Initialize FastMCP server
mcp = FastMCP("research_tools")

# Environment variables
# Serper API配置 (作为降级备选)
SERPER_KEY = os.getenv("SERPER_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
# 新的Google Serp API配置
GOOGLE_SERP_API_KEY = os.getenv("GOOGLE_SERP_API_KEY", "")
GOOGLE_SERP_API_URL_CN = "https://api.serp.hk/serp/google/search/advanced"  # 国内入口
GOOGLE_SERP_API_URL_GLOBAL = "https://api.serp.global/serp/google/search/advanced"  # 国外入口

# Helper function for token truncation
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

# Helper function for Chinese detection
def contains_chinese_basic(text: str) -> bool:
    return any('\u4E00' <= char <= '\u9FFF' for char in text)

# Jina服务读取网页
async def jina_readpage(url: str) -> str:
    """Read webpage content using Jina service."""
    max_retries = 3
    timeout = 120
    
    if not url.startswith("http"):
        url = "http://" + url
    
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    content = response.text
                    return content
                else:
                    print(f"Jina error: {response.text}", file=sys.stderr)
                    raise ValueError("jina readpage error")
        except Exception as e:
            print(f"Jina attempt {attempt + 1} failed for {url}: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                # 指数退避重试
                await asyncio.sleep(2 ** attempt)
    
    return "[visit] Failed to read page."

# 使用Jina服务读取网页（带缓存）
async def html_readpage_with_jina(url: str) -> str:
    """Read webpage content using Jina service with caching."""
    # 首先尝试从缓存获取
    try:
        async with httpx.AsyncClient(timeout=50.0) as client:
            from urllib.parse import quote
            encoded_url = quote(url, safe='')
            cache_response = await client.get(f"http://host.docker.internal:27051/content/{encoded_url}")
            if cache_response.status_code == 200:
                data = cache_response.json()
                print(f"Cache hit for {url}", file=sys.stderr)
                return data.get("content", "")
    except Exception as e:
        print(f"Cache miss for {url}: {e}", file=sys.stderr)
    
    # 缓存未命中，使用Jina服务
    print(f"Fetching {url} using Jina service", file=sys.stderr)
    content = await jina_readpage(url)
    
    if content and not content.startswith("[visit] Failed to read page."):
        print(f"Successfully fetched {url} using Jina service", file=sys.stderr)
        # 保存到缓存
        try:
            async with httpx.AsyncClient(timeout=10.0) as cache_client:
                await cache_client.post(
                    "http://host.docker.internal:27051/content",
                    json={"url": url, "content": content}
                )
                print(f"Cached content for {url}", file=sys.stderr)
        except Exception as cache_error:
            print(f"Failed to cache {url}: {cache_error}", file=sys.stderr)
        return content
    
    return "[visit] Failed to read page."


# Serper API 作为降级备选（与主API相同的重试策略）
async def google_search_with_serper_fallback(query: str) -> str:
    """Perform Google search using Serper API as fallback (with same retry strategy)."""
    print(f"Using Serper API as fallback for query: {query}", file=sys.stderr)
    
    async def _search():
        payload = {"q": query}
        headers = {
            'X-API-KEY': SERPER_KEY,
            'Content-Type': 'application/json'
        }
        
        # 内层5次重试
        for i in range(5):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://google.serper.dev/search",
                        json=payload,
                        headers=headers
                    )
                    if response.status_code == 200:
                        return response.json()
                    else:
                        print(f"Serper API error: {response.status_code} - {response.text}", file=sys.stderr)
                        return {"error": f"API error: {response.status_code}"}
            except Exception as e:
                print(f"Serper search attempt {i + 1} failed: {e}", file=sys.stderr)
                if i == 4:
                    return {"error": "Timeout"}
                await asyncio.sleep(0.5)
        
        return {"error": "Timeout"}
    
    # 外层重试循环 - 当没有organic结果时重试最多3次
    for retry_count in range(3):
        results = await _search()
        
        if "error" in results:
            if retry_count < 2:
                print(f"Serper API error, retrying... (attempt {retry_count + 1}/3)", file=sys.stderr)
                await asyncio.sleep(0.5)
                continue
            return f"Serper search failed: {results['error']}. Please try again later."
        
        if "organic" in results:
            # 有结果,跳出重试循环
            break
        else:
            # 没有结果,重试
            if retry_count < 2:
                print(f"No organic results found in Serper, retrying... (attempt {retry_count + 1}/3)", file=sys.stderr)
                await asyncio.sleep(0.5)
            else:
                # 最后一次尝试也没有结果
                return f"No results found for '{query}'. Try with a more general query."
    
    try:
        organic_results = results.get("organic", [])
        if not organic_results:
            return f"No results found for '{query}'. Try with a more general query."
        
        web_snippets = []
        for idx, page in enumerate(organic_results, 1):
            # Serper API 的字段结构
            title = page.get('title', 'No title')
            link = page.get('link', '')
            snippet = page.get('snippet', '')
            date = page.get('date', '')
            
            # 组装结果
            result_text = f"{idx}. [{title}]({link})"
            if date:
                result_text += f"\nDate: {date}"
            if snippet:
                result_text += f"\n{snippet}"
            
            web_snippets.append(result_text)
        
        content = f"A Google search for '{query}' found top {len(web_snippets)} results (via Serper fallback):\n\n## Web Results\n" + "\n\n".join(web_snippets)
        
        return content
    except Exception as e:
        print(f"Error processing Serper results: {e}", file=sys.stderr)
        return f"Error processing results for '{query}'. Please try again."


# Async wrapper for Google Search
async def google_search_with_serp(query: str) -> str:
    """Perform Google search using new Google Serp API with caching, with Serper fallback."""
    # First try to get from cache
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            from urllib.parse import quote
            encoded_query = quote(query, safe='')
            cache_response = await client.get(f"http://host.docker.internal:27051/search/{encoded_query}")
            if cache_response.status_code == 200:
                data = cache_response.json()
                print(f"Cache hit for search query: {query}", file=sys.stderr)
                return data.get("content", "")
    except Exception as e:
        print(f"Cache miss for search query: {query}: {e}", file=sys.stderr)
    
    # Cache miss, perform actual search using new Google Serp API
    async def _search():
        # 根据查询语言选择gl参数
        if contains_chinese_basic(query):
            payload = {"q": query, "gl": "CN"}
            api_url = GOOGLE_SERP_API_URL_CN  # 中文查询使用国内入口
        else:
            payload = {"q": query, "gl": "US"}
            api_url = GOOGLE_SERP_API_URL_GLOBAL  # 英文查询使用国外入口
        
        headers = {
            'Authorization': f'Bearer {GOOGLE_SERP_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        for i in range(5):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        api_url,
                        json=payload,
                        headers=headers
                    )
                    if response.status_code == 200:
                        return response.json()
                    else:
                        print(f"Google Serp API error: {response.status_code} - {response.text}", file=sys.stderr)
                        return {"error": f"API error: {response.status_code}"}
            except Exception as e:
                print(f"Search attempt {i + 1} failed: {e}", file=sys.stderr)
                if i == 4:
                    return {"error": "Timeout"}
                await asyncio.sleep(0.5)
        
        return {"error": "Timeout"}
    
    # 外层重试循环 - 当没有organic结果时重试最多3次
    primary_api_failed = False
    for retry_count in range(3):
        results = await _search()
        
        if "error" in results:
            primary_api_failed = True
            break  # 主API错误，跳出重试循环，使用fallback
        
        # 检查是否有organic结果
        if "result" in results:
            actual_results = results["result"]
        else:
            actual_results = results
        
        if "organic" in actual_results:
            # 有结果,跳出重试循环
            break
        else:
            # 没有结果,重试
            if retry_count < 2:
                print(f"No organic results found, retrying... (attempt {retry_count + 1}/3)", file=sys.stderr)
                await asyncio.sleep(0.5)
            else:
                # 最后一次尝试也没有结果，尝试使用Serper fallback
                primary_api_failed = True
    
    # 如果主API失败，使用Serper作为降级备选
    if primary_api_failed:
        print(f"Primary Google Serp API failed, switching to Serper fallback for: {query}", file=sys.stderr)
        return await google_search_with_serper_fallback(query)
    
    try:
        # actual_results已在重试循环中设置,这里直接使用
        # 检查是否有organic结果
        organic_results = actual_results.get("organic", [])
        if not organic_results:
            # 没有结果，尝试Serper fallback
            print(f"No organic results from primary API, trying Serper fallback for: {query}", file=sys.stderr)
            return await google_search_with_serper_fallback(query)
        
        web_snippets = []
        for idx, page in enumerate(organic_results, 1):
            # 提取字段,新API的字段结构
            title = page.get('title', 'No title')
            link = page.get('link', '')
            snippet = page.get('snippet', '')
            source = page.get('source', '')
            
            # 组装结果
            result_text = f"{idx}. [{title}]({link})"
            if source:
                result_text += f"\nSource: {source}"
            if snippet:
                result_text += f"\n{snippet}"
            
            web_snippets.append(result_text)
        
        # 添加总数信息(从general字段获取)
        total_count = actual_results.get("general", {}).get("count", len(web_snippets))
        content = f"A Google search for '{query}' found {total_count:,} total results (showing top {len(web_snippets)}):\n\n## Web Results\n" + "\n\n".join(web_snippets)
        
        # Save to cache
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    "http://host.docker.internal:27051/search",
                    json={"query": query, "content": content}
                )
                print(f"Cached search results for: {query}", file=sys.stderr)
        except Exception as cache_error:
            print(f"Failed to cache search for {query}: {cache_error}", file=sys.stderr)
        
        return content
    except Exception as e:
        print(f"Error processing search results: {e}, trying Serper fallback", file=sys.stderr)
        return await google_search_with_serper_fallback(query)

# # Async wrapper for Google Scholar - 已注释,新API不支持Scholar搜索
# async def google_scholar_with_serp(query: str) -> str:
#     """Perform Google Scholar search using Serper API with caching."""
#     # First try to get from cache using dedicated scholar endpoint
#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             from urllib.parse import quote
#             encoded_query = quote(query, safe='')
#             cache_response = await client.get(f"http://host.docker.internal:27051/scholar/{encoded_query}")
#             if cache_response.status_code == 200:
#                 data = cache_response.json()
#                 print(f"Cache hit for scholar query: {query}", file=sys.stderr)
#                 return data.get("content", "")
#     except Exception as e:
#         print(f"Cache miss for scholar query: {query}: {e}", file=sys.stderr)
#     
#     # Cache miss, perform actual search - 改为完全异步
#     async def _search():
#         payload = {"q": query}
#         headers = {
#             'X-API-KEY': SERPER_KEY,
#             'Content-Type': 'application/json'
#         }
#         
#         for i in range(5):
#             try:
#                 async with httpx.AsyncClient(timeout=30.0) as client:
#                     response = await client.post(
#                         "https://google.serper.dev/scholar",
#                         json=payload,
#                         headers=headers
#                     )
#                     return response.json()
#             except Exception as e:
#                 print(f"Scholar search attempt {i + 1} failed: {e}", file=sys.stderr)
#                 if i == 4:
#                     return {"error": "Timeout"}
#                 await asyncio.sleep(0.5)
#         
#         return {"error": "Timeout"}
#     
#     results = await _search()
#     
#     if "error" in results:
#         return f"Google Scholar Timeout, return None, Please try again later."
#     
#     try:
#         if "organic" not in results:
#             raise Exception(f"No results found for query: '{query}'.")
#         
#         web_snippets = []
#         idx = 0
#         if "organic" in results:
#             for page in results["organic"]:
#                 idx += 1
#                 date_published = f"\nDate published: {page['year']}" if "year" in page else ""
#                 publicationInfo = f"\npublicationInfo: {page['publicationInfo']}" if "publicationInfo" in page else ""
#                 snippet = f"\n{page['snippet']}" if "snippet" in page else ""
#                 link_info = f"pdfUrl: {page['pdfUrl']}" if "pdfUrl" in page else "no available link"
#                 citedBy = f"\ncitedBy: {page['citedBy']}" if "citedBy" in page else ""
#                 
#                 redacted_version = f"{idx}. [{page['title']}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"
#                 redacted_version = redacted_version.replace("Your browser can't play this video.", "")
#                 web_snippets.append(redacted_version)
#         
#         content = f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
#         
#         # Save to cache using dedicated scholar endpoint
#         try:
#             async with httpx.AsyncClient(timeout=10.0) as client:
#                 await client.post(
#                     "http://host.docker.internal:27051/scholar",
#                     json={"query": query, "content": content}
#                 )
#                 print(f"Cached scholar results for: {query}", file=sys.stderr)
#         except Exception as cache_error:
#             print(f"Failed to cache scholar search for {query}: {cache_error}", file=sys.stderr)
#         
#         return content
#     except:
#         return f"No results found for '{query}'. Try with a more general query."

@mcp.tool()
async def fetch_url(
    url: List[str] = Field(
        description="The URL(s) of the webpage(s) and online pdf(s) to visit. Can be a single URL or a list of URLs."
    ),
    purpose: str = Field(
        description="The purpose of the visit, what information you want to get from the webpage",
    ),
) -> str:
    """
    Fetch webpage(s) and online pdf(s) and return the content with AI summary.
    Supports parallel processing of multiple (at most 3) URLs.
    Uses Jina service for fetching content.
    """
    urls = [url] if isinstance(url, str) else url
    
    async def process_url(u: str) -> str:
        try:
            # 使用Jina服务获取内容
            content = await html_readpage_with_jina(u)
            
            if content and not content.startswith("[visit] Failed to read page."):
                content = truncate_to_tokens(content, max_tokens=95000)
                useful_information = f"The content from {u}:\n\n{content}\n\n"
                return useful_information
            else:
                return f"Failed to read content from {u}"
        except Exception as e:
            return f"Error fetching {u}: {str(e)}"
    
    # Process all URLs in parallel
    tasks = [process_url(u) for u in urls]
    results = await asyncio.gather(*tasks)
    
    return "\n=======\n".join(results)

@mcp.tool()
async def search(
    query: List[str] = Field(
        description="Array of query strings. Include multiple complementary search queries in a single call."
    )
) -> str:
    """
    Google search supports parallel processing of multiple (at most 3) queries. 
    The tool retrieves the top 10 results for each query in parallel.
    """
    queries = [query] if isinstance(query, str) else query
    
    # Process all queries in parallel
    tasks = [google_search_with_serp(q) for q in queries]
    results = await asyncio.gather(*tasks)
    
    return "\n=======\n".join(results)

# # Scholar工具已注释 - 新API不支持Google Scholar搜索
# @mcp.tool()
# async def scholar(
#     query: List[str] = Field(
#         description="The list of search queries for Google Scholar."
#     )
# ) -> str:
#     """
#     Leverage Google Scholar to retrieve relevant information from academic publications.
#     Accepts multiple queries and processes them in parallel.
#     """
#     queries = [query] if isinstance(query, str) else query
#     
#     # Process all queries in parallel (max 3 concurrent)
#     results = []
#     for i in range(0, len(queries), 3):
#         batch = queries[i:i+3]
#         tasks = [google_scholar_with_serp(q) for q in batch]
#         batch_results = await asyncio.gather(*tasks)
#         results.extend(batch_results)
#     
#     return "\n=======\n".join(results)

# @mcp.tool()
# async def wikipedia(
#     query: List[str] = Field(
#         description="The entity or topic to search on Wikipedia. Can be a single query or a list of queries."
#     )
# ) -> str:
#     """
#     Search simple Wikipedia summary for instant answers about entities and topics.
#     Processes multiple queries in parallel.
#     You are recommended to use visit tool to visit the wikipedia page for more detailed information.
#     """
#     queries = [query] if isinstance(query, str) else query
    
#     async def _duckduckgo_instant_search(entity: str) -> str:
#         """Handle DuckDuckGo Wikipedia search requests"""
#         try:
#             url = "http://host.docker.internal:5005/duckduckgo_search"
#             params = {"q": entity}
#             async with httpx.AsyncClient() as client:
#                 resp = await client.get(url, params=params)
#                 data = resp.json()
#                 return data
#         except Exception as e:
#             return f"DuckDuckGo Wikipedia search failed: {str(e)}"
    
#     # Process all queries in parallel
#     tasks = [_duckduckgo_instant_search(q) for q in queries]
#     results = await asyncio.gather(*tasks)
    
#     return "\n=======\n".join(results)

# Main entry point
def main():
    import argparse
    from starlette.applications import Starlette
    from mcp.server.sse import SseServerTransport
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    import uvicorn
    import contextlib
    from collections.abc import AsyncIterator
    
    mcp_server = mcp._mcp_server
    
    parser = argparse.ArgumentParser(description="Run Research Tools MCP Server")
    parser.add_argument("--http", action="store_true", help="Run with HTTP transport")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=3002, help="Port to listen on")
    
    args = parser.parse_args()
    
    if args.http:
        def create_starlette_app(mcp_server, *, debug: bool = False):
            sse = SseServerTransport("/messages/")
            session_manager = StreamableHTTPSessionManager(
                app=mcp_server,
                event_store=None,
                json_response=True,
                stateless=True,
            )
            
            async def handle_sse(request):
                async with sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send,
                ) as (read_stream, write_stream):
                    await mcp_server.run(
                        read_stream,
                        write_stream,
                        mcp_server.create_initialization_options(),
                    )
            
            async def handle_streamable_http(scope, receive, send):
                await session_manager.handle_request(scope, receive, send)
            
            @contextlib.asynccontextmanager
            async def lifespan(app) -> AsyncIterator[None]:
                async with session_manager.run():
                    print("Research Tools MCP Server started!", file=sys.stderr)
                    try:
                        yield
                    finally:
                        print("Server shutting down...", file=sys.stderr)
            
            from starlette.routing import Mount, Route
            return Starlette(
                debug=debug,
                routes=[
                    Route("/sse", endpoint=handle_sse),
                    Mount("/mcp", app=handle_streamable_http),
                    Mount("/messages/", app=sse.handle_post_message),
                ],
                lifespan=lifespan,
            )
        
        starlette_app = create_starlette_app(mcp_server, debug=True)
        uvicorn.run(starlette_app, host=args.host, port=args.port)
    else:
        mcp.run()

if __name__ == "__main__":
    main()

