import contextlib
import enum
from mmap import MADV_DONTDUMP
import sys
import os
from collections.abc import AsyncIterator
from token import OP
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from markitdown import MarkItDown
import uvicorn
from openai import OpenAI
import asyncio
from typing import Optional
from typing import Annotated
import camelot
from plugin import register_converters
import toml
from pathlib import Path
from urllib.parse import unquote, urlparse

# Initialize FastMCP server for MarkItDown (SSE)
mcp = FastMCP("file_reader_enhanced")

@mcp.tool()
async def read_file_enhanced(
    uri: str = Field(
        description="The URI of the file to convert to markdown. Supports http://, https://, file://, or data: schemes. Examples: 'file:///path/to/document.pdf', 'https://example.com/image.jpg'",
    ),
    llm_process_prompt: Optional[str] = Field(
        description="Specific requirements for processing images, videos and audios. Must be provided if you want to get information from images and videos",
        default=None
    ),
    process_type: str = Field(
        description="The type of the file to process. Can be 'audio', 'video', 'image_ocr', 'image_description', 'others'",
        default="others"
    ),
) -> str:
    """
A universal file processing tool that converts various formats to structured markdown.

## Supported Formats
- Office documents: Word, Excel, PowerPoint
- PDF: text, tables, images
- Images: OCR and AI description
- Media: audio transcription, video analysis
- Archives: ZIP, RAR extraction
- Web: Wikipedia, HTML

## Features
- Supports file:// and https:// sources
- AI-powered analysis for images, audio, and video
- Multiple processing prompts per file for flexible output

Easily convert and analyze any file type through a unified interface.
"""
    # 对 uri 做特殊处理
    if uri.startswith("file://"):
        # 去掉 file:// 前缀，并应用 root_path
        file_path = file_uri_to_path(uri)
        file_path = apply_root_path(file_path)
        uri = Path(file_path).as_uri()
    elif not (uri.startswith("http://") or uri.startswith("https://") or uri.startswith("data:")):
        # 如果不是标准协议，认为是本地路径
        file_path = apply_root_path(uri)
        uri = Path(file_path).as_uri()

    # Execute synchronous MarkItDown conversion in thread pool to avoid blocking
    if uri.startswith("file://") and Path(file_uri_to_path(uri)).suffix.lower() == ".pdf":
        markdown = mineru_pdf_to_markdown(file_uri_to_path(uri))
    else:
        md = MarkItDown(
                    enable_plugins=check_plugins_enabled(),
        )
        register_converters(md)
        llm_settings = llm_config.get(process_type, llm_config["others"])
        try:
            markdown = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: md.convert_uri(uri, process_type=process_type,                    
                        llm_prompt=llm_process_prompt,
                        llm_model=llm_settings["llm_model"],
                        llm_client=create_llm_client(llm_settings)).markdown,
                ),
                timeout=60
            )
        except asyncio.TimeoutError:
            # 这里写超时后的处理逻辑，比如返回错误信息、日志等
            markdown = "处理超时，请稍后重试。"
    return markdown


def apply_root_path(uri: str) -> str:
    root_path = Path(os.getenv("ROOT_PATH") or "/app/data").resolve()
    requested_path = Path(unquote(uri))
    candidate = requested_path.resolve() if requested_path.is_absolute() else (root_path / requested_path).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("File path is outside the configured root path") from exc
    return str(candidate)


def file_uri_to_path(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("Only local file URIs are supported")
    return unquote(parsed.path)


def create_llm_client(settings):
    if not settings.get("llm_api_key"):
        return None
    return OpenAI(
        base_url=settings["llm_base_url"],
        api_key=settings["llm_api_key"],
    )

def check_plugins_enabled() -> bool:
    return os.getenv("MARKITDOWN_ENABLE_PLUGINS", "false").strip().lower() in (
        "true",
        "1",
        "yes",
    )

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    sse = SseServerTransport("/messages/")
    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=True,
        stateless=True,
    )

    async def handle_sse(request: Request) -> None:
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

    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for session manager."""
        async with session_manager.run():
            print("Application started with StreamableHTTP session manager!")
            try:
                yield
            finally:
                print("Application shutting down...")

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/mcp", app=handle_streamable_http),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        lifespan=lifespan,
    )

# TODO: mineru pdf to markdown, use camelot to replace first
def mineru_pdf_to_markdown(file_path: str) -> str:

    print(f"Reading PDF file: {file_path}")
    # 读取 PDF 表格
    tables = camelot.read_pdf(file_path)
    # 取第一个表格，转为 pandas DataFrame
    df = tables[0].df
    # DataFrame 转为 Markdown 格式字符串
    md_str = df.to_markdown(index=False)

    return md_str

def load_llm_config(config_path):
    import toml
    config = toml.load(config_path)
    shared_api_key = os.getenv("READ_FILE_LLM_API_KEY") or config.get("llm_api_key")
    shared_base_url = os.getenv("READ_FILE_LLM_BASE_URL") or config.get("llm_base_url")
    llm_config = {
        "image_description":{
            "llm_model": config.get("image_llm_model") or "qwen-vl-plus-latest",
            "llm_base_url": os.getenv("READ_FILE_IMAGE_LLM_BASE_URL") or config.get("image_llm_base_url") or shared_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "llm_api_key": os.getenv("READ_FILE_IMAGE_LLM_API_KEY") or config.get("image_llm_api_key") or shared_api_key,
        },
        "image_ocr":{
            "llm_model": config.get("ocr_llm_model") or "qwen-vl-ocr-latest",
            "llm_base_url": os.getenv("READ_FILE_OCR_LLM_BASE_URL") or config.get("ocr_llm_base_url") or shared_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "llm_api_key": os.getenv("READ_FILE_OCR_LLM_API_KEY") or config.get("ocr_llm_api_key") or shared_api_key,
        },
        "audio":{
            "llm_model": config.get("audio_llm_model") or "qwen-omni-turbo-latest",
            "llm_base_url": os.getenv("READ_FILE_AUDIO_LLM_BASE_URL") or config.get("audio_llm_base_url") or shared_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "llm_api_key": os.getenv("READ_FILE_AUDIO_LLM_API_KEY") or config.get("audio_llm_api_key") or shared_api_key,
        },
        "video":{
            "llm_model": config.get("video_llm_model") or "qwen-omni-turbo-latest",
            "llm_base_url": os.getenv("READ_FILE_VIDEO_LLM_BASE_URL") or config.get("video_llm_base_url") or shared_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "llm_api_key": os.getenv("READ_FILE_VIDEO_LLM_API_KEY") or config.get("video_llm_api_key") or shared_api_key,
        },
        "others":{
            "llm_model": config.get("llm_model") or "qwen-omni-turbo-latest",
            "llm_base_url": shared_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "llm_api_key": shared_api_key,
        }
    }
    return llm_config

# Main entry point
def main():
    import argparse

    mcp_server = mcp._mcp_server

    parser = argparse.ArgumentParser(description="Run a MarkItDown MCP server")

    parser.add_argument(
        "--http",
        action="store_true",
        help="Run the server with Streamable HTTP and SSE transport rather than STDIO (default: False)",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="(Deprecated) An alias for --http (default: False)",
    )
    parser.add_argument(
        "--host", default=None, help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to listen on (default: 3001)"
    )
    parser.add_argument(
        "--config",
        default="/home/gongziqin/mcp_servers/feature-server/mcp_servers/read_file_enhanced/assets/config.toml",
        help="Path to config.toml file (default: assets/config.toml)",
    )
    parser.add_argument(
        "--root_path",
        help="Root path to apply to the URI (default: '')",
    )

    args = parser.parse_args()

    os.environ["ROOT_PATH"] = args.root_path or "/app/data"

    global llm_config
    llm_config = load_llm_config(args.config)

    use_http = args.http or args.sse

    if not use_http and (args.host or args.port):
        parser.error(
            "Host and port arguments are only valid when using streamable HTTP or SSE transport (see: --http)."
        )
        sys.exit(1)

    if use_http:
        starlette_app = create_starlette_app(mcp_server, debug=True)
        uvicorn.run(
            starlette_app,
            host=args.host if args.host else "127.0.0.1",
            port=args.port if args.port else 3001,
        )
    else:
        mcp.run()

if __name__ == "__main__":
    main()
