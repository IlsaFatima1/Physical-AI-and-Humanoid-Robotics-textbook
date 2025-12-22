from mcp.server.fastmcp import FastMCP

mcp = FastMCP("physical-ai-textbook-mcp")

@mcp.tool()
def health_check() -> str:
    return "MCP server is running and authorized"

if __name__ == "__main__":
    mcp.run()
