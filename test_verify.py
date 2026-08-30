import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from mcp_servers.verifier_server import verify_answer
result = verify_answer("What is the speed of light?", "The speed of light is 100 miles per hour.")
print(result)
