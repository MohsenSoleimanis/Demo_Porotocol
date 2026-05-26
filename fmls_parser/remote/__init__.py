"""Remote heavy-model parsing.

Layout:
  - schemas.py : request/response models shared by client and server
  - client.py  : runs locally, calls a Lightning AI box over an SSH-forwarded port
  - server.py  : FastAPI app — copy to Lightning AI along with requirements-remote.txt

This split lets `client.py` import without requiring torch/docling locally.
"""
