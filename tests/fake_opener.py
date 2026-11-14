"""Test double for a urllib opener: records the request and returns a fixed reply."""

import io


class FakeOpener:
  """Stands in for urllib.request.OpenerDirector so no socket is opened."""

  def __init__(self, obj_reply: bytes = b"", obj_error: Exception | None = None) -> None:
    """Store the reply body to return, or the error to raise, and start with no recorded calls."""
    self.obj_reply = obj_reply
    self.obj_error = obj_error
    self.l_requests: list = []
    self.l_timeouts: list = []

  def open(self, obj_request: object, timeout: float | None = None) -> io.BytesIO:
    """Record the request and the timeout, then return the reply as a readable stream or raise."""
    self.l_requests.append(obj_request)
    self.l_timeouts.append(timeout)
    if self.obj_error is not None:
      raise self.obj_error
    return io.BytesIO(self.obj_reply)
