"""Hub-side WebRTC relay: latest JPEG from agent → browser (Orin uploads via HTTP)."""
from __future__ import annotations

import asyncio
import io
import time
from fractions import Fraction
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import av
    import numpy as np
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaRelay

    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False
    VideoStreamTrack = object  # type: ignore

FrameGetter = Callable[[str], Tuple[Optional[bytes], float]]

_pcs: Dict[str, RTCPeerConnection] = {}
_relay = MediaRelay() if WEBRTC_OK else None


class _JpegRelayTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, rover_id: str, get_frame: FrameGetter, fps: float = 15.0) -> None:
        super().__init__()
        self._rover_id = rover_id
        self._get_frame = get_frame
        self._fps = max(5.0, fps)
        self._idx = 0

    async def recv(self) -> Any:
        pts = self._idx
        self._idx += 1
        await asyncio.sleep(1.0 / self._fps)
        jpeg, _stamp = self._get_frame(self._rover_id)
        if not jpeg:
            arr = np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            try:
                from PIL import Image

                img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                arr = np.array(img)
                arr = arr[:, :, ::-1].copy()  # RGB → BGR for VideoFrame
            except Exception:
                arr = np.zeros((360, 640, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts = pts
        frame.time_base = Fraction(1, int(self._fps))
        return frame


async def create_relay_answer(
    rover_id: str,
    offer_sdp: str,
    get_frame: FrameGetter,
    ice_servers: list,
    *,
    fps: float = 15.0,
) -> str:
    if not WEBRTC_OK:
        raise RuntimeError("aiortc_not_installed")

    old = _pcs.pop(rover_id, None)
    if old is not None:
        await old.close()

    pc = RTCPeerConnection(configuration={"iceServers": ice_servers})
    _pcs[rover_id] = pc
    track = _JpegRelayTrack(rover_id, get_frame, fps=fps)
    if _relay is not None:
        pc.addTrack(_relay.subscribe(track))
    else:
        pc.addTrack(track)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc.localDescription.sdp


async def close_relay(rover_id: str) -> None:
    pc = _pcs.pop(rover_id, None)
    if pc is not None:
        await pc.close()
