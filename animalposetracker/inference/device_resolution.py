"""Turn "no choice" or the generic CPU alias into a device key the backend tables accept.

The device names the UI offers ("AMD CPU", "NVIDIA GPU", ...) are also the keys of
``ENGINEtoBackend`` and ``ENGINEtoDEVICE``. Callers outside the GUI were left to spell
them by hand: ``--device`` and ``build_inference_session`` both defaulted to a bare
``"CPU"``, which is not a key of the ONNX table, so the ONNX backend raised
``KeyError: 'CPU'`` before it ever loaded a model. This module resolves the two cases a
caller cannot resolve on its own -- nothing chosen, or the generic ``"CPU"`` -- and
leaves every other name untouched.

It is deliberately not a validator: engines differ in what they accept (the OpenCV
backend matches ``"CPU"`` by substring, so "AMD CPU" works there even though the name
is no key of its table). Verifying a name stays the backend's business.

Nothing here imports Qt: the CLI and the session builder are the main callers, and the
GUI imports it too so a platform is detected in one place.
"""

from typing import Optional

from animalposetracker.inference.constant import ENGINEtoBackend, ENGINEtoDEVICE


def detect_platform() -> str:
    """Return 'Intel' / 'AMD' / 'ARM' from the CPU vendor, or 'Unknown'.

    An unrecognised vendor is not an error: it only means the vendor-specific device
    lookups stay off, which is what an unknown vendor should look like. (Moved here
    from the GUI's inference page: the CLI and the session builder need it more than
    the page does.)
    """
    import cpuinfo

    info = cpuinfo.get_cpu_info()
    vendor = info.get("vendor_id_raw", "") or info.get("vendor_id", "") or ""
    if vendor == "GenuineIntel":
        return "Intel"
    if vendor == "AuthenticAMD":
        return "AMD"
    if "ARM" in info.get("arch", ""):
        return "ARM"
    return "Unknown"


def default_device_for(engine: str, platform: Optional[str] = None) -> Optional[str]:
    """First device ``engine`` offers on ``platform``, or None when it offers none."""
    if platform is None:
        platform = detect_platform()
    devices = ENGINEtoDEVICE.get(engine, {}).get(platform, [])
    return devices[0] if devices else None


def resolve_device(engine: str, device: Optional[str] = None) -> str:
    """Resolve a device choice to the name the backend tables use.

    ``None`` becomes the engine's first device on the detected platform (an engine with
    none there reports it, instead of failing later with a KeyError). The generic
    ``"CPU"`` becomes the platform-keyed spelling when the engine has no plain ``"CPU"``
    key. Anything else passes through unchanged.
    """
    if device is None:
        resolved = default_device_for(engine)
        if resolved is None:
            platform = detect_platform()
            raise ValueError(
                f"Engine {engine!r} offers no device on this {platform} platform; "
                "choose an engine that supports this machine"
            )
        return resolved
    table = ENGINEtoBackend.get(engine, {})
    if device == "CPU" and "CPU" not in table:
        platform_spelling = f"{detect_platform()} CPU"
        if platform_spelling in table:
            return platform_spelling
    return device


__all__ = ["detect_platform", "default_device_for", "resolve_device"]
