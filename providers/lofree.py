"""Lofree Hyzen keyboards, and AULA V9 mice, on the shared 0xFF1C/0x92 exchange.

Two makers sit on this one collection and this one transaction. Lofree's web driver is
the source for the exchange itself; the AULA mouse was measured against its own software
(AULA HUB) and on the hardware, and it answers the same frames, with the level at the
same offset. What AULA adds is a charge state one byte past the level and a link-state
frame the receiver pushes on its own (see AULA NOTE below).

Source: Lofree's own web driver (hyzen.lofree.tech). Its code is obfuscated; the
exchange below comes from its device list and from its battery routine, with the
strings decoded by the page's own decoder functions.

Device list: vendor 0x388D, collection usage page 0xFF1C, usage 0x92:
    0x0024  Hyzen on the USB cable     (connection type 1, device id 101)
    0x0025  Hyzen on the 2.4 GHz dongle (connection type 2, device id 101)
    0x0029  a second model on the cable (device id 102)

Every command is a small transaction on HID report 0x04. After the report id, a
frame is 31 bytes on the dongle (63 on the cable):

    host   00 00 01                       start
    kbd    .. .. 01 ...                   byte 2 = 01: start acknowledged
    host   00 00 <cmd> 00 00 00 00        the command, no data, offset 0
    kbd    .. .. <cmd> .. 00 00 .. <d0>   byte 2 = cmd (or 00), bytes 4-5 = offset 0,
                                          the answer starts at byte 7
    host   00 00 02                       end
    kbd    .. .. 02 ...                   byte 2 = 02: end acknowledged

The web driver reads the battery only over the dongle, and not when the same
keyboard is also on its cable:
    command 0xAA  "is the keyboard online": d0 = 0 means offline
    command 0x1A  battery: d0 = the level in %
The web driver shows no charging state, so this provider does not either.

AULA V9 mice - the same exchange, measured on a V9 Max+ (dongle 320F:22A0, cable
320F:22B7) against the maker's own app (AULA HUB, an Electron app whose HID calls were
hooked over CDP) and then on the hardware:

  * the mouse acknowledges START and answers command 0x1A with the level at the same
    DATA_AT offset 7 - measured `00 00 1a 00 00 00 00 59 00` = 89 % - so the Lofree
    transaction works on it unchanged. It also answers the same command without the
    START/END handshake, but the documented way is used here.
  * one byte past the level is the charge state, which Lofree's keyboards do not send:
    0 = not charging, 1 = charging, 2 = charged. Measured: `.. 64 00`, `.. 63 01` and
    `.. 64 02` while the same mouse sat idle, on the cable and fully charged. The maker's
    own app treats 1 and 2 as "charging", and so does this provider.
  * a level of 0 is not a level: a receiver that has just been re-plugged answers zeros
    until the mouse reports, and 0 would also fire a false low-battery alert.
  * the receiver answers from its cache whether or not the mouse is switched on, so the
    level alone never proves it is there. The link state comes from a frame the receiver
    pushes every ~0.5 s while the link is up: `04 AA 00 AA 00 00 00 00 <state>` with
    0xFF while the mouse is on and 0x00 while it is off (measured five seconds apart, same
    session). The frame is not used to *find* the mouse - it is absent when the mouse is
    idle - only to correct a reading that arrives while the link is known to be down.
  * the mouse sleeps when idle and the next poll wakes it, so one 0x00 is not conclusive.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

try:
    import hid
except ImportError:              # pragma: no cover
    hid = None

from . import hidlist
from .base import DeviceStatus, Provider, hexdump, log

LOFREE_VID = 0x388D
USAGE_PAGE, USAGE = 0xFF1C, 0x0092

DONGLES = {0x0025: "Lofree Hyzen"}           # the pids that are read (2.4 GHz)
WIRED_OF = {0x0025: 0x0024}                  # the same keyboard on its cable

# AULA V9 mice on the same collection and the same command; the cable and the dongle of
# one mouse are two ids of one device and share one icon.
AULA_VID = 0x320F
AULA_PIDS = {0x22A0: "AULA V9 Max+ (2.4 GHz receiver)",
             0x22B7: "AULA V9 Max+ (USB cable)"}

REPORT_ID = 0x04
FRAME = 31                                   # bytes after the report id, on the dongle
START, END = 0x01, 0x02
CMD_ONLINE, CMD_BATTERY = 0xAA, 0x1A
DATA_AT = 7                                  # the answer starts here (after the report id)
TIMEOUT = 3.0                                # s, the web driver's minimum timeout
READ_MS = 100

# AULA only: the charge state one byte past the level, and the pushed link-state frame.
CHARGE_AT = DATA_AT + 1
HEARTBEAT_AT = DATA_AT                    # byte 7 of the payload after `04 AA 00 AA`
LINK_ON = 0xFF


def frame(b2: int) -> List[int]:
    """Report 0x04 with byte 2 set: 00 00 <b2> 00 ..., 31 bytes after the id."""
    return [REPORT_ID, 0x00, 0x00, b2] + [0x00] * (FRAME - 3)


def payload(r) -> Optional[List[int]]:
    """The bytes after the report id of an input report 0x04, or None."""
    if not r or len(r) < 2 or r[0] != REPORT_ID:
        return None
    return list(r[1:])


class LofreeProvider(Provider):
    name = "lofree"

    def __init__(self):
        self._diag: List[str] = []
        self._link_down = False

    def _wait(self, dev, want, deadline: float, watch_link: bool = False) -> Optional[List[int]]:
        """Read reports 0x04 until `want(payload)` is true, or until the deadline.

        With `watch_link`, AULA's pushed link-state frame (`04 AA 00 AA ... <state>`) is
        picked up on the way: 0x00 means the mouse is switched off, and a reading that
        arrives then is the receiver's cache, not the mouse.
        """
        while time.time() < deadline:
            try:
                r = dev.read(64, READ_MS)
            except (OSError, ValueError) as e:
                self._diag.append(f"    read: {e}")
                return None
            p = payload(r)
            if p is None:
                continue
            if watch_link and len(p) > HEARTBEAT_AT and p[0] == 0xAA and p[2] == 0xAA:
                if p[HEARTBEAT_AT] != LINK_ON:
                    self._link_down = True
                continue
            if want(p):
                return p
        return None

    def _command(self, dev, cmd: int, watch_link: bool = False) -> Optional[List[int]]:
        """One transaction; the payload of the answer, or None."""
        deadline = time.time() + TIMEOUT
        dev.write(frame(START))
        if self._wait(dev, lambda p: len(p) >= 3 and p[2] == START, deadline,
                      watch_link) is None:
            self._diag.append(f"    cmd {cmd:02x}: no start acknowledgement")
            return None
        dev.write([REPORT_ID, 0x00, 0x00, cmd, 0x00, 0x00, 0x00, 0x00] + [0x00] * (FRAME - 7))
        p = self._wait(dev, lambda p: (len(p) > DATA_AT and p[2] in (cmd, 0x00)
                                       and p[4] == 0x00 and p[5] == 0x00), deadline,
                       watch_link)
        # close the transaction in any case, as the web driver does
        dev.write(frame(END))
        self._wait(dev, lambda p: len(p) >= 3 and p[2] == END, min(deadline, time.time() + 0.5))
        if p is None:
            self._diag.append(f"    cmd {cmd:02x}: no answer")
            return None
        self._diag.append(f"    cmd {cmd:02x}: {hexdump(p, 10)}")
        return p

    def _open(self, path):
        dev = hid.device()
        try:
            dev.open_path(path)
        except (OSError, IOError) as e:
            self._diag.append(f"    open: {e}")
            return None
        return dev

    def _read(self, path) -> Optional[int]:
        """A Lofree keyboard's level: the online command first, as the web driver does."""
        dev = self._open(path)
        if dev is None:
            return None
        try:
            try:
                online = self._command(dev, CMD_ONLINE)
                if online is None:
                    return None
                if online[DATA_AT] == 0:
                    self._diag.append("    keyboard offline (off or asleep)")
                    return None
                answer = self._command(dev, CMD_BATTERY)
            except (OSError, IOError, ValueError) as e:
                self._diag.append(f"    write: {e}")
                return None
            if answer is None:
                return None
            level = answer[DATA_AT]
            if not 0 <= level <= 100:
                self._diag.append(f"    level {level} out of range, not shown")
                return None
            return level
        finally:
            try:
                dev.close()
            except Exception:
                pass

    def _read_mouse(self, path) -> Optional[Tuple[int, bool]]:
        """An AULA mouse's (level, charging); the battery command alone is enough."""
        self._link_down = False
        dev = self._open(path)
        if dev is None:
            return None
        try:
            try:
                answer = self._command(dev, CMD_BATTERY, watch_link=True)
            except (OSError, IOError, ValueError) as e:
                self._diag.append(f"    write: {e}")
                return None
            if answer is None:
                return None
            level = answer[DATA_AT]
            # 0 is not a level: a receiver that was just re-plugged answers zeros until the
            # mouse reports, and 0 % would also fire a false low-battery alert
            if not 1 <= level <= 100:
                self._diag.append(f"    level {level} out of range, not shown")
                return None
            charge = answer[CHARGE_AT] if len(answer) > CHARGE_AT else 0
            if charge not in (0, 1, 2):
                charge = 0
            return level, charge in (1, 2)
        finally:
            try:
                dev.close()
            except Exception:
                pass

    def _lofree_keyboards(self, out: List[DeviceStatus]) -> None:
        try:
            infos = hidlist.enumerate(LOFREE_VID)
        except Exception as e:  # pragma: no cover
            log.warning("hid.enumerate(lofree): %s", e)
            return
        ids = {d["product_id"] for d in infos if d.get("vendor_id", LOFREE_VID) == LOFREE_VID}
        for d in infos:
            pid = d["product_id"]
            if pid not in DONGLES or (d.get("usage_page"), d.get("usage")) != (USAGE_PAGE, USAGE):
                continue
            product = (d.get("product_string") or "").split("@")[0].strip()
            name = f"Lofree {product}" if product else DONGLES[pid]
            self._diag.append(f"[Lofree] {name} pid={pid:04x}")
            if WIRED_OF.get(pid) in ids:
                self._diag.append("  the keyboard is on its cable: the dongle is not read")
                continue
            level = self._read(d["path"])
            if level is not None:
                self._diag.append(f"  -> {level}%")
                out.append(DeviceStatus(f"lofree:{pid:04x}", name, level, False, True, "lofree",
                                        kind="keyboard"))

    def _aula_mice(self, out: List[DeviceStatus]) -> None:
        try:
            infos = hidlist.enumerate(AULA_VID)
        except Exception as e:  # pragma: no cover
            log.warning("hid.enumerate(aula): %s", e)
            return
        cols = [d for d in infos
                if d["product_id"] in AULA_PIDS
                and (d.get("usage_page"), d.get("usage")) == (USAGE_PAGE, USAGE)]
        if not cols:
            return
        # the cable first: it is the live one when the mouse is plugged in
        cols.sort(key=lambda d: (d["product_id"] != 0x22B7,))
        name = ""
        got = None
        for d in cols:
            product = (d.get("product_string") or "").strip()
            if product:
                name = product
            pid = d["product_id"]
            self._diag.append(f"[AULA] {name or AULA_PIDS[pid]} pid={pid:04x}")
            got = self._read_mouse(d["path"])
            if got is not None:
                break
        if got is None:
            return
        level, charging = got
        online = not self._link_down
        self._diag.append(f"  -> {level}%{', charging' if charging else ''}"
                          + ("" if online else ", link reported down"))
        out.append(DeviceStatus("aula:mouse", name or AULA_PIDS[0x22A0], level, charging,
                                online, "lofree", kind="mouse"))

    def poll(self) -> List[DeviceStatus]:
        self._diag = []
        if hid is None:
            return []
        out: List[DeviceStatus] = []
        self._lofree_keyboards(out)
        self._aula_mice(out)
        return out

    def diagnostics(self) -> List[str]:
        return list(self._diag)
