"""
Monkey-patch for discord-ext-voice-recv to harden the audio receive pipeline.

Wraps PacketDecoder._process_packet() to catch two classes of errors that
otherwise crash the PacketRouter thread and kill all audio reception:

1. DAVE decrypt failures (ValueError) — during DAVE handshake transitions,
   some packets arrive unencrypted or with stale epoch keys.
2. Opus decode failures (OpusError: corrupted stream) — when DAVE decryption
   produces garbage data, the Opus decoder raises on the corrupted payload.

Both are caught and the offending packet is dropped so the router continues.

See: https://github.com/imayhaveborkedit/discord-ext-voice-recv/pull/56
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Counters for throttled logging
_dave_fail_count: int = 0
_opus_fail_count: int = 0


def apply() -> None:
    """Apply safety patches to voice-recv's PacketDecoder."""
    global _dave_fail_count, _opus_fail_count

    try:
        from discord.ext.voice_recv.opus import PacketDecoder, VoiceData
    except ImportError:
        logger.warning("Could not import voice_recv.opus — patch not applied")
        return

    try:
        from davey import MediaType
        has_dave = True
    except ImportError:
        has_dave = False

    if not has_dave:
        logger.info("davey not installed — DAVE decrypt patch not needed")
        return

    log = logging.getLogger("discord.ext.voice_recv.opus")

    def _patched_process_packet(self, packet):
        """
        Patched _process_packet that catches DAVE decrypt failures
        and Opus decode errors so the PacketRouter stays alive.
        """
        global _dave_fail_count, _opus_fail_count

        pcm = None
        member = self._get_cached_member()

        if packet.payload != 120:
            return None

        if member is None:
            self._cached_id = self.sink.voice_client._get_id_from_ssrc(self.ssrc)  # type: ignore
            member = self._get_cached_member()

        # --- DAVE inner decryption ---
        if (
            has_dave
            and not packet.is_silence()
            and packet.decrypted_data is not None
            and self.vc._connection.dave_session is not None
            and self.vc._connection.dave_session.ready
        ):
            if member is None:
                return None

            try:
                packet.decrypted_data = self.vc._connection.dave_session.decrypt(
                    member.id,
                    MediaType.audio,
                    bytes(packet.decrypted_data),
                )
            except (ValueError, Exception) as e:
                _dave_fail_count += 1
                if _dave_fail_count in (1, 10, 50, 200) or _dave_fail_count % 500 == 0:
                    log.warning(
                        "DAVE decrypt failed (count=%d), dropping packet: %s",
                        _dave_fail_count,
                        e,
                    )
                return None

        # --- Opus decode ---
        if not self.sink.wants_opus():
            try:
                packet, pcm = self._decode_packet(packet)
            except Exception as e:
                _opus_fail_count += 1
                if _opus_fail_count in (1, 10, 50, 200) or _opus_fail_count % 500 == 0:
                    log.warning(
                        "Opus decode failed (count=%d), dropping packet: %s",
                        _opus_fail_count,
                        e,
                    )
                return None

        data = VoiceData(packet, member, pcm=pcm)
        self._last_seq = packet.sequence
        self._last_ts = packet.timestamp

        return data

    PacketDecoder._process_packet = _patched_process_packet
    logger.info("Applied DAVE + Opus safety patch to PacketDecoder._process_packet")
