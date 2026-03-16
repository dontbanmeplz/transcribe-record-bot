"""
Monkey-patch for discord-ext-voice-recv to handle DAVE decryption failures.

The rdphillips7 fork's PacketDecoder._process_packet() crashes the entire
audio receive pipeline when a DAVE decrypt fails (e.g., during the DAVE
handshake transition period where some packets arrive unencrypted).

This patch wraps the decrypt call in a try/except so failed packets are
dropped instead of crashing the PacketRouter thread.

See: https://github.com/imayhaveborkedit/discord-ext-voice-recv/pull/56
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply() -> None:
    """Apply the DAVE decrypt safety patch to voice-recv's PacketDecoder."""
    try:
        from discord.ext.voice_recv.opus import PacketDecoder, VoiceData
    except ImportError:
        logger.warning("Could not import voice_recv.opus — DAVE patch not applied")
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

    # Save original method
    _original_process_packet = PacketDecoder._process_packet

    def _patched_process_packet(self, packet):
        """
        Patched _process_packet that catches DAVE decrypt failures.

        When decrypt fails (e.g., UnencryptedWhenPassthroughDisabled),
        the packet is dropped instead of crashing the PacketRouter.
        """
        pcm = None
        member = self._get_cached_member()

        if packet.payload != 120:
            return None

        if member is None:
            self._cached_id = self.sink.voice_client._get_id_from_ssrc(self.ssrc)  # type: ignore
            member = self._get_cached_member()

        if (
            has_dave
            and not packet.is_silence()
            and packet.decrypted_data is not None
            and self.vc._connection.dave_session is not None
            and self.vc._connection.dave_session.ready
        ):
            if member is None:
                # Can't decrypt without a user ID — drop packet
                return None

            try:
                packet.decrypted_data = self.vc._connection.dave_session.decrypt(
                    member.id,
                    MediaType.audio,
                    bytes(packet.decrypted_data),
                )
            except (ValueError, Exception) as e:
                # Log first occurrence, then throttle
                _patched_process_packet._fail_count = getattr(
                    _patched_process_packet, "_fail_count", 0
                ) + 1
                count = _patched_process_packet._fail_count
                if count in (1, 10, 50, 200) or count % 500 == 0:
                    log.warning(
                        "DAVE decrypt failed (count=%d), dropping packet: %s",
                        count,
                        e,
                    )
                return None

        if not self.sink.wants_opus():
            packet, pcm = self._decode_packet(packet)

        data = VoiceData(packet, member, pcm=pcm)
        self._last_seq = packet.sequence
        self._last_ts = packet.timestamp

        return data

    PacketDecoder._process_packet = _patched_process_packet
    logger.info("Applied DAVE decrypt safety patch to PacketDecoder._process_packet")
