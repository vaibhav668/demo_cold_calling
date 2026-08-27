    # ââ Main receive loop ââââââââââââââââââââââââââââââââââââââââââââââââââââ
    frame_seq = 0
    cust_speaking_start_t = 0.0
    last_voice_energy_t = 0.0
    pre_roll_buffer = collections.deque(maxlen=10)  # 200ms rolling pre-roll buffer (10 x 20ms chunks)

    try:
        logger.info(f"[DEMO-WS] Firing sub-second greeting pipeline for session {session_id}")
        await _fire_pipeline("[CALL_START]")

        while not sm.is_terminal():
            data = await websocket.receive()

            if data.get("type") == "websocket.disconnect":
                logger.info(f"[DEMO-WS] Browser disconnected for session {session_id}")
                break

            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    event = msg.get("event")
                    if event == "ping":
                        await websocket.send_json({"event": "pong"})
                    elif event == "stop":
                        logger.info(f"[DEMO-WS] Stop event received for session {session_id}")
                        break
                    elif event == "playback_ended":
                        logger.info(f"[MIC-SYNC] Browser playback completed for session {session_id}. Resetting VAD and setting WAITING_FOR_CUSTOMER.")
                        vad.reset()
                        await _send_state_change(CallState.WAITING_FOR_CUSTOMER)
                except Exception:
                    pass

            elif "bytes" in data:
                pcm_data = data["bytes"]  # 16kHz 16-bit PCM mono (32 bytes/ms)
                frame_seq += 1
                frame_ms = len(pcm_data) / 32.0

                import audioop
                try:
                    rms_val = audioop.rms(pcm_data, 2)
                    peak_val = audioop.max(pcm_data, 2)
                except Exception:
                    rms_val, peak_val = 0, 0

                if frame_seq % 50 == 1:
                    logger.info(
                        f"[AUDIO-INGEST] session_id={session_id} seq={frame_seq} bytes={len(pcm_data)} "
                        f"format=pcm_s16le sample_rate=16000 channels=1 duration_ms={frame_ms:.0f}ms rms={rms_val} peak={peak_val}"
                    )

                v_start = time.perf_counter()

                # VAD during AI speech: barge-in detection
                if sm.is_ai_speaking():
                    if not greeting_completed:
                        vad.reset()
                        continue
                    loop_time = loop.time()
                    if loop_time - sm.ai_speech_start_time > 1.2:
                        vad_event = await loop.run_in_executor(None, vad.process_frame, pcm_data)
                        v_elapsed = (time.perf_counter() - v_start) * 1000.0
                        vad_timings.append(v_elapsed)

                        if vad_event == "speech_start":
                            await _barge_in()
                    else:
                        vad.reset()
                    continue

                if sm.state in (
                    CallState.TRANSCRIBING,
                    CallState.THINKING,
                    CallState.GENERATING_RESPONSE,
                    CallState.CALL_COMPLETED,
                    CallState.ERROR,
                ):
                    continue

                loop_time = loop.time()
                if sm.is_waiting():
                    pre_roll_buffer.append(pcm_data)
                    if (loop_time - sm.waiting_start_time < 0.4):
                        vad.reset()
                        continue

                # Normal VAD processing
                vad_event = await loop.run_in_executor(None, vad.process_frame, pcm_data)
                v_elapsed = (time.perf_counter() - v_start) * 1000.0
                vad_timings.append(v_elapsed)

                if sm.state == CallState.CUSTOMER_SPEAKING:
                    utterance_buffer.extend(pcm_data)
                    if rms_val >= 350:
                        last_voice_energy_t = loop_time

                if vad_event == "speech_start":
                    if sm.is_waiting():
                        logger.info(f"[DEMO-WS] Speech start detected for session {session_id} (pre-roll buffer={len(pre_roll_buffer)} chunks)")
                        utterance_buffer.clear()
                        # Prepend 200ms rolling pre-roll audio to prevent initial phoneme clipping
                        for pre_chunk in pre_roll_buffer:
                            utterance_buffer.extend(pre_chunk)
                        pre_roll_buffer.clear()
                        utterance_buffer.extend(pcm_data)

                        last_intermediate_stt_len = 0
                        cust_speaking_start_t = loop_time
                        last_voice_energy_t = loop_time
                        vad.provider._in_speech = True
                        if hasattr(vad.provider, '_speech_confirmed'):
                            vad.provider._speech_confirmed = True
                        await _send_state_change(CallState.CUSTOMER_SPEAKING)

                should_finalize = False
                if sm.state == CallState.CUSTOMER_SPEAKING:
                    if vad_event == "speech_end":
                        should_finalize = True
                        logger.info(f"[VAD-EVENT] speech_end fired cleanly for session {session_id}")
                    elif cust_speaking_start_t > 0:
                        speech_dur = loop_time - cust_speaking_start_t
                        silence_dur = loop_time - last_voice_energy_t
                        if speech_dur > 0.3 and silence_dur >= 0.6:
                            should_finalize = True
                            logger.info(f"[VAD-TIMEOUT] Finalizing utterance on 600ms silence timeout (speech_dur={speech_dur:.2f}s, silence_dur={silence_dur:.2f}s)")
                        elif speech_dur >= 8.0:
                            should_finalize = True
                            logger.info(f"[VAD-TIMEOUT] Finalizing utterance on 8.0s max duration limit (speech_dur={speech_dur:.2f}s)")

                if should_finalize:
                    cust_speaking_start_t = 0.0
                    last_voice_energy_t = 0.0
                    turn_start_t = time.perf_counter()
                    user_speech_end_t = time.perf_counter()
                    utterance_bytes = bytes(utterance_buffer)
                    utterance_buffer.clear()
                    pre_roll_buffer.clear()
                    last_intermediate_stt_len = 0
                    vad.reset()

                    await _safe_cancel_task(intermediate_stt_task)
                    intermediate_stt_task = None

                    from app.services.speech.stt.faster_whisper_provider import calculate_pcm_metadata
                    meta = calculate_pcm_metadata(utterance_bytes, sample_rate=16000, channels=1, sample_width=2)
                    duration_ms = meta["duration_ms"]
                    segment_id = str(uuid.uuid4())[:8]

                    logger.info(
                        f"[STT-PREPROCESS] source: bytes={meta['bytes']} samples={meta['samples']} rate=16000 duration={meta['duration_ms']:.1f}ms rms={meta['rms']} peak={meta['peak']} | "
                        f"stt_input: bytes={meta['bytes']} samples={meta['samples']} rate=16000 duration={meta['duration_ms']:.1f}ms"
                    )
                    logger.info(
                        f"[TELEPHONY-AUDIO] encoding=pcm_s16le sample_rate=16000 channels=1 "
                        f"sample_width=2 raw_bytes={meta['bytes']} duration_ms={meta['duration_ms']:.1f}ms"
                    )
                    logger.info(
                        f"[STT-SEGMENT-FINALIZED] segment_id={segment_id} "
                        f"duration_ms={meta['duration_ms']:.1f}ms bytes={meta['bytes']} "
                        f"sample_rate=16000 channels=1 rms={meta['rms']} peak={meta['peak']}"
                    )
                    logger.info(
                        f"[VAD-STT] session_id={session_id} segment_id={segment_id} "
                        f"audio_duration_ms={meta['duration_ms']:.1f}ms buffer_samples={meta['samples']} "
                        f"sample_rate=16000 channels=1 rms={meta['rms']} peak={meta['peak']}"
                    )

                    # PRE-STT VALIDATION GUARD
                    sm_manager_pre = SessionManager()
                    cur_st_pre = await sm_manager_pre.get_session_state(session_id) or "WAIT_FOR_NAME"
                    pre_valid, pre_reason, dur_ms = validate_stt_audio_pre_whisper(utterance_bytes, current_state=cur_st_pre)
                    if not pre_valid:
                        await _send_state_change(CallState.WAITING_FOR_CUSTOMER)
                        rec_say = get_recovery_message(language_code, "name" if cur_st_pre == "WAIT_FOR_NAME" else "general")
                        await _fire_pipeline(rec_say)
                        continue

                    await _send_state_change(CallState.TRANSCRIBING)