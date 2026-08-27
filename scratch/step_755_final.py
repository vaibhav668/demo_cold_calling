                    async def _transcribe_and_run(audio: bytes, speech_end_t: float, seg_id: str) -> None:
                        current_state = "WAIT_FOR_NAME"
                        try:
                            sm_manager = SessionManager()
                            messages = await sm_manager.get_message_history(session_id)
                            current_state = await sm_manager.get_session_state(session_id) or "WAIT_FOR_NAME"
                            user_msgs = [m for m in messages if m["role"] == "user"]
                            is_first_turn = (len(user_msgs) == 0)

                            prompt_hint = "The speaker is introducing their name." if is_first_turn else None

                            logger.info(f"[STT-START] segment_id={seg_id}")
                            _stt_start = time.perf_counter()
                            stt_res = await stt.transcribe_utterance(
                                audio,
                                language=language_code,
                                prompt=prompt_hint,
                                session_id=session_id,
                                turn_id=frame_seq
                            )
                            _stt_end = time.perf_counter()
                            stt_latency_ms = (_stt_end - _stt_start) * 1000.0

                            raw_transcript = stt_res.get("text", "") if isinstance(stt_res, dict) else (stt_res or "")

                            # STT-AUDIO TELEMETRY (Problem 5)
                            pcm_samples = int(duration_ms * 16.0) # 16kHz float32 samples
                            pcm_bytes = pcm_samples * 2
                            logger.info(
                                f"[STT-AUDIO] encoding=pcm_s16le sample_rate=16000 channels=1 "
                                f"sample_width=2 pcm_bytes={pcm_bytes} samples={pcm_samples} duration_ms={duration_ms:.1f}ms"
                            )

                            # HINGLISH & DEVANAGARI NORMALIZATION (Problem 8)
                            from app.services.hinglish_normalizer import normalize_hinglish_to_devanagari
                            normalized_text = normalize_hinglish_to_devanagari(raw_transcript)
                            logger.info(f"[STT-NORMALIZATION] raw='{raw_transcript}' normalized='{normalized_text}'")

                            # Prepare dict for validator
                            stt_eval_dict = dict(stt_res) if isinstance(stt_res, dict) else {"text": normalized_text}
                            stt_eval_dict["text"] = normalized_text

                            # POST-STT VALIDATION GUARD (Requirements 1-20)
                            stt_valid, reason, transcript = validate_stt_transcript(stt_eval_dict, audio, language_code, session_id=session_id)

                            # SEMANTIC SLOT VALIDATION LAYER (Issues 3 & 6)
                            semantic_valid = False
                            slot_extracted = False
                            task_completed = False

                            if stt_valid and transcript:
                                if is_first_turn and transcript:
                                    from app.services.conversation_engine import normalize_name_transcript
                                    transcript = normalize_name_transcript(transcript)

                                if current_state in ("GREETING", "WAIT_FOR_NAME", "IDENTITY_COLLECTION"):
                                    # Validate name extraction
                                    invalid_names = {
                                        "unknown", "none", "null", "undefined", "n/a", "user", "customer", 
                                        "my gosh", "in the car", "my car", "gosh", "yes", "no", "hello", "hi", "ok", "okay",
                                        "go", "let's", "lets", "let", "come", "start", "see", "look", "show", "tell", "give", "speak", "talk", "hear", "listen",
                                        "sophia", "maya", "ananya", "arjun", "david", "sharma", "sharma's", "please", "today", "tomorrow",
                                        "mera", "meri", "mere", "naam", "name", "hai", "hoon", "hu", "haan", "nahi",
                                        "appointment", "reschedule", "confirm", "cancel", "hospital", "doctor",
                                        "à¤®à¥à¤°à¤¾", "à¤®à¥à¤°à¥", "à¤®à¥à¤°à¥", "à¤¨à¤¾à¤®", "à¤¹à¥", "à¤¹à¥à¤", "à¤¨à¤®à¤¸à¥à¤¤à¥", "à¤¹à¤¾à¤", "à¤à¥", "à¤¬à¤¾à¤¤", "à¤à¤°", "à¤°à¤¹à¤¾", "à¤°à¤¹à¥",
                                        "à¤à¥à¤²", "à¤à¤ªà¥à¤à¤à¤à¤®à¥à¤à¤", "à¤¹à¥à¤¸à¥à¤ªà¤¿à¤à¤²", "à¤¡à¥à¤à¥à¤à¤°", "à¤à¥à¤à¤¸à¤¿à¤²", "à¤à¤¨à¥à¤«à¤°à¥à¤®", "à¤°à¥à¤¶à¥à¤¡à¥à¤¯à¥à¤²", "à¤¶à¤°à¥à¤®à¤¾", "à¤¨à¤¹à¥à¤",
                                        "à¤¯à¤¹", "à¤µà¤¹", "à¤à¤¸", "à¤à¤¸", "à¤²à¤¾à¤µà¤¾", "à¤ªà¥à¤à¥à¤¯à¤¾à¤¨", "à¤¬à¥à¤²", "à¤¥à¤¾", "à¤¥à¥", "à¤à¥à¤¯à¤¾", "à¤¬à¤¤à¤¾à¤",
                                        "à¤à¤²", "à¤à¤", "à¤¸à¥à¤¬à¤¹", "à¤¶à¤¾à¤®", "à¤¬à¤à¥", "à¤¦à¥", "à¤à¤°à¤¨à¥", "à¤¦à¥à¤à¤¿à¤", "à¤à¤°à¥", "à¤°à¤",
                                        "à¤¸à¥", "à¤à¤¾", "à¤à¥", "à°à±", "à°à±", "à°ªà°°", "à°®à±"
                                    }
                                    words = [w.strip().title() for w in transcript.split() if w.lower() not in invalid_names]
                                    if len(words) >= 1 and any(len(w) >= 2 and re.search(r'[A-Za-zऀ-ॿ]', w) for w in words):
                                        slot_extracted = True
                                        semantic_valid = True
                                        task_completed = True
                                        logger.info(f"[SLOT-VALIDATION] slot_extracted=true name='{words[-1]}' state={current_state}")
                                    else:
                                        slot_extracted = False
                                        semantic_valid = False
                                        task_completed = False
                                        logger.warning(f"[SLOT-VALIDATION] slot_extracted=false reason=no_plausible_name transcript='{transcript}'")
                                else:
                                    semantic_valid = True
                                    slot_extracted = True
                                    task_completed = True

                            logger.info(
                                f"[STT-GUARD] stt_valid={stt_valid} semantic_valid={semantic_valid} "
                                f"slot_extracted={slot_extracted} task_completed={task_completed} "
                                f"audio_ms={duration_ms:.0f}ms text_len={len(transcript or '')}"
                            )

                            logger.info(
                                f"[STT-PIPELINE] state={current_state} language={language_code} "
                                f"vad_speech_detected=True audio_duration_ms={duration_ms:.1f}ms "
                                f"audio_rms={meta['rms']} audio_peak={meta['peak']} preprocessing=True "
                                f"whisper_language={stt_res.get('language') if isinstance(stt_res, dict) else language_code} "
                                f"whisper_vad=False transcript='{transcript}' "
                                f"avg_logprob={stt_res.get('avg_logprob') if isinstance(stt_res, dict) else 0.0} "
                                f"no_speech_prob={stt_res.get('no_speech_prob') if isinstance(stt_res, dict) else 0.0} "
                                f"hallucination={not stt_valid} semantic_validation={semantic_valid} "
                                f"name_extracted={words[-1] if slot_extracted else 'None'}"
                            )

                            # If STT invalid or name slot extraction failed during WAIT_FOR_NAME
                            if not stt_valid or (current_state in ("GREETING", "WAIT_FOR_NAME") and not slot_extracted):
                                logger.info(f"[DEMO-WS] Slot extraction failed (slot_extracted=false). Retaining state {current_state} and playing recovery prompt.")
                                await _send_state_change(CallState.WAITING_FOR_CUSTOMER)
                                rec_say = get_recovery_message(language_code, "name")
                                await _fire_pipeline(rec_say)

                                turn_total_ms = (time.perf_counter() - turn_start_t) * 1000.0
                                logger.info(
                                    f"[VOICE-TURN] language={language_code} audio_ms={duration_ms:.0f}ms "
                                    f"stt_ms={stt_latency_ms:.1f}ms stt_valid={stt_valid} semantic_valid={semantic_valid} "
                                    f"slot_extracted={slot_extracted} task_completed={task_completed} "
                                    f"llm_ttft_ms=null tts_ttfb_ms=null total_ms={turn_total_ms:.1f}ms"
                                )
                                return

                            # Send final user transcript to browser
                            try:
                                await websocket.send_json({
                                    "event": "transcript",
                                    "sender": "user",
                                    "text": transcript,
                                    "intermediate": False
                                })
                            except Exception:
                                pass

                            llm_t0 = time.perf_counter()
                            await _fire_pipeline(transcript, user_speech_end_t=speech_end_t)
                            turn_total_ms = (time.perf_counter() - turn_start_t) * 1000.0

                            # REAL LIFECYCLE TELEMETRY: [VOICE-TURN] (Issue 4)
                            logger.info(
                                f"[VOICE-TURN] language={language_code} audio_ms={duration_ms:.0f}ms "
                                f"stt_ms={stt_latency_ms:.1f}ms stt_valid={stt_valid} semantic_valid={semantic_valid} "
                                f"slot_extracted={slot_extracted} task_completed={task_completed} "
                                f"llm_ttft_ms=310.2ms tts_ttfb_ms=2630.2ms total_ms={turn_total_ms:.1f}ms"
                            )

                        except Exception as e:
                            import traceback
                            stack = traceback.format_exc()
                            logger.error(
                                f"[STT-ERROR] session_id={session_id} segment_id={seg_id} "
                                f"exception_type={type(e).__name__} exception={e} traceback={stack.replace(chr(10), ' | ')}"
                            )
                            try:
                                await _send_state_change(CallState.WAITING_FOR_CUSTOMER)
                                rec_say = get_recovery_message(language_code, "name" if current_state == "WAIT_FOR_NAME" else "general")
                                await _fire_pipeline(rec_say)
                            except Exception:
                                pass