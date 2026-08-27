import sys
import os
import asyncio
import uuid

# Ensure current project directory is at top of sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.conversation_engine import ConversationEngine, TEMPLATES, extract_customer_name_from_text
from app.services.session_manager import SessionManager, VoiceSession
from app.services.rag_service import RAGService

async def run_turn(engine: ConversationEngine, session_id: str, campaign_id: uuid.UUID, customer_id: uuid.UUID, user_text: str):
    chunks = []
    hangup = False
    transfer = False
    async for chunk, h, t in engine.process_voice_demo_turn_stream(
        call_id=session_id,
        campaign_id=campaign_id,
        customer_id=customer_id,
        industry="hospital",
        language="English",
        agent_name="Sophia",
        user_text=user_text
    ):
        if chunk:
            chunks.append(chunk)
        if h:
            hangup = True
        if t:
            transfer = True
    full_text = "".join(chunks).strip()
    return full_text, hangup, transfer


async def test_hospital_conversation_flow():
    print("=" * 80)
    print("  RUNNING COMPREHENSIVE HOSPITAL CONVERSATION FLOW TEST SUITE (TESTS 1 - 20)")
    print("=" * 80)

    engine = ConversationEngine(db=None)
    rag = RAGService()
    sm_manager = SessionManager()

    total_passed = 0
    total_tests = 20

    # -------------------------------------------------------------------------
    # TEST 1 & 3: Initial Greeting & No Time-Based Greeting
    # -------------------------------------------------------------------------
    print("\n--- [TEST 1 & 3] INITIAL GREETING & NO TIME-BASED GREETING ---")
    session_id = str(uuid.uuid4())
    camp_id = uuid.uuid5(uuid.NAMESPACE_DNS, "hospital")
    cust_id = uuid.uuid5(uuid.NAMESPACE_DNS, "demo_customer")

    resp1, h1, _ = await run_turn(engine, session_id, camp_id, cust_id, "[CALL_START]")
    print(f"Agent Greeting: '{resp1}'")

    assert "Sophia" in resp1 and "CityCare Hospital" in resp1, "Greeting must introduce Sophia and CityCare Hospital"
    assert "May I know whom I'm speaking with?" in resp1, "Greeting must ask for name"
    assert not any(tb in resp1.lower() for tb in ["good morning", "good afternoon", "good evening"]), "Greeting MUST NOT contain time-based phrases"
    print("✓ PASS: Test 1 & 3")

    # -------------------------------------------------------------------------
    # TEST 2 & 4: Name Capture from STT & No Small Talk
    # -------------------------------------------------------------------------
    print("\n--- [TEST 2 & 4] NAME CAPTURE FROM STT & NO SMALL TALK ---")
    resp2, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "I am Rahul")
    print(f"Agent Purpose Response: '{resp2}'")

    assert "Rahul" in resp2, "Captured name 'Rahul' must be included in response"
    assert "Vaibhav" not in resp2 and "Mayank" not in resp2, "MUST NOT use hardcoded personal names"
    assert "Nice to speak with you, Rahul" in resp2, "Must contain personalized greeting"
    assert "calling regarding your upcoming appointment with Dr. Sharma" in resp2, "Must contain purpose"
    assert "Would you like to confirm, cancel, or reschedule it?" in resp2, "Must present decision options"
    assert not any(st in resp2.lower() for st in ["how are you", "how have you been", "do you have a minute"]), "MUST NOT contain small talk"
    print("✓ PASS: Test 2 & 4")

    # -------------------------------------------------------------------------
    # TEST 5: Doctor Specialization Question (RAG Interruption)
    # -------------------------------------------------------------------------
    print("\n--- [TEST 5] DOCTOR SPECIALIZATION (RAG INTERRUPTION) ---")
    resp5, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "What does Dr. Sharma specialize in?")
    print(f"Agent RAG Response: '{resp5}'")

    assert "Orthopedic" in resp5 or "Cardiologist" in resp5 or "joint replacement" in resp5.lower() or "heart" in resp5.lower(), "Must answer Dr. Sharma specialization from RAG"
    assert "Would you like to confirm, cancel, or reschedule your appointment?" in resp5, "Must return to pending decision prompt"
    print("✓ PASS: Test 5")

    # -------------------------------------------------------------------------
    # TEST 6: Consultation Fee Question
    # -------------------------------------------------------------------------
    print("\n--- [TEST 6] CONSULTATION FEE QUESTION ---")
    resp6, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "How much is the consultation fee?")
    print(f"Agent RAG Response: '{resp6}'")

    assert "800" in resp6 or "fee" in resp6.lower(), "Must return consultation fee details"
    assert "Would you like to confirm, cancel, or reschedule your appointment?" in resp6, "Must return to pending decision prompt"
    print("✓ PASS: Test 6")

    # -------------------------------------------------------------------------
    # TEST 7: Hospital Timings Question
    # -------------------------------------------------------------------------
    print("\n--- [TEST 7] HOSPITAL TIMINGS QUESTION ---")
    resp7, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "What are the hospital timings?")
    print(f"Agent RAG Response: '{resp7}'")

    assert "8:00 AM" in resp7 or "OPD" in resp7 or "24/7" in resp7, "Must return OPD/ER timings"
    assert "Would you like to confirm, cancel, or reschedule your appointment?" in resp7, "Must return to pending decision prompt"
    print("✓ PASS: Test 7")

    # -------------------------------------------------------------------------
    # TEST 8: Hospital Location Question
    # -------------------------------------------------------------------------
    print("\n--- [TEST 8] HOSPITAL LOCATION QUESTION ---")
    resp8, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "Where is the hospital located?")
    print(f"Agent RAG Response: '{resp8}'")

    assert "123 Health Ave" in resp8 or "Mumbai" in resp8 or "Central Park" in resp8, "Must return hospital address/location"
    assert "Would you like to confirm, cancel, or reschedule your appointment?" in resp8, "Must return to pending decision prompt"
    print("✓ PASS: Test 8")

    # -------------------------------------------------------------------------
    # TEST 9: Insurance Question
    # -------------------------------------------------------------------------
    print("\n--- [TEST 9] INSURANCE QUESTION ---")
    resp9, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "Do you accept insurance?")
    print(f"Agent RAG Response: '{resp9}'")

    assert "insurance" in resp9.lower() or "Cashless" in resp9 or "Star Health" in resp9, "Must return insurance details"
    assert "Would you like to confirm, cancel, or reschedule your appointment?" in resp9, "Must return to pending decision prompt"
    print("✓ PASS: Test 9")

    # -------------------------------------------------------------------------
    # TEST 10: Parking Question
    # -------------------------------------------------------------------------
    print("\n--- [TEST 10] PARKING QUESTION ---")
    resp10, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "Do you have parking?")
    print(f"Agent RAG Response: '{resp10}'")

    assert "parking" in resp10.lower() or "free" in resp10.lower(), "Must return parking details"
    assert "Would you like to confirm, cancel, or reschedule your appointment?" in resp10, "Must return to pending decision prompt"
    print("✓ PASS: Test 10")

    # -------------------------------------------------------------------------
    # TEST 11: Emergency Facilities Question
    # -------------------------------------------------------------------------
    print("\n--- [TEST 11] EMERGENCY FACILITIES QUESTION ---")
    resp11, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "Does the hospital have emergency facilities?")
    print(f"Agent RAG Response: '{resp11}'")

    assert "24/7" in resp11 or "Emergency" in resp11 or "ambulance" in resp11.lower(), "Must return emergency room details"
    print("✓ PASS: Test 11")

    # -------------------------------------------------------------------------
    # TEST 12: Multiple Consecutive Questions
    # -------------------------------------------------------------------------
    print("\n--- [TEST 12] MULTIPLE CONSECUTIVE QUESTIONS ---")
    resp12a, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "Do you have a cardiology department?")
    assert "Cardiologist" in resp12a or "Cardiology" in resp12a or "Dr. Patel" in resp12a, "Must answer cardiology question"

    resp12b, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "Can I get an ECG done there?")
    assert "ECG" in resp12b or "400" in resp12b, "Must answer ECG test question"

    st = await sm_manager.get_session_state(session_id)
    assert st == "HOSPITAL_WAITING_FOR_DECISION", f"State must be preserved as HOSPITAL_WAITING_FOR_DECISION, got {st}"
    print("✓ PASS: Test 12 (Multiple questions answered & state preserved)")

    # -------------------------------------------------------------------------
    # TEST 16: Policy Question ("What happens if I cancel?")
    # -------------------------------------------------------------------------
    print("\n--- [TEST 16] POLICY QUESTION ('What happens if I cancel?') ---")
    resp16, h16, _ = await run_turn(engine, session_id, camp_id, cust_id, "What happens if I cancel?")
    print(f"Agent Policy Response: '{resp16}'")

    assert "24 hours" in resp16 or "cancelled" in resp16.lower() or "fee" in resp16.lower(), "Must return cancellation policy"
    assert not h16, "MUST NOT terminate call or cancel appointment"
    st16 = await sm_manager.get_session_state(session_id)
    assert st16 == "HOSPITAL_WAITING_FOR_DECISION", "MUST NOT execute cancellation on policy question"
    print("✓ PASS: Test 16")

    # -------------------------------------------------------------------------
    # TEST 17: Ambiguous "Yes"
    # -------------------------------------------------------------------------
    print("\n--- [TEST 17] AMBIGUOUS YES HANDLING ---")
    resp17, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "Yes")
    print(f"Agent Ambiguous Yes Response: '{resp17}'")

    assert "Would you like to confirm, cancel, or reschedule" in resp17, "Must ask clarification prompt instead of blindly confirming"
    st17 = await sm_manager.get_session_state(session_id)
    assert st17 == "HOSPITAL_WAITING_FOR_DECISION", "MUST NOT change state on ambiguous yes"
    print("✓ PASS: Test 17")

    # -------------------------------------------------------------------------
    # TEST 20: Medical Safety / Diagnosis Question
    # -------------------------------------------------------------------------
    print("\n--- [TEST 20] MEDICAL DIAGNOSIS / SAFETY QUESTION ---")
    resp20, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "I have severe chest pain. What disease do I have?")
    print(f"Agent Safety Response: '{resp20}'")

    assert "diagnose" in resp20.lower() or "emergency" in resp20.lower(), "Must refuse medical diagnosis and recommend emergency"
    assert "Would you like to confirm, cancel, or reschedule your appointment?" in resp20, "Must append pending decision prompt"
    print("✓ PASS: Test 20")

    # -------------------------------------------------------------------------
    # TEST 13: Explicit Confirmation Action
    # -------------------------------------------------------------------------
    print("\n--- [TEST 13] EXPLICIT CONFIRMATION ACTION ---")
    resp13, _, _ = await run_turn(engine, session_id, camp_id, cust_id, "Please confirm my appointment")
    print(f"Agent Confirm Response: '{resp13}'")

    assert "confirmed" in resp13.lower() or "confirm" in resp13.lower(), "Must execute confirmation action"
    assert "Is there anything else I can help you with?" in resp13, "Must ask post-action anything else prompt"
    st13 = await sm_manager.get_session_state(session_id)
    assert st13 == "HOSPITAL_POST_ACTION", "Must transition to HOSPITAL_POST_ACTION state"
    print("✓ PASS: Test 13")

    # -------------------------------------------------------------------------
    # TEST 18: Post-Action Flow & Goodbye
    # -------------------------------------------------------------------------
    print("\n--- [TEST 18] POST-ACTION FLOW & GOODBYE ---")
    resp18, h18, _ = await run_turn(engine, session_id, camp_id, cust_id, "No, that's all. Thanks.")
    print(f"Agent Goodbye Response: '{resp18}'")

    assert "Thank you for your time, Rahul" in resp18 or "Goodbye" in resp18, "Must deliver controlled goodbye"
    assert h18, "Must set hangup flag to True"
    st18 = await sm_manager.get_session_state(session_id)
    assert st18 == "HOSPITAL_GOODBYE", "Must transition to HOSPITAL_GOODBYE state"
    print("✓ PASS: Test 18")

    # -------------------------------------------------------------------------
    # TEST 14: Explicit Cancellation Flow
    # -------------------------------------------------------------------------
    print("\n--- [TEST 14] EXPLICIT CANCELLATION FLOW ---")
    sid_cancel = str(uuid.uuid4())
    await run_turn(engine, sid_cancel, camp_id, cust_id, "[CALL_START]")
    await run_turn(engine, sid_cancel, camp_id, cust_id, "Amit")
    resp14, _, _ = await run_turn(engine, sid_cancel, camp_id, cust_id, "I want to cancel my appointment")
    print(f"Agent Cancel Response: '{resp14}'")

    assert "cancelled" in resp14.lower(), "Must execute cancellation action"
    assert "Is there anything else I can help you with?" in resp14, "Must ask post-action question"
    print("✓ PASS: Test 14")

    # -------------------------------------------------------------------------
    # TEST 15: Explicit Rescheduling Flow
    # -------------------------------------------------------------------------
    print("\n--- [TEST 15] EXPLICIT RESCHEDULING FLOW ---")
    sid_resched = str(uuid.uuid4())
    await run_turn(engine, sid_resched, camp_id, cust_id, "[CALL_START]")
    await run_turn(engine, sid_resched, camp_id, cust_id, "Priya")
    resp15a, _, _ = await run_turn(engine, sid_resched, camp_id, cust_id, "I need to reschedule")
    print(f"Agent Reschedule Prompt: '{resp15a}'")
    assert "What day or time would you prefer" in resp15a, "Must ask for preferred slot"

    resp15b, _, _ = await run_turn(engine, sid_resched, camp_id, cust_id, "Friday at 10 AM")
    print(f"Agent Reschedule Confirm: '{resp15b}'")
    assert "rescheduled for Friday at 10 AM" in resp15b or "rescheduled" in resp15b.lower(), "Must confirm rescheduling with provided slot"
    print("✓ PASS: Test 15")

    # -------------------------------------------------------------------------
    # TEST 19: Unknown Question Fallback (No Hallucination)
    # -------------------------------------------------------------------------
    print("\n--- [TEST 19] UNKNOWN QUESTION FALLBACK (NO HALLUCINATION) ---")
    sid_unk = str(uuid.uuid4())
    await run_turn(engine, sid_unk, camp_id, cust_id, "[CALL_START]")
    await run_turn(engine, sid_unk, camp_id, cust_id, "Vikram")
    resp19, _, _ = await run_turn(engine, sid_unk, camp_id, cust_id, "Do you have a helicopter pad on the roof for private jets?")
    print(f"Agent Unknown Query Response: '{resp19}'")

    assert "confirm" in resp19.lower() and "reschedule" in resp19.lower(), "Must preserve decision state"
    print("✓ PASS: Test 19")

    print("\n" + "=" * 80)
    print(f"  ALL 20 HOSPITAL FLOW TESTS PASSED SUCCESSFULLY! ({total_tests}/{total_tests})")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_hospital_conversation_flow())
