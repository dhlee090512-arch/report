# llm_manager.py 내 MultiLLMManager 부분 교체

from config import (
    GEMINI_API_KEY_1, GEMINI_API_KEY_2, 
    GROQ_API_KEY_1, GROQ_API_KEY_2, 
    TEST_MODE
)

class MultiLLMManager:
    def __init__(self, gemini_keys, groq_keys):
        # 1. Gemini 클라이언트 풀 (1순위 Key 1 -> 2순위 Key 2)
        self.gemini_keys = [k.strip() for k in gemini_keys if k and k.strip()]
        self.gemini_clients = []
        for i, k in enumerate(self.gemini_keys):
            try:
                c = genai.Client(api_key=k)
                self.gemini_clients.append(c)
                print(f"✅ [Gemini Client #{i+1}] 초기화 완료 (gemini-3.5-flash-lite)")
            except Exception as e:
                print(f"⚠️ Gemini Key #{i+1} 초기화 실패: {e}")

        # 2. Groq 클라이언트 풀 (3순위 백업)
        self.groq_keys = [k.strip() for k in groq_keys if k and k.strip()]
        self.current_groq_index = 0
        self.groq_client = None
        self._init_groq_client()
        self.last_gemini_call_time = 0

    def _init_groq_client(self):
        if self.groq_keys and self.current_groq_index < len(self.groq_keys):
            try:
                self.groq_client = Groq(api_key=self.groq_keys[self.current_groq_index])
                print(f"✅ [3순위 Groq] Key #{self.current_groq_index + 1} 초기화 완료")
            except Exception as e:
                print(f"⚠️ Groq Key #{self.current_groq_index + 1} 초기화 실패: {e}")
                self.groq_client = None

    def switch_to_next_groq(self):
        self.current_groq_index += 1
        if self.current_groq_index < len(self.groq_keys):
            print(f"🔄 Groq Key #{self.current_groq_index + 1}로 자동 전환 중...")
            self._init_groq_client()
            return True
        else:
            print("🚨 모든 Groq API Key가 소진되었습니다.")
            self.groq_client = None
            return False

    def is_available(self):
        return (len(self.gemini_clients) > 0 or self.groq_client is not None) and not TEST_MODE

    def generate_completion(self, prompt, temperature=0.3, max_tokens=1800):
        if TEST_MODE:
            raise RuntimeError("TEST_MODE가 활성화되어 있어 AI 호출을 스킵합니다.")

        # [1순위 & 2순위: Gemini Key 1번 -> 2번 순차 실행]
        for idx, client in enumerate(self.gemini_clients):
            for attempt in range(2):
                try:
                    elapsed = time.time() - self.last_gemini_call_time
                    if elapsed < 4.2:
                        time.sleep(4.2 - elapsed)

                    print(f"⚡ [{idx+1}순위 Gemini Key #{idx+1}] gemini-3.5-flash-lite 요청 전송 중... (시도 {attempt+1}/2)")
                    self.last_gemini_call_time = time.time()
                    res = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt
                    )
                    if res and res.text:
                        return res.text.strip()
                except Exception as e:
                    err_str = str(e)
                    if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt == 0:
                        print(f"⏳ Gemini Key #{idx+1} 503 서버 혼잡 감지 -> 3초 대기 후 1회 재시도")
                        time.sleep(3)
                        continue
                    print(f"⚠️ Gemini Key #{idx+1} 일시 오류/429/503 ({e}) -> 다음 순위로 전환")
                    break

        # [3순위: Groq 백업 풀 순회]
        groq_model_candidates = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama3-70b-8192", "llama-3.1-8b-instant"]
        while self.groq_client:
            for g_model in groq_model_candidates:
                try:
                    print(f"⚡ [3순위 Groq] Key #{self.current_groq_index + 1} ({g_model}) 요청 전송 중...")
                    res = self.groq_client.chat.completions.create(
                        model=g_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return res.choices[0].message.content.strip()
                except RateLimitError:
                    print(f"🔄 Groq Key #{self.current_groq_index + 1} 쿼터 초과!")
                    break
                except Exception as e:
                    err_text = str(e)
                    if "404" in err_text or "model_not_found" in err_text:
                        continue
                    print(f"⚠️ Groq 호출 예외 ({g_model}): {e}")
                    break

            if not self.switch_to_next_groq():
                break

        raise RuntimeError("모든 AI API(Gemini 1/2번 및 Groq 1/2번) 호출에 실패했습니다.")

# 인스턴스 생성
llm_mgr = MultiLLMManager(
    gemini_keys=[GEMINI_API_KEY_1, GEMINI_API_KEY_2], 
    groq_keys=[GROQ_API_KEY_1, GROQ_API_KEY_2]
)
