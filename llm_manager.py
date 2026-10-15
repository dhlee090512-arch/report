# llm_manager.py
import re
import time
import pandas as pd
import numpy as np
from groq import Groq, RateLimitError
from google import genai
from config import (
    GEMINI_API_KEY_1, GEMINI_API_KEY_2, 
    GROQ_API_KEY_1, GROQ_API_KEY_2, 
    TEST_MODE
)

# =========================================================
# [숫자 포맷 유틸리티]
# =========================================================
def fmt_price(val, is_krw=True, show_decimal=False):
    if val is None or pd.isna(val):
        return "0원" if is_krw else "$0.00"
    try:
        f_val = float(val)
    except Exception:
        return "0원" if is_krw else "$0.00"

    if is_krw:
        if show_decimal:
            return f"{f_val:,.2f}원"
        else:
            return f"{int(round(f_val)):,}원"
    else:
        if show_decimal or not f_val.is_integer():
            return f"${f_val:,.2f}"
        else:
            return f"${int(round(f_val)):,}"

def fmt_num(val):
    if val is None or pd.isna(val):
        return "0"
    try:
        f_val = float(val)
        if f_val.is_integer():
            return f"{int(f_val):,}"
        else:
            return f"{f_val:,.2f}".rstrip('0').rstrip('.')
    except Exception:
        return "0"

# =========================================================
# 🎯 호가 단위(Tick Size) 자동 보정
# =========================================================
def adjust_to_tick_size(price, is_krw=True):
    if price is None or price <= 0:
        return price
    if not is_krw:
        if price >= 1.0:
            return round(price, 2)
        else:
            return round(price, 4)

    p = float(price)
    if p < 2000: tick = 1
    elif p < 5000: tick = 5
    elif p < 20000: tick = 10
    elif p < 50000: tick = 50
    elif p < 200000: tick = 100
    elif p < 500000: tick = 500
    else: tick = 1000
    return int(round(p / tick) * tick)

# =========================================================
# 🛡️ ATR(14) 기반 손절 버퍼 검증
# =========================================================
def calculate_atr(df, period=14):
    try:
        high = df['High']
        low = df['Low']
        close_prev = df['Close'].shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return float(tr.rolling(period, min_periods=1).mean().iloc[-1])
    except Exception:
        return 0.0

def validate_stop_loss_with_atr(entry_price, stop_loss_price, atr_val, is_krw=True):
    if not entry_price or not stop_loss_price or atr_val <= 0:
        return stop_loss_price
    min_buffer = atr_val * 1.0
    max_allowed_stop = entry_price - min_buffer
    if stop_loss_price > max_allowed_stop:
        return adjust_to_tick_size(max_allowed_stop, is_krw)
    return adjust_to_tick_size(stop_loss_price, is_krw)

# =========================================================
# 📈 Wilder's RSI(14) 및 Signal(9) 계산
# =========================================================
def calculate_wilder_rsi(series, period=14, signal_period=9):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_signal = rsi.rolling(signal_period, min_periods=1).mean()
    return rsi, rsi_signal

def extract_peaks_and_troughs(df_60, is_krw=True):
    try:
        closes = df_60['Close'].values
        peaks = []
        troughs = []
        for i in range(2, len(closes) - 2):
            if closes[i] > closes[i-1] and closes[i] > closes[i-2] and closes[i] > closes[i+1] and closes[i] > closes[i+2]:
                peaks.append(closes[i])
            elif closes[i] < closes[i-1] and closes[i] < closes[i-2] and closes[i] < closes[i+1] and closes[i] < closes[i+2]:
                troughs.append(closes[i])
        last_low = fmt_price(troughs[-1], is_krw) if troughs else "데이터 미수집"
        last_high = fmt_price(peaks[-1], is_krw) if peaks else "데이터 미수집"
        return f"최근 반등 저점({last_low}) -> 최근 저항 고점({last_high}) / 주요 마디점 {len(troughs)}개 형성"
    except Exception:
        return "파동 마디점 안정화 진행 중"

def parse_price_from_text(text, key_prefix, is_krw=True, current_price=0.0):
    if not text:
        return None
    try:
        match = re.search(rf'{key_prefix}\s*:\s*([^\n]+)', text)
        if match:
            raw_str = match.group(1).strip()
            digits = re.findall(r'[\d\.]+', raw_str.replace(',', ''))
            if digits:
                val = float(digits[0])
                if val > 0:
                    return adjust_to_tick_size(val, is_krw)
    except Exception:
        pass
    return None

def sanitize_text(text):
    if not text: return ""
    cleaned = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]', '', str(text))
    return cleaned.replace("파싱_", "").strip()

# =========================================================
# 🏛️ [3단계 AI 다중화 매니저]
# =========================================================
class MultiLLMManager:
    def __init__(self, gemini_keys, groq_keys):
        self.gemini_keys = [k.strip() for k in gemini_keys if k and k.strip()]
        self.gemini_clients = []
        for i, k in enumerate(self.gemini_keys):
            try:
                c = genai.Client(api_key=k)
                self.gemini_clients.append(c)
                print(f"✅ [Gemini Client #{i+1}] 초기화 완료 (gemini-3.5-flash-lite)")
            except Exception as e:
                print(f"⚠️ Gemini Key #{i+1} 초기화 실패: {e}")

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

        # [3순위: Groq 백업 풀]
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

llm_mgr = MultiLLMManager(
    gemini_keys=[GEMINI_API_KEY_1, GEMINI_API_KEY_2], 
    groq_keys=[GROQ_API_KEY_1, GROQ_API_KEY_2]
)
