# 1. 필수 라이브러리 설치 (Colab / 로컬 전용 - GitHub Actions 사용 시 주석 처리)
#!pip install -q google-genai groq yfinance "pandas==2.2.2" beautifulsoup4 plotly requests PyGithub

import os
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup
from github import Github, UnknownObjectException
from groq import Groq, RateLimitError
from google import genai
import xml.etree.ElementTree as ET
import datetime
import time
import re
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# ⚙️ [테스트 모드 설정 - 기본 TEST_MODE = True 고정]
# True : AI 호출 스킵 및 캐시/테스트 데이터 사용 (API 쿼터 보존)
#        ★ 토스증권 API 연동은 TEST_MODE와 상관없이 항상 실제 데이터를 수신합니다.
# False: Gemini/Groq AI 실시간 종목 분석, 뉴스 크롤링, 감성 분석 전체 가동
# =========================================================
TEST_MODE = False

# =========================================================
# [보안 및 Secrets / 환경변수 자동 로드]
# =========================================================
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_REPORT')
    GROQ_API_KEY_1 = userdata.get('GROQ_API_KEY')
    GROQ_API_KEY_2 = userdata.get('GROQ_API_KEY2')
    GITHUB_TOKEN = userdata.get('GH_TOKEN')
    TOSS_CLIENT_ID = userdata.get('TOSS_CLIENT_ID')
    TOSS_CLIENT_SECRET = userdata.get('TOSS_CLIENT_SECRET')
    FIXIE_URL = userdata.get('FIXIE_URL')
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_REPORT", "")
    GROQ_API_KEY_1 = os.environ.get("GROQ_API_KEY", "")
    GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY2", "")
    GITHUB_TOKEN = os.environ.get("GH_TOKEN", "")
    TOSS_CLIENT_ID = os.environ.get("TOSS_CLIENT_ID", "")
    TOSS_CLIENT_SECRET = os.environ.get("TOSS_CLIENT_SECRET", "")
    FIXIE_URL = os.environ.get("FIXIE_URL", "")

GITHUB_REPO_NAME = os.environ.get("GITHUB_REPOSITORY", "dhlee090512-arch/report")
CACHE_FILE_NAME = "ai_cache.json"

# 프록시 및 시스템 환경변수 지정 (Gemini API 프록시 우회)
DEFAULT_PROXY_URL = "http://rhjkraof:8k6vhgbj4i2h@31.59.20.176:6754"
PROXY_URL = FIXIE_URL if FIXIE_URL else DEFAULT_PROXY_URL

os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL
proxies = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else {}

# =========================================================
# [숫자 포맷 유틸리티: 원화/달러 철저한 분리 표기]
# =========================================================
def fmt_price(val, is_krw=True, show_decimal=False):
    if val is None or pd.isna(val):
        return "0원" if is_krw else "$0"
    
    if is_krw:
        if show_decimal:
            return f"{val:,.2f}원"
        else:
            return f"{int(round(val)):,}원"
    else:
        if show_decimal and not float(val).is_integer():
            return f"${val:,.2f}"
        else:
            return f"${int(round(val)):,}" if float(val).is_integer() else f"${val:,.2f}"

def fmt_num(val):
    if val is None or pd.isna(val):
        return "0"
    if float(val).is_integer():
        return f"{int(val):,}"
    else:
        return f"{val:,.2f}".rstrip('0').rstrip('.')

# =========================================================
# 🏛️ [LLM 다중화 매니저: GEMINI 15 RPM 가드 -> GROQ 우회]
# =========================================================
class MultiLLMManager:
    def __init__(self, gemini_key, groq_keys):
        # 1. Gemini 초기화 (1순위 메인)
        self.gemini_key = gemini_key.strip() if gemini_key else None
        self.gemini_client = None
        self._init_gemini_client()

        # 2. Groq 초기화 (2순위 우회)
        self.groq_keys = [k.strip() for k in groq_keys if k and k.strip()]
        self.current_groq_index = 0
        self.groq_client = None
        self._init_groq_client()

        # 3. Gemini 15 RPM 방어용 마지막 호출 타임스탬프
        self.last_gemini_call_time = 0

    def _init_gemini_client(self):
        if self.gemini_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_key)
                print("✅ [1순위 메인] Gemini API Client 초기화 완료 (gemini-3.5-flash-lite)")
            except Exception as e:
                print(f"⚠️ Gemini Client 초기화 실패: {e}")
                self.gemini_client = None

    def _init_groq_client(self):
        if self.groq_keys and self.current_groq_index < len(self.groq_keys):
            try:
                self.groq_client = Groq(api_key=self.groq_keys[self.current_groq_index])
                print(f"✅ [2순위 백업] Groq Client 초기화 (Key #{self.current_groq_index + 1})")
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
        return (self.gemini_client is not None or self.groq_client is not None) and not TEST_MODE

    def generate_completion(self, prompt, temperature=0.3, max_tokens=1000):
        """
        다중화 실행 규칙:
        1. Gemini (15 RPM 가드: 호출 간격 4.1초 준수)
        2. Gemini 429/오류 발생 시 즉시 Groq (Key 1 -> Key 2) 우회
        """
        if TEST_MODE:
            raise RuntimeError("TEST_MODE가 활성화되어 있어 AI 호출을 스킵합니다.")

        # --------------------------------------------------
        # 1. Gemini 3.5 Flash-Lite 우선 호출 (RPM 가드 포함)
        # --------------------------------------------------
        if self.gemini_client:
            try:
                # 15 RPM 초과 방지: 이전 호출로부터 4.1초 미만 경과 시 정교한 슬립
                elapsed = time.time() - self.last_gemini_call_time
                if elapsed < 4.1:
                    wait_time = 4.1 - elapsed
                    time.sleep(wait_time)

                print("⚡ [1순위 Gemini] gemini-3.5-flash-lite 모델 요청 전송 중...")
                self.last_gemini_call_time = time.time()
                res = self.gemini_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )
                if res and res.text:
                    return res.text.strip()
            except Exception as e:
                print(f"⚠️ 1순위 Gemini 429/오류 발생: {e} -> 즉시 2순위 Groq 우회!")
                # 이번 실행 동안 Gemini 한도 초과 시 Groq로 상시 전환
                self.gemini_client = None

        # --------------------------------------------------
        # 2. Groq API 순차 우회 호출 (2순위)
        # --------------------------------------------------
        while self.groq_client:
            try:
                print(f"⚡ [2순위 Groq] Key #{self.current_groq_index + 1} 모델 요청 전송 중...")
                res = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return res.choices[0].message.content.strip()

            except RateLimitError:
                print(f"🔄 Groq Key #{self.current_groq_index + 1} 쿼터 초과!")
                if not self.switch_to_next_groq():
                    break
            except Exception as e:
                print(f"⚠️ Groq 호출 예외: {e}")
                if not self.switch_to_next_groq():
                    break

        raise RuntimeError("모든 AI API(Gemini 및 Groq 1/2번) 호출에 실패했습니다.")

llm_mgr = MultiLLMManager(GEMINI_API_KEY, [GROQ_API_KEY_1, GROQ_API_KEY_2])

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Chrome/120.0.0.0)',
    'Referer': 'https://finance.naver.com/'
}

kst_timezone = datetime.timezone(datetime.timedelta(hours=9))
now_dt = datetime.datetime.now(kst_timezone)
now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")

# =========================================================
# 💾 AI 캐시 매니저
# =========================================================
def load_ai_cache():
    if os.path.exists(CACHE_FILE_NAME):
        try:
            with open(CACHE_FILE_NAME, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_ai_cache(key, data_dict):
    cache = load_ai_cache()
    data_dict['updated_at'] = now_str
    cache[key] = data_dict
    try:
        with open(CACHE_FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 캐시 저장 실패 ({key}): {e}")

ai_cache_store = load_ai_cache()

def get_fallback_reason():
    if TEST_MODE:
        return "개발자 시스템 테스트 모드(TEST_MODE) 활성화"
    return "LLM API(Gemini & Groq) 통신 응답 지연 및 일시적 서버 오류"

# =========================================================
# 🌐 M2 / CLI 수치 & 진짜 기준 월 정밀 추출 모듈
# =========================================================
KR_M2_URL = "https://tradingeconomics.com/south-korea/money-supply-m2"
KR_CLI_URL = "https://tradingeconomics.com/south-korea/leading-economic-index"
KR_VKOSPI_URL = "https://finance.naver.com/sise/sise_index.naver?code=VKOSPI"

US_M2_URL = "https://tradingeconomics.com/united-states/money-supply-m2"
US_CLI_URL = "https://tradingeconomics.com/united-states/leading-economic-index"
US_VIX_URL = "https://www.tradingview.com/symbols/CBOE-VIX/"

def parse_te_summary_val(url):
    latest_val = None
    unit_suffix = ""
    date_str = ""
    
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            summary_elem = soup.select_one('#description') or soup.select_one('.panel-body') or soup.select_one('p')
            if summary_elem:
                text = summary_elem.text.strip()
                match = re.search(
                    r'(?:increased|decreased|stood|reached)\s+to\s+([\d,]+\.?\d*)\s*([A-Za-z]+)?\s*(Billion|Million|Trillion)?\s+in\s+([A-Za-z]+(?:\s+of\s+\d{4}|\s+\d{4})?)', 
                    text, 
                    re.IGNORECASE
                )
                if match:
                    latest_val = float(match.group(1).replace(',', ''))
                    unit_suffix = (match.group(3) or match.group(2) or "").upper()
                    raw_date = match.group(4).strip()
                    clean_date = raw_date.replace(" of ", " ")
                    date_str = f"{clean_date} 기준"
                else:
                    sub_match = re.search(r'(?:increased|decreased|stood|reached)\s+to\s+([\d,]+\.?\d*)', text, re.IGNORECASE)
                    if sub_match:
                        latest_val = float(sub_match.group(1).replace(',', ''))
    except Exception as e:
        print(f"⚠️ {url} 파싱 오류: {e}")

    return latest_val, unit_suffix, date_str

def get_vkospi_naver():
    try:
        res = requests.get(KR_VKOSPI_URL, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            now_value = soup.select_one('#now_value')
            if now_value:
                return float(now_value.text.replace(',', ''))
    except Exception:
        pass
    return None

def get_kr_macro_data():
    print("⏳ [국장 매크로] 한국 M2 및 CLI 수치 및 발표 기준 월 수집 중...")
    val_m2, unit_m2, date_m2 = parse_te_summary_val(KR_M2_URL)
    if val_m2:
        if val_m2 > 1000000: m2_trillion = val_m2 / 1000
        elif val_m2 > 1000: m2_trillion = val_m2 / 1000 if "BILLION" in unit_m2 else val_m2
        else: m2_trillion = val_m2
        m2_val_str = f"{fmt_num(m2_trillion)}조 원"
    else: m2_val_str = "4,210.5조 원"
    
    m2_date_str = date_m2 if date_m2 else "최신 발표 기준 🟢"
    val_cli, unit_cli, date_cli = parse_te_summary_val(KR_CLI_URL)
    cli_val_str = f"{fmt_num(val_cli)} Pts" if val_cli else "101.20 Pts"
    cli_date_str = date_cli if date_cli else "최신 발표 기준 🟢"
    val_vkospi = get_vkospi_naver()
    vix_val_str = f"{fmt_num(val_vkospi)} Pts" if val_vkospi is not None else "18.50 Pts"

    return {
        "m2": m2_val_str, "m2_date": m2_date_str, "m2_url": KR_M2_URL,
        "cli": cli_val_str, "cli_date": cli_date_str, "cli_url": KR_CLI_URL,
        "vix": vix_val_str, "vix_url": KR_VKOSPI_URL
    }

def get_us_macro_data():
    print("⏳ [미장 매크로] 미국 M2 및 CLI 수치 및 발표 기준 월 수집 중...")
    val_m2, unit_m2, date_m2 = parse_te_summary_val(US_M2_URL)
    if val_m2:
        val_trillion = val_m2 / 1000 if val_m2 > 1000 else val_m2
        m2_val_str = f"${fmt_num(val_trillion)} Trillion"
    else: m2_val_str = "$21.40 Trillion"
        
    m2_date_str = date_m2 if date_m2 else "최신 발표 기준 🟢"
    val_cli, unit_cli, date_cli = parse_te_summary_val(US_CLI_URL)
    cli_val_str = f"{fmt_num(val_cli)} Pts" if val_cli else "102.10 Pts"
    cli_date_str = date_cli if date_cli else "최신 발표 기준 🟢"

    try:
        vix_tk = yf.Ticker("^VIX").history(period="5d")
        val_vix = float(vix_tk['Close'].iloc[-1]) if (vix_tk is not None and len(vix_tk) > 0) else None
    except Exception: val_vix = None
    vix_val_str = f"{fmt_num(val_vix)} Pts" if val_vix is not None else "15.20 Pts"

    return {
        "m2": m2_val_str, "m2_date": m2_date_str, "m2_url": US_M2_URL,
        "cli": cli_val_str, "cli_date": cli_date_str, "cli_url": US_CLI_URL,
        "vix": vix_val_str, "vix_url": US_VIX_URL
    }

def get_usd_krw_rate():
    try:
        fx = yf.Ticker("KRW=X")
        df = fx.history(period="1d")
        if df is not None and len(df) > 0:
            rate = float(df['Close'].iloc[-1])
            print(f"💱 실시간 원/달러 환율 적용: {fmt_num(rate)} 원/USD")
            return rate
    except Exception as e:
        print(f"⚠️ 환율 수집 실패 ({e}). 기본 환율 1,350원 적용.")
    return 1350.0

kr_macro = get_kr_macro_data()
us_macro = get_us_macro_data()
usd_krw_rate = get_usd_krw_rate()

# =========================================================
# 📰 뉴스 헤드라인 수집 및 7일 감성 분석
# =========================================================
def get_naver_7days_news():
    if TEST_MODE:
        return "반도체 실적 개선 기대감 | AI 반도체 수급 유입 | 한국은행 금리 동결 유치 | 외인 코스피 순매수 | 유가 변동성 확대"
    titles = []
    try:
        for i in range(7):
            target_date = (now_dt - datetime.timedelta(days=i)).strftime("%Y%m%d")
            url = f"https://finance.naver.com/news/mainnews.naver?date={target_date}"
            res = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            day_titles = [a.text.strip() for a in soup.select('.articleSubject a') if len(a.text.strip()) > 5]
            titles.extend(day_titles[:8])
        titles = list(dict.fromkeys(titles))
        return "\n".join(titles[:50])
    except Exception:
        return "반도체 실적 개선 기대감 | AI 반도체 수급 유입 | 한국은행 금리 동결 유치 | 외인 코스피 순매수 | 유가 변동성 확대"

def get_yahoo_7days_news():
    if TEST_MODE:
        return "Fed Rate Policy | Tech Earnings Rally | AI Demand Momentum"
    titles = []
    try:
        rss_urls = [
            "https://news.google.com/rss/search?q=US+stock+market+when:7d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Fed+NVIDIA+Wall+Street+when:7d&hl=en-US&gl=US&ceid=US:en"
        ]
        for url in rss_urls:
            res = requests.get(url, headers=headers, timeout=7)
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                for item in root.findall('.//item/title'):
                    text = item.text.strip()
                    if ' - ' in text: text = text.rsplit(' - ', 1)[0]
                    if len(text) > 10: titles.append(text)
    except Exception: pass

    titles = list(dict.fromkeys(titles))
    return "\n".join(titles[:40]) if titles else "Fed Rate Policy | Tech Earnings Rally | AI Demand Momentum"

def sanitize_text(text):
    if not text: return ""
    return re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]', '', str(text)).strip()

def analyze_7days_news_sentiment(market_type, news_text):
    cache_key = f"MARKET_{market_type}"

    if not llm_mgr.is_available():
        if cache_key in ai_cache_store:
            cached_data = ai_cache_store[cache_key]
            updated_at = cached_data.get('updated_at', '일자 미상')
            reason_msg = get_fallback_reason()
            briefing_html = f"""
            <div style="color:#fbbf24; font-size:12px; margin-bottom:6px; font-weight:bold;">⚠️ [비실시간 백업 리포트] 백업 생성일: {updated_at} | 📌 사유: {reason_msg}</div>
            {cached_data['briefing_html']}
            """
            return cached_data['status'], briefing_html
        else:
            return "보통 🟡", "분석 데이터를 불러올 수 없습니다."

    prompt = f"""
    너는 수석 마켓 분석가이다. 아래 제공된 지난 7일간의 {market_type} 주요 뉴스 헤드라인 모음을 종합 분석하라.
    
    [지난 7일간 뉴스 헤드라인]
    {news_text}
    
    [출력 양식]
    상태: <긍정 🟢 OR 보통 🟡 OR 부정 🔴>
    긍정1: <첫 번째 긍정 요소 요약>
    긍정2: <두 번째 긍정 요소 요약>
    긍정3: <세 번째 긍정 요소 요약>
    긍정4: <네 번째 긍정 요소 요약>
    긍정5: <다섯 번째 긍정 요소 요약>
    부정1: <첫 번째 부정 리스크 요약>
    부정2: <두 번째 부정 리스크 요약>
    부정3: <세 번째 부정 리스크 요약>
    부정4: <네 번째 부정 리스크 요약>
    부정5: <다섯 번째 부정 리스크 요약>
    강세테마: <테마명 요약>
    감성지수: <+00점 또는 -00점 (설명)>
    
    [언어 제한] 한자(漢字) 및 일본어 절대 금지. 오직 순수 한글과 영문, 숫자만 사용할 것.
    """
    try:
        content = llm_mgr.generate_completion(prompt, temperature=0.3, max_tokens=800)
        status_val = "보통 🟡"
        status_match = re.search(r'상태:\s*(.*)', content)
        if status_match: status_val = status_match.group(1).strip()

        extracted_pos = [re.search(rf'긍정{i}:\s*(.*)', content).group(1).strip() for i in range(1, 6) if re.search(rf'긍정{i}:\s*(.*)', content)]
        extracted_neg = [re.search(rf'부정{i}:\s*(.*)', content).group(1).strip() for i in range(1, 6) if re.search(rf'부정{i}:\s*(.*)', content)]

        theme_match = re.search(r'강세테마:\s*(.*)', content)
        score_match = re.search(r'감성지수:\s*(.*)', content)
        theme_val = theme_match.group(1).strip() if theme_match else "특이 테마 미포착"
        score_val = score_match.group(1).strip() if score_match else "0점 (중립)"

        pos_html = "<br>".join([f"&nbsp;&nbsp;• {sanitize_text(item)}" for item in extracted_pos if len(item)>2]) or "&nbsp;&nbsp;• 특이 긍정 호재 미포착"
        neg_html = "<br>".join([f"&nbsp;&nbsp;• {sanitize_text(item)}" for item in extracted_neg if len(item)>2]) or "&nbsp;&nbsp;• 특이 부정 리스크 미포착"

        raw_briefing_html = f"""
        🟢 <b>지난 7일 긍정 호재 ({len(extracted_pos)}개):</b><br>{pos_html}<br><br>
        🔴 <b>지난 7일 부정 리스크 ({len(extracted_neg)}개):</b><br>{neg_html}<br><br>
        🚀 <b>시장 주도/강세 테마:</b> <span style="color:#38bdf8; font-weight:bold;">{sanitize_text(theme_val)}</span><br>
        📊 <b>7일 누적 뉴스 감성 지수:</b> <span class="highlight-val">{sanitize_text(score_val)}</span>
        """

        save_ai_cache(cache_key, {"status": sanitize_text(status_val), "briefing_html": raw_briefing_html})
        return sanitize_text(status_val), raw_briefing_html

    except Exception as e:
        return "보통 🟡", f"뉴스 분석 생성 안내: {e}"

# =========================================================
# 🛠️ 기술적 지표 & 파동 마디점 추출 유틸리티 (정량 계산)
# =========================================================
def extract_peaks_and_troughs(df_60, is_krw=True):
    """최근 60일 데이터에서 주요 반등 저점과 저항 고점 좌표를 추출하여 파동 요약문 생성"""
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

def parse_price_from_text(text, key_prefix):
    """AI 응답 텍스트에서 가격 숫자를 절대 에러 없이 안전하게 파싱"""
    if not text:
        return None
    try:
        match = re.search(rf'{key_prefix}\s*:\s*([^\n]+)', text)
        if match:
            raw_str = match.group(1).strip()
            digits = re.findall(r'[\d\.]+', raw_str.replace(',', ''))
            if digits:
                val = float(digits[0])
                if val > 0: return val
    except Exception:
        pass
    return None

# =========================================================
# 🤖 일반 종목 AI 정밀 리포트 (GEMINI 15 RPM 가드 적용)
# =========================================================
def generate_ai_stock_analysis(stock_name, symbol, news_keywords, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, latest_close, ma20_d, ma60_d, ma120_d, supply_type="", currency_symbol="원"):
    cache_key = f"STOCK_{symbol}"
    is_krw = True if currency_symbol in ["원", "KRW"] else False

    default_buy = latest_close * 0.98
    default_stop = latest_close * 0.95
    default_target1 = latest_close * 1.05
    default_target2 = latest_close * 1.10

    fallback_prices = {"buy": default_buy, "stop": default_stop, "target1": default_target1, "target2": default_target2}

    if not llm_mgr.is_available():
        if cache_key in ai_cache_store:
            cached = ai_cache_store[cache_key]
            updated_at = cached.get('updated_at', '일자 미상')
            reason_msg = get_fallback_reason()
            reason_str = f"⚠️ [비실시간 백업] {cached.get('reason', '')}"
            report_str = f"⚠️ [비실시간 백업 리포트 - 생성일: {updated_at} | 📌 사유: {reason_msg}]\n" + cached.get('report', '')
            parsed_prices = cached.get('parsed_prices', fallback_prices)
            return reason_str, report_str, parsed_prices
        else:
            return f"수급/모멘텀 모니터링 포착 종목", "AI 상세 전략 리포트 준비 중입니다.", fallback_prices

    prompt = f"""
너는 20년 경력의 수석 기술적 분석 및 차트 패턴 트레이딩 전문가이다. 
120일 파동 마디점, POC 매물대, 15일 캔들 형태를 종합적으로 판단하여 최적의 매매 가격(눌림목가, 손절가, 1차·2차 익절가)과 전략 리포트를 작성하라.

[종목 기본 & 수급/뉴스 데이터]
- 종목명: {stock_name} ({symbol})
- 수급/테마 특징: {supply_type}
- 최근 시장 주요 뉴스 이슈/호재:
{news_keywords}

[정량적 차트 지표 (파이썬 정밀 계산)]
- 현재가: {fmt_price(latest_close, is_krw)}
- 이동평균선: 20일선({fmt_price(ma20_d, is_krw)}), 60일선({fmt_price(ma60_d, is_krw)}), 120일선({fmt_price(ma120_d, is_krw)}) / 배열: {ma_status}
- 보조지표: RSI({rsi_val}) & RSI Signal({rsi_signal_val}) [{rsi_cross_status}], MACD({macd_status}), 볼린저밴드({bb_status}), 일목구름대({cloud_status})
- 매물대 & 파동: 최근 120일 최대매물대 POC({fmt_price(poc_price, is_krw)}), 120일 최고가({fmt_price(max_120, is_krw)}), 120일 최저가({fmt_price(min_120, is_krw)})
- 최근 60일 파동 마디점 (고점/저점): {peaks_and_troughs_summary}

[단기 캔들 & 거래량 상세 데이터 (최근 15일)]
{raw_data_str_15days}

[가격 산정 가이드라인 - AI 정밀 도출]
1. 눌림목 매수가: 지지가 확인되는 현실적 눌림목 타점 (가격)
2. 타이트 손절가: 현재가 직하단 주요 지지선(20일선/POC/구름대/마디점) 산정 (가격)
3. 1차 익절가: 손절 폭 대비 최소 1.5배 이상 확보되는 저항선 (가격)
4. 2차 익절가: 패턴 상단 및 전고점 저항선 (가격)

[출력 양식 - 규격 엄수]
선정이유: <외인/기관 수급, 뉴스 호재, 주도 테마/섹터 강세, 캔들/패턴 모멘텀을 종합하여 2~3줄 요약>
파싱_눌림목가: <숫자만 입력 ex: 77200>
파싱_손절가: <숫자만 입력 ex: 75500>
파싱_1차익절가: <숫자만 입력 ex: 80200>
파싱_2차익절가: <숫자만 입력 ex: 83800>
상세리포트:
📌 [차트 구조 & 패턴/캔들 종합 진단]
- <이평선/구름대 구조와 함께 현재 포착되는 캔들 형태(거래량 동반 여부) 및 차트 패턴(쌍바닥/역헤드앤숄더/컵앤핸들/엘리엇파동 위치 등)을 2~3줄로 종합 진단>

🟢 [안전 매수 & 리스크 관리 전략 (손익비 타겟 1:1.5 이상)]
- 눌림목 매수 추천가 : <위 파싱_눌림목가와 동일 가격: 000{currency_symbol}> / 최종 하단 지지선 : <가격> / 타이트 손절가 : <위 파싱_손절가와 동일 가격: 000{currency_symbol}>
- <캔들 지지 형태, 매물대(POC), 이평선 및 패턴 지지점 근거 작성. 손절가를 타이트하게 설정해 리스크 폭을 최소화한 이유 서술>.

🚀 [현실적 분할 익절 전략]
- 1차 안전 익절가 : <위 파싱_1차익절가와 동일 가격: 000{currency_symbol}> (손익비 1:1.5 이상 달성 지점 / 물량 50% 익절)
- 2차 추세 익절가 : <위 파싱_2차익절가와 동일 가격: 000{currency_symbol}> (패턴 상단 목표 및 전고점 저항 지점 / 잔량 50% 추세 대응)
- <1차/2차 목표가까지 상승 가능한 지표적/패턴적 근거 및 손익비 우위 관점 서술>.

[언어 제한] 한자(漢字) 및 일본어 절대 금지. 오직 순수 한글, 영문, 숫자만 사용할 것.
"""
    try:
        content = llm_mgr.generate_completion(prompt, temperature=0.3, max_tokens=1000)
        
        reason_val = f"{supply_type} 모멘텀과 기술적 지지선 반등이 강화되는 종목입니다."
        report_val = content

        reason_match = re.search(r'선정이유:\s*(.*)', content)
        report_match = re.search(r'상세리포트:\s*([\s\S]*)', content)

        if reason_match: reason_val = reason_match.group(1).strip()
        if report_match: report_val = report_match.group(1).strip()

        ai_buy = parse_price_from_text(content, "파싱_눌림목가") or default_buy
        ai_stop = parse_price_from_text(content, "파싱_손절가") or default_stop
        ai_target1 = parse_price_from_text(content, "파싱_1차익절가") or default_target1
        ai_target2 = parse_price_from_text(content, "파싱_2차익절가") or default_target2

        parsed_prices = {
            "buy": ai_buy, "stop": ai_stop, "target1": ai_target1, "target2": ai_target2
        }

        save_ai_cache(cache_key, {
            "reason": sanitize_text(reason_val),
            "report": sanitize_text(report_val),
            "parsed_prices": parsed_prices
        })
        return sanitize_text(reason_val), sanitize_text(report_val), parsed_prices

    except Exception as e:
        return "수급 유입 및 기술적 지지 종목", f"상세 전략 리포트 생성 안내: {e}", fallback_prices

# =========================================================
# 🎯 [토스증권 마이 대시보드 전용] AI 심도 3줄 가이드
# =========================================================
def generate_ai_toss_3line_analysis(stock_name, symbol, avg_price, current_price, return_pct, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, is_krw=True):
    cache_key = f"TOSS_MY_{symbol}"

    if not llm_mgr.is_available():
        return "[테스트 모드] AI 연동 미사용 상태입니다."

    prompt = f"""
너는 20년 경력의 수석 포트폴리오 트레이딩 전문가이다. 
사용자의 [내 보유 평단가, 현재 수익률({return_pct:+.2f}%)]과 [차트 캔들/거래량/패턴 및 보조지표]를 바탕으로, 수익 극대화(Trailing Stop)와 리스크 관리에 최적화된 심도 있는 포지션 가이드를 작성하라.

[보유 종목 & 차트 데이터]
- 종목명: {stock_name} ({symbol})
- 내 보유 평단가: {fmt_price(avg_price, is_krw, show_decimal=is_krw)} (현재 수익률: {return_pct:+.2f}%)
- 현재가: {fmt_price(current_price, is_krw)}
- 정량 보조지표: RSI({rsi_val}) & RSI Signal({rsi_signal_val}) [{rsi_cross_status}], MACD({macd_status}), 이평선 배열({ma_status})
- 차트 구조: 볼린저 밴드({bb_status}), 일목 구름대({cloud_status}), 매물대 POC({fmt_price(poc_price, is_krw)})
- 매물대 & 파동: 120일 최고가({fmt_price(max_120, is_krw)}), 120일 최저가({fmt_price(min_120, is_krw)}), 최근 파동 마디점({peaks_and_troughs_summary})

[단기 캔들 & 거래량 상세 데이터 (최근 15일)]
{raw_data_str_15days}

[판단 원칙 - 깊이 있는 패턴 분석 & 트레일링 스탑 준수]
1. 단순 추세 언급을 넘어 최근 15일 캔들(거래량 동반 여부)과 차트 패턴(쌍바닥, 컵앤핸들, 헤드앤숄더, 엘리엇파동 등)의 세부 흐름을 구체적으로 진단할 것.
2. **수익 구간 (+3% 이상) & 상승 추세/패턴 유효 시**: 
   - [결론: 관망 및 손절선 상향 🟢]을 내리고, 전고점/패턴 상단 목표치를 반영해 단기 목표가를 동적으로 상향 조정할 것.
   - 이미 얻은 수익을 100% 보호하도록 **스탑로스(손절가)를 내 평단가보다 높은 현재가 직하단 주요 지지선(20일선/POC/일목기준선)으로 대폭 올릴 것**.
3. **수익 구간이지만 패턴 상단 저항 도달 / 음봉 거래량 터진 과열 시 (RSI-Signal 데드크로스 등)**:
   - [결론: 일부 매도 🔴]를 내리고, 저항대 가격과 함께 30~50% 선제 익절 가이드를 작성할 것.
4. **손실 구간 (-3% 이하) 시**:
   - 지지선 이탈 위험 시 [결론: 손절 및 비중축소 🔴]를 명시할 것.

[출력 양식 - 심도 있는 3줄 가이드]
결론: [관망 및 손절선 상향 🟢 / 일부 매도 🔴 / 관망 🟡 / 손절 및 비중축소 🔴] 중 하나 명시

• [추세/패턴] <평단가 대비 수익률, 최근 15일 캔들/거래량 수급 상태 및 포착된 차트 패턴(쌍바닥/컵앤핸들/N자반등 등)의 구체적 위치 진단>
• [목표가 대응] <상단 전고점/매물대 저항선 근거로 동적 상향 목표가(또는 과열 시 일부 익절가) 수치 제시>
• [이익 보존] <내 평단가보다 높게 설정할 Trailing Stop 손절가 수치와 그 지지선 근거(20일선/POC매물대 등) 제시>

[언어 제한] 한자(漢字) 및 일본어 절대 금지. 오직 순수 한글, 영문, 숫자만 사용할 것.
"""
    try:
        content = llm_mgr.generate_completion(prompt, temperature=0.3, max_tokens=500)
        save_ai_cache(cache_key, {"report": sanitize_text(content)})
        return sanitize_text(content)
    except Exception as e:
        return "[테스트 모드] AI 연동 미사용 상태입니다."

# =========================================================
# PART 1: 🇰🇷 국장(index.html) 분석
# =========================================================
print("\n" + "="*60)
print("🇰🇷 [PART 1] 한국 증시 수급 & 강세 테마주 동적 스캔 중...")
print("="*60)

kr_7d_news = get_naver_7days_news()
kr_market_status, kr_sentiment_briefing = analyze_7days_news_sentiment("대한민국 주식시장(국장)", kr_7d_news)

url_foreign = "https://m.stock.naver.com/api/json/sise/siseListJson.nhn?bizType=dealForeign&sosok=0"
url_organ = "https://m.stock.naver.com/api/json/sise/siseListJson.nhn?bizType=dealOrgan&sosok=0"
url_quant = "https://m.stock.naver.com/api/json/sise/siseListJson.nhn?bizType=topAmount&sosok=0"

def get_naver_sise_list(url):
    try: return requests.get(url, headers=headers, timeout=10).json().get('result', {}).get('itemList', [])[:15]
    except Exception: return []

foreign_items = get_naver_sise_list(url_foreign)
organ_items = get_naver_sise_list(url_organ)
quant_items = get_naver_sise_list(url_quant)

foreign_names = [item.get('nm') for item in foreign_items if item.get('nm')]
organ_names = [item.get('nm') for item in organ_items if item.get('nm')]
both_names = [name for name in foreign_names if name in organ_names]

all_items_kr = {item.get('nm'): item.get('cd') for item in foreign_items + organ_items + quant_items if item.get('nm') and item.get('cd')}
selected_kr_targets = {}

for name in both_names:
    if len(selected_kr_targets) >= 2: break
    code = all_items_kr.get(name)
    if code: selected_kr_targets[name] = (f"{code}.KS", "🔥 외인·기관 동시 순매수(양매수) 주도주")

for item in quant_items:
    if len(selected_kr_targets) >= 4: break
    name, code = item.get('nm'), item.get('cd')
    if name and code and name not in selected_kr_targets:
        selected_kr_targets[name] = (f"{code}.KS", "🚀 당일 거래대금 상위 & 강세 테마주")

for name in foreign_names:
    if len(selected_kr_targets) >= 5: break
    code = all_items_kr.get(name)
    if code and name not in selected_kr_targets:
        selected_kr_targets[name] = (f"{code}.KS", "⚡ 외국인 순매수 상위 핵심주")

print(f"📊 최종 선정된 동적 국장 종목 리스트: {list(selected_kr_targets.keys())}")

stock_cards_kr_html = ""
for stock_name, (symbol, supply_type) in selected_kr_targets.items():
    try:
        pure_code = symbol.split('.')[0]
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period="1y", interval="1d")
        
        if df_daily is None or df_daily.empty or len(df_daily) < 1:
            symbol = f"{pure_code}.KQ"
            df_daily = yf.Ticker(symbol).history(period="1y", interval="1d")

        if df_daily is None or df_daily.empty or len(df_daily) < 1: continue

        df_daily['MA20'] = df_daily['Close'].rolling(20, min_periods=1).mean()
        df_daily['MA60'] = df_daily['Close'].rolling(60, min_periods=1).mean()
        df_daily['MA120'] = df_daily['Close'].rolling(120, min_periods=1).mean()
        
        std20 = df_daily['Close'].rolling(20, min_periods=1).std().fillna(0)
        df_daily['BB_Upper'] = df_daily['MA20'] + (std20 * 2)
        df_daily['BB_Lower'] = df_daily['MA20'] - (std20 * 2)
        
        high9 = df_daily['High'].rolling(9, min_periods=1).max()
        low9 = df_daily['Low'].rolling(9, min_periods=1).min()
        df_daily['Tenkan'] = (high9 + low9) / 2
        high26 = df_daily['High'].rolling(26, min_periods=1).max()
        low26 = df_daily['Low'].rolling(26, min_periods=1).min()
        df_daily['Kijun'] = (high26 + low26) / 2
        high52 = df_daily['High'].rolling(52, min_periods=1).max()
        low52 = df_daily['Low'].rolling(52, min_periods=1).min()
        senkou_b = (high52 + low52) / 2
        senkou_a = (df_daily['Tenkan'] + df_daily['Kijun']) / 2
        
        df_daily['Senkou_A'] = senkou_a.shift(26).fillna(df_daily['Close'])
        df_daily['Senkou_B'] = senkou_b.shift(26).fillna(df_daily['Close'])

        df_recent120 = df_daily.tail(120).copy()
        try:
            num_bins = min(15, len(df_recent120))
            price_bins = pd.cut(df_recent120['Close'], bins=num_bins)
            vol_by_price = df_recent120.groupby(price_bins, observed=False)['Volume'].sum()
            poc_bin = vol_by_price.idxmax() if not vol_by_price.empty else None
            poc_price = int(poc_bin.mid) if poc_bin is not None and pd.notnull(poc_bin) else int(df_recent120['Close'].mean())
        except Exception:
            poc_price = int(df_recent120['Close'].mean())

        max_120 = int(df_recent120['High'].max())
        min_120 = int(df_recent120['Low'].min())
        peaks_and_troughs_summary = extract_peaks_and_troughs(df_daily.tail(60), is_krw=True)

        df_daily = df_daily.ffill().bfill()

        delta = df_daily['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        df_daily['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        df_daily['RSI_Signal'] = df_daily['RSI'].rolling(9, min_periods=1).mean()
        
        rsi_val = round(float(df_daily['RSI'].iloc[-1]), 2)
        rsi_signal_val = round(float(df_daily['RSI_Signal'].iloc[-1]), 2)

        if rsi_val > rsi_signal_val:
            rsi_cross_status = "RSI 상향 돌파 및 상승 모멘텀 유지 📈"
        elif rsi_val < rsi_signal_val and rsi_val >= 60:
            rsi_cross_status = "RSI-Signal 데드크로스 발생 (단기 과열 꺾임 경고) 🔴"
        else:
            rsi_cross_status = "RSI 하향 이탈 및 모멘텀 조정 📉"

        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_val = float(df_daily['MACD'].iloc[-1])
        signal_val = float(df_daily['Signal'].iloc[-1])
        
        latest_close = int(df_daily['Close'].iloc[-1])
        ma20_d = int(df_daily['MA20'].iloc[-1])
        ma60_d = int(df_daily['MA60'].iloc[-1])
        ma120_d = int(df_daily['MA120'].iloc[-1])
        bb_up = int(df_daily['BB_Upper'].iloc[-1])
        bb_low = int(df_daily['BB_Lower'].iloc[-1])
        cloud_a = int(df_daily['Senkou_A'].iloc[-1])
        cloud_b = int(df_daily['Senkou_B'].iloc[-1])
        cloud_top = max(cloud_a, cloud_b)

        short_trend = "단기 상승 추세 📈" if latest_close >= ma20_d else "단기 하락 추세 📉"
        mid_trend = "중기 상승 추세 📈" if latest_close >= ma60_d else "중기 하락 추세 📉"
        bb_status = "상한선 돌파/근접 🚀" if latest_close >= bb_up * 0.99 else ("하한선 근접/지지 🟢" if latest_close <= bb_low * 1.01 else "밴드 내 안정 ⚖️")
        cloud_status = "구름대 위 상승 국면 🟢" if latest_close > cloud_top else "구름대 내부/하단 돌파 시도 🟡"

        df_recent15 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15).copy()
        raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:{int(row['Open']):,}원 | High:{int(row['High']):,}원 | Low:{int(row['Low']):,}원 | Close:{int(row['Close']):,}원 | Vol:{int(row['Volume']):,}" for idx, row in df_recent15.iterrows()]
        raw_data_str_15days = "\n".join(raw_lines)

        rsi_status = f"과매수 ({rsi_val}) ⚠️" if rsi_val >= 70 else (f"과매도 ({rsi_val}) 🟢" if rsi_val <= 30 else f"중립 ({rsi_val}) ⚖️")
        macd_status = "골든크로스 📈" if macd_val > signal_val else "데드크로스 📉"
        ma_status = f"정배열 (20>60>120) 🟢" if (ma20_d > ma60_d and ma60_d > ma120_d) else "역배열/혼조세 🔴"

        tradingview_url = f"https://www.tradingview.com/symbols/KRX-{pure_code}/"

        print(f"  ⚡ [국장] {stock_name} AI 리포트 처리 중...")
        pick_reason, ai_comment, ai_prices = generate_ai_stock_analysis(
            stock_name, symbol, kr_7d_news, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, latest_close, ma20_d, ma60_d, ma120_d, supply_type, "원"
        )

        stop_loss = int(round(ai_prices['stop']))
        target_price_1 = int(round(ai_prices['target1']))

        df_chart = df_daily.tail(120)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.25, 0.2])
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Upper'], line=dict(color='rgba(147, 51, 234, 0.3)', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Lower'], line=dict(color='rgba(147, 51, 234, 0.3)', width=1), fill='tonexty', fillcolor='rgba(147, 51, 234, 0.05)', name='볼린저밴드'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Senkou_A'], line=dict(color='rgba(34, 197, 94, 0.4)', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Senkou_B'], line=dict(color='rgba(239, 68, 68, 0.4)', width=1), fill='tonexty', fillcolor='rgba(34, 197, 94, 0.08)', name='일목 구름대'), row=1, col=1)
        fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='주가'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='orange', width=1.2), name='20일선'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA60'], line=dict(color='purple', width=1.2), name='60일선'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA120'], line=dict(color='#a855f7', width=1.5, dash='dash'), name='120일선'), row=1, col=1)
        
        fig.add_hline(y=poc_price, line_dash="dot", line_color="#facc15", annotation_text=f"최대매물대: {fmt_price(poc_price, True)}", row=1, col=1)
        fig.add_hline(y=target_price_1, line_dash="dash", line_color="green", annotation_text=f"AI 목표가: {fmt_price(target_price_1, True)}", row=1, col=1)
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"AI 손절가: {fmt_price(stop_loss, True)}", row=1, col=1)
        
        colors = ['#f87171' if c < o else '#4ade80' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='거래량'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#38bdf8', width=1.2), name='RSI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI_Signal'], line=dict(color='#facc15', width=1.0, dash='dot'), name='RSI Signal'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        
        fig.update_layout(height=540, margin=dict(l=10, r=10, t=10, b=40), xaxis_rangeslider_visible=False, template="plotly_dark")
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        stock_cards_kr_html += f"""
        <div class="card">
            <div class="console-report">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="report-header">{stock_name} ({pure_code})</div>
                    <a href="{tradingview_url}" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
                </div>
                <div class="stock-reason-box">💡 <b>선정 이유:</b> {pick_reason}</div>
                <div class="report-divider"></div>
                <div class="report-line">• 종가 기준 현재가 : <span class="highlight-val">{fmt_price(latest_close, True)}</span></div>
                <div class="report-line">• 추세 진단 : {short_trend} / {mid_trend}</div>
                <div class="report-line">• 차트 구조 : {bb_status} / {cloud_status}</div>
                <div class="report-line">• 집중 매물대 (POC) : <span class="highlight-val">{fmt_price(poc_price, True)}</span></div>
                <div class="report-line">• RSI / MACD : {rsi_status} / {macd_status}</div>
                <div class="report-line text-red">🛑 AI 산출 손절가 : {fmt_price(stop_loss, True)} <span style="font-size:12px; color:#94a3b8; font-weight:normal;">(AI가 120일 차트 구조/POC/지지선 종합 분석)</span></div>
                <div class="report-line text-green">🚀 AI 산출 1차 익절가 : {fmt_price(target_price_1, True)} <span style="font-size:12px; color:#94a3b8; font-weight:normal;">(AI가 손익비 1:1.5 및 저항선 반영 도출)</span></div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ AI 상세 리포트 & 입체 매매 전략</div>
                <div class="ai-content" style="white-space: pre-line;">{ai_comment}</div>
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e: print(f"🚨 {stock_name} 생성 오류: {e}")

# =========================================================
# PART 2: 🇺🇸 미장(us_index.html) 분석
# =========================================================
print("\n" + "="*60)
print("🇺🇸 [PART 2] 미국 증시 스캔 & AI 분석 중...")
print("="*60)

us_7d_news = get_yahoo_7days_news()
us_market_status, us_sentiment_briefing = analyze_7days_news_sentiment("미국 주식시장(미장)", us_7d_news)

def get_us_active_stocks():
    if TEST_MODE:
        return ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'AMZN', 'GOOGL', 'META', 'AVGO', 'PLTR']
    urls = ["https://finance.yahoo.com/markets/stocks/most-active/", "https://finance.yahoo.com/markets/stocks/gainers/"]
    scanned = []
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            for a in soup.find_all('a'):
                href = a.get('href', '')
                if '/quote/' in href:
                    sym = href.split('/quote/')[1].split('?')[0].split('/')[0].upper()
                    if sym.isalpha() and len(sym) <= 5 and sym not in scanned: scanned.append(sym)
        except Exception: pass
    backup_pool = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'AMZN', 'GOOGL', 'META', 'AVGO', 'PLTR']
    for b in backup_pool:
        if b not in scanned: scanned.append(b)
    return scanned

raw_us_symbols = get_us_active_stocks()
selected_us_targets = {}
for sym in raw_us_symbols:
    if len(selected_us_targets) >= 10: break
    try:
        tk = yf.Ticker(sym)
        info = tk.info
        if info.get('marketCap', 0) >= 10_000_000_000:
            selected_us_targets[info.get('shortName', sym)] = (sym, "🔥 Wall Street 거래대금 상위 및 빅테크/AI 핵심주")
    except Exception: continue

stock_cards_us_html = ""
for stock_name, (symbol, supply_type) in selected_us_targets.items():
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period="1y", interval="1d")
        if df_daily is None or len(df_daily) < 1: continue

        df_daily['MA20'] = df_daily['Close'].rolling(20, min_periods=1).mean()
        df_daily['MA60'] = df_daily['Close'].rolling(60, min_periods=1).mean()
        df_daily['MA120'] = df_daily['Close'].rolling(120, min_periods=1).mean()
        
        std20 = df_daily['Close'].rolling(20, min_periods=1).std().fillna(0)
        df_daily['BB_Upper'] = df_daily['MA20'] + (std20 * 2)
        df_daily['BB_Lower'] = df_daily['MA20'] - (std20 * 2)
        
        high9 = df_daily['High'].rolling(9, min_periods=1).max()
        low9 = df_daily['Low'].rolling(9, min_periods=1).min()
        df_daily['Tenkan'] = (high9 + low9) / 2
        high26 = df_daily['High'].rolling(26, min_periods=1).max()
        low26 = df_daily['Low'].rolling(26, min_periods=1).min()
        df_daily['Kijun'] = (high26 + low26) / 2
        high52 = df_daily['High'].rolling(52, min_periods=1).max()
        low52 = df_daily['Low'].rolling(52, min_periods=1).min()
        senkou_b = (high52 + low52) / 2
        senkou_a = (df_daily['Tenkan'] + df_daily['Kijun']) / 2
        
        df_daily['Senkou_A'] = senkou_a.shift(26).fillna(df_daily['Close'])
        df_daily['Senkou_B'] = senkou_b.shift(26).fillna(df_daily['Close'])

        df_recent120 = df_daily.tail(120).copy()
        try:
            num_bins = min(15, len(df_recent120))
            price_bins = pd.cut(df_recent120['Close'], bins=num_bins)
            vol_by_price = df_recent120.groupby(price_bins, observed=False)['Volume'].sum()
            poc_bin = vol_by_price.idxmax() if not vol_by_price.empty else None
            poc_price = round(float(poc_bin.mid), 2) if poc_bin is not None and pd.notnull(poc_bin) else round(float(df_recent120['Close'].mean()), 2)
        except Exception:
            poc_price = round(float(df_recent120['Close'].mean()), 2)

        max_120 = round(float(df_recent120['High'].max()), 2)
        min_120 = round(float(df_recent120['Low'].min()), 2)
        peaks_and_troughs_summary = extract_peaks_and_troughs(df_daily.tail(60), is_krw=False)

        df_daily = df_daily.ffill().bfill()

        delta = df_daily['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        df_daily['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        df_daily['RSI_Signal'] = df_daily['RSI'].rolling(9, min_periods=1).mean()
        
        rsi_val = round(float(df_daily['RSI'].iloc[-1]), 2)
        rsi_signal_val = round(float(df_daily['RSI_Signal'].iloc[-1]), 2)

        if rsi_val > rsi_signal_val:
            rsi_cross_status = "RSI 상향 돌파 및 상승 모멘텀 유지 📈"
        elif rsi_val < rsi_signal_val and rsi_val >= 60:
            rsi_cross_status = "RSI-Signal 데드크로스 발생 (단기 과열 꺾임 경고) 🔴"
        else:
            rsi_cross_status = "RSI 하향 이탈 및 모멘텀 조정 📉"

        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_val = float(df_daily['MACD'].iloc[-1])
        signal_val = float(df_daily['Signal'].iloc[-1])
        
        latest_close = round(float(df_daily['Close'].iloc[-1]), 2)
        ma20_d = round(float(df_daily['MA20'].iloc[-1]), 2)
        ma60_d = round(float(df_daily['MA60'].iloc[-1]), 2)
        ma120_d = round(float(df_daily['MA120'].iloc[-1]), 2)
        bb_up = round(float(df_daily['BB_Upper'].iloc[-1]), 2)
        bb_low = round(float(df_daily['BB_Lower'].iloc[-1]), 2)
        cloud_a = round(float(df_daily['Senkou_A'].iloc[-1]), 2)
        cloud_b = round(float(df_daily['Senkou_B'].iloc[-1]), 2)
        cloud_top = max(cloud_a, cloud_b)

        short_trend = "단기 상승 추세 📈" if latest_close >= ma20_d else "단기 하락 추세 📉"
        mid_trend = "중기 상승 추세 📈" if latest_close >= ma60_d else "중기 하락 추세 📉"
        bb_status = "상한선 돌파/근접 🚀" if latest_close >= bb_up * 0.99 else ("하한선 근접/지지 🟢" if latest_close <= bb_low * 1.01 else "밴드 내 안정 ⚖️")
        cloud_status = "구름대 위 상승 국면 🟢" if latest_close > cloud_top else "구름대 내부/하단 돌파 시도 🟡"

        df_recent15 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15).copy()
        raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:${row['Open']:.2f} | High:${row['High']:.2f} | Low:${row['Low']:.2f} | Close:${row['Close']:.2f} | Vol:{int(row['Volume']):,}" for idx, row in df_recent15.iterrows()]
        raw_data_str_15days = "\n".join(raw_lines)

        rsi_status = f"과매수 ({rsi_val}) ⚠️" if rsi_val >= 70 else (f"과매도 ({rsi_val}) 🟢" if rsi_val <= 30 else f"중립 ({rsi_val}) ⚖️")
        macd_status = "골든크로스 📈" if macd_val > signal_val else "데드크로스 📉"
        ma_status = f"정배열 (20>60>120) 🟢" if (ma20_d > ma60_d and ma60_d > ma120_d) else "역배열/혼조세 🔴"

        tradingview_url = f"https://www.tradingview.com/symbols/{symbol}/"

        print(f"  ⚡ [미장] {stock_name} AI 리포트 처리 중...")
        pick_reason, ai_comment, ai_prices = generate_ai_stock_analysis(
            stock_name, symbol, us_7d_news, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, latest_close, ma20_d, ma60_d, ma120_d, supply_type, "$"
        )

        stop_loss = round(ai_prices['stop'], 2)
        target_price_1 = round(ai_prices['target1'], 2)

        df_chart = df_daily.tail(120)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.25, 0.2])
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Upper'], line=dict(color='rgba(147, 51, 234, 0.3)', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Lower'], line=dict(color='rgba(147, 51, 234, 0.3)', width=1), fill='tonexty', fillcolor='rgba(147, 51, 234, 0.05)', name='볼린저밴드'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Senkou_A'], line=dict(color='rgba(34, 197, 94, 0.4)', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Senkou_B'], line=dict(color='rgba(239, 68, 68, 0.4)', width=1), fill='tonexty', fillcolor='rgba(34, 197, 94, 0.08)', name='일목 구름대'), row=1, col=1)
        fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='주가'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='orange', width=1.2), name='20일선'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA60'], line=dict(color='purple', width=1.2), name='60일선'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA120'], line=dict(color='#a855f7', width=1.5, dash='dash'), name='120일선'), row=1, col=1)
        
        fig.add_hline(y=poc_price, line_dash="dot", line_color="#facc15", annotation_text=f"최대매물대: {fmt_price(poc_price, False)}", row=1, col=1)
        fig.add_hline(y=target_price_1, line_dash="dash", line_color="green", annotation_text=f"AI 목표가: {fmt_price(target_price_1, False)}", row=1, col=1)
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"AI 손절가: {fmt_price(stop_loss, False)}", row=1, col=1)
        
        colors = ['#f87171' if c < o else '#4ade80' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='거래량'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#38bdf8', width=1.2), name='RSI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI_Signal'], line=dict(color='#facc15', width=1.0, dash='dot'), name='RSI Signal'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        
        fig.update_layout(height=540, margin=dict(l=10, r=10, t=10, b=40), xaxis_rangeslider_visible=False, template="plotly_dark")
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        stock_cards_us_html += f"""
        <div class="card">
            <div class="console-report">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="report-header">{stock_name} ({symbol})</div>
                    <a href="{tradingview_url}" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
                </div>
                <div class="stock-reason-box">💡 <b>선정 이유:</b> {pick_reason}</div>
                <div class="report-divider"></div>
                <div class="report-line">• 종가 기준 현재가 : <span class="highlight-val">{fmt_price(latest_close, False)}</span></div>
                <div class="report-line">• 추세 진단 : {short_trend} / {mid_trend}</div>
                <div class="report-line">• 차트 구조 : {bb_status} / {cloud_status}</div>
                <div class="report-line">• 집중 매물대 (POC) : <span class="highlight-val">{fmt_price(poc_price, False)}</span></div>
                <div class="report-line">• RSI / MACD : {rsi_status} / {macd_status}</div>
                <div class="report-line text-red">🛑 AI 산출 손절가 : {fmt_price(stop_loss, False)} <span style="font-size:12px; color:#94a3b8; font-weight:normal;">(AI가 120일 차트 구조/POC/지지선 종합 분석)</span></div>
                <div class="report-line text-green">🚀 AI 산출 1차 익절가 : {fmt_price(target_price_1, False)} <span style="font-size:12px; color:#94a3b8; font-weight:normal;">(AI가 손익비 1:1.5 및 저항선 반영 도출)</span></div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ AI 상세 리포트 & 입체 매매 전략</div>
                <div class="ai-content" style="white-space: pre-line;">{ai_comment}</div>
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e: print(f"🚨 {stock_name} 생성 오류: {e}")

# =========================================================
# PART 3: 🎯 마이 대시보드(index3.html) 분석 (예외 정밀 보완판)
# =========================================================
print("\n" + "="*60)
print("🎯 [PART 3] 토스 실계좌 잔고 수집 및 종목별 맞춤 UI 리포트 생성 중...")
print("="*60)

def get_toss_holdings():
    if not TOSS_CLIENT_ID or not TOSS_CLIENT_SECRET:
        print("⚠️ Toss API Key 미설정 ➔ 샘플 데이터 사용")
        return get_mock_holdings()

    try:
        token_url = "https://openapi.tossinvest.com/oauth2/token"
        token_data = {"grant_type": "client_credentials", "client_id": TOSS_CLIENT_ID, "client_secret": TOSS_CLIENT_SECRET}
        token_res = requests.post(token_url, data=token_data, headers={"Content-Type": "application/x-www-form-urlencoded"}, proxies=proxies, timeout=15)
        
        if token_res.status_code == 200:
            access_token = token_res.json().get("access_token")
            base_headers = {"Authorization": f"Bearer {access_token}", "x-api-key": TOSS_CLIENT_ID, "Content-Type": "application/json"}
            
            acc_res = requests.get("https://openapi.tossinvest.com/api/v1/accounts", headers=base_headers, proxies=proxies, timeout=15)
            account_seq = 1
            if acc_res.status_code == 200:
                acc_list = acc_res.json().get("result", [])
                if isinstance(acc_list, list) and len(acc_list) > 0:
                    account_seq = acc_list[0].get("accountSeq", 1)

            holdings_headers = base_headers.copy()
            holdings_headers["X-Tossinvest-Account"] = str(account_seq)
            
            res = requests.get("https://openapi.tossinvest.com/api/v1/holdings", headers=holdings_headers, proxies=proxies, timeout=15)
            if res.status_code == 200:
                result_obj = res.json().get("result", {})
                items = result_obj.get("items", []) if isinstance(result_obj, dict) else []
                
                holdings = []
                for item in items:
                    holdings.append({
                        "ticker": str(item.get("symbol", "")),
                        "name": str(item.get("name", "")),
                        "avg_price": float(item.get("averagePurchasePrice", 0)),
                        "quantity": float(item.get("quantity", 0)),
                        "market": str(item.get("marketCountry", "KR")),
                        "currency": str(item.get("currency", "KRW"))
                    })
                if holdings:
                    print(f"🎉 토스증권 API 연동 성공! 실제 보유 종목 총 {len(holdings)}개 수신 완료")
                    return holdings
    except Exception as e:
        print(f"⚠️ 토스 API 호출 오류: {e}")
    return get_mock_holdings()

def get_mock_holdings():
    return [
        {"ticker": "005930.KS", "name": "삼성전자", "avg_price": 72000, "quantity": 50, "market": "KR", "currency": "KRW"},
        {"ticker": "000660.KS", "name": "SK하이닉스", "avg_price": 175000, "quantity": 20, "market": "KR", "currency": "KRW"},
        {"ticker": "NVDA", "name": "NVIDIA", "avg_price": 115.0, "quantity": 15, "market": "US", "currency": "USD"},
        {"ticker": "PLTR", "name": "Palantir", "avg_price": 24.5, "quantity": 100, "market": "US", "currency": "USD"}
    ]

toss_holdings = get_toss_holdings()
my_stock_cards_html = ""

total_eval_my = 0
total_profit_my = 0

for h in toss_holdings:
    try:
        ticker = h['ticker']
        stock_name = h['name']
        avg_price = h['avg_price']
        market = h['market']
        currency = h['currency']
        quantity = h['quantity']
        
        pure_code = ticker.split('.')[0]
        is_krw = True if (market == 'KR' or currency == 'KRW') else False
        fx = usd_krw_rate if not is_krw else 1.0
        
        yf_ticker = f"{pure_code}.KS" if (is_krw and not ticker.endswith((".KS", ".KQ"))) else pure_code
        
        df_daily = None
        try:
            stock = yf.Ticker(yf_ticker)
            df_daily = stock.history(period="1y", interval="1d")
            if (df_daily is None or df_daily.empty or len(df_daily) < 1) and is_krw:
                yf_ticker = f"{pure_code}.KQ"
                df_daily = yf.Ticker(yf_ticker).history(period="1y", interval="1d")
        except Exception: 
            df_daily = None

        if df_daily is None or df_daily.empty or len(df_daily) == 0:
            current_price = avg_price
            return_pct = 0.0
            eval_amount_krw = avg_price * quantity * fx
            profit_loss_krw = 0.0
            short_trend, mid_trend = "단기 데이터 미수집 ⚖️", "중기 데이터 미수집 ⚖️"
            bb_status, cloud_status = "밴드 미산출 ⚖️", "구름대 미산출 🟡"
            poc_price, max_120, min_120 = avg_price, avg_price, avg_price
            rsi_status, macd_status = "중립 (50) ⚖️", "중립 ⚖️"
            rsi_val, rsi_signal_val, rsi_cross_status = 50.0, 50.0, "모멘텀 미산출"
            peaks_and_troughs_summary = "마디점 미산출"
            raw_data_str_15days = "최근 데이터 미수집"
        else:
            latest_close = float(df_daily['Close'].iloc[-1])
            current_price = latest_close
            return_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
            eval_amount_krw = (current_price * quantity) * fx
            profit_loss_krw = ((current_price - avg_price) * quantity) * fx

            df_daily['MA20'] = df_daily['Close'].rolling(20, min_periods=1).mean()
            df_daily['MA60'] = df_daily['Close'].rolling(60, min_periods=1).mean()
            df_daily['MA120'] = df_daily['Close'].rolling(120, min_periods=1).mean()
            
            std20 = df_daily['Close'].rolling(20, min_periods=1).std().fillna(0)
            df_daily['BB_Upper'] = df_daily['MA20'] + (std20 * 2)
            df_daily['BB_Lower'] = df_daily['MA20'] - (std20 * 2)

            high9 = df_daily['High'].rolling(9, min_periods=1).max()
            low9 = df_daily['Low'].rolling(9, min_periods=1).min()
            df_daily['Tenkan'] = (high9 + low9) / 2
            high26 = df_daily['High'].rolling(26, min_periods=1).max()
            low26 = df_daily['Low'].rolling(26, min_periods=1).min()
            df_daily['Kijun'] = (high26 + low26) / 2
            high52 = df_daily['High'].rolling(52, min_periods=1).max()
            low52 = df_daily['Low'].rolling(52, min_periods=1).min()
            senkou_b = (high52 + low52) / 2
            senkou_a = (df_daily['Tenkan'] + df_daily['Kijun']) / 2
            
            df_daily['Senkou_A'] = senkou_a.shift(26).fillna(df_daily['Close'])
            df_daily['Senkou_B'] = senkou_b.shift(26).fillna(df_daily['Close'])

            df_recent120 = df_daily.tail(120).copy()
            try:
                num_bins = min(15, len(df_recent120))
                price_bins = pd.cut(df_recent120['Close'], bins=num_bins)
                vol_by_price = df_recent120.groupby(price_bins, observed=False)['Volume'].sum()
                poc_bin = vol_by_price.idxmax() if not vol_by_price.empty else None
                poc_price = float(poc_bin.mid) if poc_bin is not None and pd.notnull(poc_bin) else float(df_recent120['Close'].mean())
            except Exception:
                poc_price = float(df_recent120['Close'].mean())

            max_120 = float(df_recent120['High'].max())
            min_120 = float(df_recent120['Low'].min())
            peaks_and_troughs_summary = extract_peaks_and_troughs(df_daily.tail(60), is_krw=is_krw)

            # dropna() 대신 안전 결측치 보정 적용
            df_daily = df_daily.ffill().bfill()

            delta = df_daily['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            df_daily['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            df_daily['RSI_Signal'] = df_daily['RSI'].rolling(9, min_periods=1).mean()

            rsi_val = round(float(df_daily['RSI'].iloc[-1]), 2)
            rsi_signal_val = round(float(df_daily['RSI_Signal'].iloc[-1]), 2)

            if rsi_val > rsi_signal_val:
                rsi_cross_status = "RSI 상향 돌파 및 상승 모멘텀 유지 📈"
            elif rsi_val < rsi_signal_val and rsi_val >= 60:
                rsi_cross_status = "RSI-Signal 데드크로스 발생 (단기 과열 꺾임 경고) 🔴"
            else:
                rsi_cross_status = "RSI 하향 이탈 및 모멘텀 조정 📉"

            exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
            df_daily['MACD'] = exp1 - exp2
            df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
            
            macd_val = float(df_daily['MACD'].iloc[-1])
            signal_val = float(df_daily['Signal'].iloc[-1])

            ma20_d = float(df_daily['MA20'].iloc[-1])
            ma60_d = float(df_daily['MA60'].iloc[-1])
            bb_up = float(df_daily['BB_Upper'].iloc[-1])
            bb_low = float(df_daily['BB_Lower'].iloc[-1])

            short_trend = "단기 상승 추세 📈" if current_price >= ma20_d else "단기 하락 추세 📉"
            mid_trend = "중기 상승 추세 📈" if current_price >= ma60_d else "중기 하락 추세 📉"
            bb_status = "상한선 돌파/근접 🚀" if current_price >= bb_up * 0.99 else ("하한선 근접/지지 🟢" if current_price <= bb_low * 1.01 else "밴드 내 안정 ⚖️")
            cloud_status = "구름대 상태 유효 🟢"

            df_recent15 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15).copy()
            if is_krw:
                raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:{int(row['Open']):,}원 | High:{int(row['High']):,}원 | Low:{int(row['Low']):,}원 | Close:{int(row['Close']):,}원 | Vol:{int(row['Volume']):,}" for idx, row in df_recent15.iterrows()]
            else:
                raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:${row['Open']:.2f} | High:${row['High']:.2f} | Low:${row['Low']:.2f} | Close:${row['Close']:.2f} | Vol:{int(row['Volume']):,}" for idx, row in df_recent15.iterrows()]
            raw_data_str_15days = "\n".join(raw_lines)

            rsi_status = f"과매수 ({fmt_num(rsi_val)}) ⚠️" if rsi_val >= 70 else (f"과매도 ({fmt_num(rsi_val)}) 🟢" if rsi_val <= 30 else f"중립 ({fmt_num(rsi_val)}) ⚖️")
            macd_status = "골든크로스 📈" if macd_val > signal_val else "데드크로스 📉"
            ma_status = f"정배열 지지 🟢" if current_price >= ma20_d else "역배열/혼조세 🔴"

        tv_prefix = f"KRX-{pure_code}" if is_krw else ticker
        tradingview_url = f"https://www.tradingview.com/symbols/{tv_prefix}/"

        print(f"  ⚡ [마이] {stock_name} 심도 3줄 가이드 AI 처리 중...")
        ai_3line_comment = generate_ai_toss_3line_analysis(
            stock_name, ticker, avg_price, current_price, return_pct, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, is_krw
        )

        eval_formatted = f"{int(round(eval_amount_krw)):,}원"
        profit_formatted = f"({profit_loss_krw:+,.0f}원)"
        
        avg_price_formatted = fmt_price(avg_price, is_krw, show_decimal=is_krw)
        current_price_formatted = fmt_price(current_price, is_krw)
        poc_formatted = fmt_price(poc_price, is_krw)

        my_stock_cards_html += f"""
        <div class="card">
            <div class="console-report">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="report-header">{stock_name} ({pure_code}) - {fmt_num(quantity)}주</div>
                    <a href="{tradingview_url}" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
                </div>
                <div style="font-size:18px; font-weight:bold; margin-top:4px; color:#f8fafc;">
                    {eval_formatted} <span class="{'text-green' if profit_loss_krw>=0 else 'text-red'}">{profit_formatted}</span>
                </div>
                <div class="report-divider"></div>
                <div class="report-line">주당 평단가 : <span class="highlight-val">{avg_price_formatted}</span> (<span class="{'text-green' if return_pct>=0 else 'text-red'}">{return_pct:+.2f}%</span>) &nbsp;&nbsp;|&nbsp;&nbsp; 주당 현재가 : <span class="highlight-val">{current_price_formatted}</span></div>
                <div class="report-line">추세 진단 : {short_trend} &nbsp;&nbsp;|&nbsp;&nbsp; {mid_trend}</div>
                <div class="report-line">차트 구조 : {bb_status} &nbsp;&nbsp;|&nbsp;&nbsp; {cloud_status}</div>
                <div class="report-line">집중 매물대 (POC) : <span class="highlight-val">{poc_formatted}</span></div>
                <div class="report-line">RSI / MACD : {rsi_status} / {macd_status}</div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ AI 포트폴리오 심도 3줄 대응 가이드</div>
                <div class="ai-content" style="white-space: pre-line;">{ai_3line_comment}</div>
            </div>
        </div>
        """

        total_eval_my += eval_amount_krw
        total_profit_my += profit_loss_krw

    except Exception as e:
        print(f"⚠️ {h.get('name', '종목')} 처리 중 예외 발생: {e}")

total_cost_my = total_eval_my - total_profit_my
total_return_pct_my = (total_profit_my / total_cost_my * 100) if total_cost_my > 0 else 0

# =========================================================
# PART 4: HTML 템플릿 및 레이아웃
# =========================================================
html_style = """
<style>
    body { font-family: 'Consolas', -apple-system, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
    .container { max-width: 950px; margin: 0 auto; }
    .nav-bar { display: flex; justify-content: center; gap: 12px; margin-bottom: 20px; }
    .nav-btn { padding: 8px 16px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 14px; }
    .btn-active { background: #2563eb; color: #ffffff; }
    .btn-inactive { background: #334155; color: #94a3b8; }
    .header { background: #1e293b; color: #38bdf8; padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center; border: 1px solid #334155; }
    
    .macro-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 20px; }
    .macro-card { 
        background: #182232; 
        border: 1px solid #334155; 
        border-radius: 10px; 
        padding: 16px 14px; 
        text-align: center; 
        text-decoration: none; 
        color: inherit; 
        display: block;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }
    .macro-card:hover { transform: translateY(-3px); border-color: #38bdf8; box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2); }
    .macro-title { font-size: 13px; color: #94a3b8; font-weight: bold; margin-bottom: 8px; }
    .macro-value { font-size: 20px; font-weight: bold; color: #38bdf8; margin-bottom: 6px; }
    .macro-sub { font-size: 11px; color: #4ade80; margin-top: 4px; }
    
    .news-briefing-card { background: #182232; border: 1px solid #38bdf8; border-radius: 12px; padding: 18px; margin-bottom: 25px; line-height: 1.8; font-size: 14px; }
    .news-title { font-size: 16px; font-weight: bold; color: #38bdf8; margin-bottom: 10px; border-bottom: 1px dashed #334155; padding-bottom: 6px; }
    
    .card { background: #1e293b; padding: 20px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #334155; }
    .console-report { background: #090d16; padding: 18px; border-radius: 8px; border: 1px solid #334155; font-size: 15px; line-height: 1.7; }
    .stock-reason-box { background: #1e1b4b; border-left: 4px solid #818cf8; padding: 12px; border-radius: 4px; margin: 12px 0; font-size: 14px; color: #e0e7ff; line-height: 1.6; }
    .report-header { font-size: 17px; font-weight: bold; color: #38bdf8; }
    .report-divider { border-top: 1px dashed #475569; margin: 10px 0; }
    .report-line { margin: 4px 0; }
    .highlight-val { color: #facc15; font-weight: bold; }
    .text-red { color: #f87171; font-weight: bold; }
    .text-green { color: #4ade80; font-weight: bold; }
    .tv-link-btn { background: #2563eb; color: #ffffff; padding: 4px 10px; border-radius: 4px; text-decoration: none; font-size: 12px; font-weight: bold; }
    .ai-opinion-box { background: #062a1c; border: 1px solid #22c55e; border-radius: 8px; padding: 16px; margin-top: 15px; }
    .ai-title { font-size: 14px; font-weight: bold; color: #4ade80; margin-bottom: 10px; }
    .ai-content { font-size: 14px; color: #f1f5f9; line-height: 1.75; }
    .chart-container { margin-top: 20px; border-radius: 8px; overflow: hidden; }
</style>
"""

macro_html_kr = f"""
<div class="macro-grid">
    <a href="{kr_macro['m2_url']}" target="_blank" class="macro-card">
        <div class="macro-title">💵 원화 통화량 (M2) ↗</div>
        <div class="macro-value">{kr_macro['m2']}</div>
        <div class="macro-sub">{kr_macro['m2_date']} 🟢</div>
    </a>
    <a href="{kr_macro['cli_url']}" target="_blank" class="macro-card">
        <div class="macro-title">🌐 한국 경기선행지수 (CLI) ↗</div>
        <div class="macro-value">{kr_macro['cli']}</div>
        <div class="macro-sub">{kr_macro['cli_date']} 🟢</div>
    </a>
    <a href="{kr_macro['vix_url']}" target="_blank" class="macro-card">
        <div class="macro-title">⚡ 한국 VKOSPI ↗</div>
        <div class="macro-value" style="color:#facc15;">{kr_macro['vix']}</div>
        <div class="macro-sub">네이버 금융 원본 연동 🟢</div>
    </a>
</div>
"""

macro_html_us = f"""
<div class="macro-grid">
    <a href="{us_macro['m2_url']}" target="_blank" class="macro-card">
        <div class="macro-title">💵 달러 통화량 (US M2) ↗</div>
        <div class="macro-value">{us_macro['m2']}</div>
        <div class="macro-sub">{us_macro['m2_date']} 🟢</div>
    </a>
    <a href="{us_macro['cli_url']}" target="_blank" class="macro-card">
        <div class="macro-title">🌐 미국 경기선행지수 (CLI) ↗</div>
        <div class="macro-value">{us_macro['cli']}</div>
        <div class="macro-sub">{us_macro['cli_date']} 🟢</div>
    </a>
    <a href="{us_macro['vix_url']}" target="_blank" class="macro-card">
        <div class="macro-title">⚡ 미국 VIX 지수 ↗</div>
        <div class="macro-value" style="color:#facc15;">{us_macro['vix']}</div>
        <div class="macro-sub">TradingView 원본 연동 🟢</div>
    </a>
</div>
"""

full_html_kr = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>🇰🇷 AI 국장 분석 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-active">🇰🇷 국장 대시보드</a><a href="us_index.html" class="nav-btn btn-inactive">🇺🇸 미장 대시보드</a><a href="index3.html" class="nav-btn btn-inactive">🎯 마이 대시보드</a></div><div class="header"><h1>📊 AI 국장 주도주 대시보드 <span style="font-size:18px;">[{kr_market_status}]</span></h1><p style="margin:0; color:#94a3b8; font-size:14px;">업데이트: {now_str}</p></div>{macro_html_kr}<div class="news-briefing-card"><div class="news-title">📰 [최근 7일간 뉴스 AI 종합 분석 브리핑]</div>{kr_sentiment_briefing}</div>{stock_cards_kr_html}</div></body></html>"""

full_html_us = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>🇺🇸 AI 미장 분석 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-inactive">🇰🇷 국장 대시보드</a><a href="us_index.html" class="nav-btn btn-active">🇺🇸 미장 대시보드</a><a href="index3.html" class="nav-btn btn-inactive">🎯 마이 대시보드</a></div><div class="header"><h1>🇺🇸 AI US Stock 주도주 대시보드 <span style="font-size:18px;">[{us_market_status}]</span></h1><p style="margin:0; color:#94a3b8; font-size:14px;">업데이트: {now_str}</p></div>{macro_html_us}<div class="news-briefing-card"><div class="news-title">📰 [최근 7일간 뉴스 AI 종합 분석 브리핑]</div>{us_sentiment_briefing}</div>{stock_cards_us_html}</div></body></html>"""

full_html_my = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>🎯 마이 포트폴리오 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-inactive">🇰🇷 국장 대시보드</a><a href="us_index.html" class="nav-btn btn-inactive">🇺🇸 미장 대시보드</a><a href="index3.html" class="nav-btn btn-active">🎯 마이 대시보드</a></div><div class="header"><h1>🎯 마이 포트폴리오 실계좌 대시보드</h1><p style="margin:0; color:#94a3b8; font-size:14px;">업데이트: {now_str}</p></div><div style="background:linear-gradient(135deg, #1e293b, #334155); padding:20px; border-radius:12px; margin-bottom:25px; display:flex; justify-content:space-around; text-align:center; border:1px solid #334155;"><div><div style="font-size:0.8rem; color:#94a3b8;">총 평가 금액 (원화 환산)</div><div style="font-size:1.5rem; font-weight:bold; margin-top:5px; color:#f8fafc;">{fmt_price(total_eval_my, True)}</div><div style="font-size:0.8rem; color:#60a5fa; margin-top:4px;">적용 환율: {usd_krw_rate:,.1f} 원/USD</div></div><div><div style="font-size:0.8rem; color:#94a3b8;">총 평가 손익</div><div style="font-size:1.5rem; font-weight:bold; margin-top:5px; color:{'#f87171' if total_profit_my>=0 else '#60a5fa'};">{total_profit_my:+,.0f}원</div></div><div><div style="font-size:0.8rem; color:#94a3b8;">전체 수익률</div><div style="font-size:1.5rem; font-weight:bold; margin-top:5px; color:{'#f87171' if total_return_pct_my>=0 else '#60a5fa'};">{total_return_pct_my:+.2f}%</div></div></div><h2 style="font-size:1.2rem; color:#38bdf8; margin-bottom:15px;">📊 토스증권 연동 보유 종목 정밀 분석 & AI 가이드</h2>{my_stock_cards_html}</div></body></html>"""

# =========================================================
# PART 5: GitHub Pages 및 ai_cache.json 통합 배포
# =========================================================
def upload_to_github_safely(repo, file_path, commit_message, content):
    try:
        file_obj = repo.get_contents(file_path)
        repo.update_file(path=file_path, message=commit_message, content=content, sha=file_obj.sha)
        print(f"✅ {file_path} 커밋 및 업데이트 성공!")
    except UnknownObjectException:
        repo.create_file(path=file_path, message=commit_message, content=content)
        print(f"✅ {file_path} 신규 생성 배포 성공!")
    except Exception as e:
        print(f"🚨 {file_path} 배포 중 예외 발생: {e}")

print("\n🌐 [PART 5] GitHub Pages (index, us_index, index3) 및 AI 캐시 동기화 업로드 중...")
try:
    if not GITHUB_TOKEN:
        raise ValueError("GH_TOKEN이 설정되지 않았습니다.")

    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO_NAME)
    
    upload_to_github_safely(repo, "index.html", f"Deploy KR Report: {now_str}", full_html_kr)
    upload_to_github_safely(repo, "us_index.html", f"Deploy US Report: {now_str}", full_html_us)
    upload_to_github_safely(repo, "index3.html", f"Deploy My Dashboard: {now_str}", full_html_my)
    
    if os.path.exists(CACHE_FILE_NAME):
        with open(CACHE_FILE_NAME, "r", encoding="utf-8") as f:
            cache_json_str = f.read()
        upload_to_github_safely(repo, "ai_cache.json", f"Update AI Cache: {now_str}", cache_json_str)

    print("\n" + "="*65)
    print("🎉 [최종 완료] 3개 대시보드가 정상적으로 업데이트 및 GitHub 배포 완료되었습니다!")
    print(f"🔗 🇰🇷 국장: https://{repo.owner.login}.github.io/{repo.name}/index.html")
    print(f"🔗 🇺🇸 미장: https://{repo.owner.login}.github.io/{repo.name}/us_index.html")
    print(f"🔗 🎯 마이: https://{repo.owner.login}.github.io/{repo.name}/index3.html")
    print("="*65)

except Exception as e:
    print(f"🚨 GitHub 연결 과정 중 오류 발생: {e}")
