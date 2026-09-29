# 1. 필수 라이브러리 자동 설치 및 안전 Import (GitHub Actions / Colab 100% 호환)
import sys
import subprocess

for pkg in [
    "google-genai", "groq", "yfinance", "pandas==2.2.2", 
    "beautifulsoup4", "plotly", "requests", "PyGithub", "pandas_market_calendars"
]:
    try:
        mod_name = pkg.split("==")[0].replace("-", "_")
        __import__(mod_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

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
import calendar
import time
import re
import warnings
import pandas_market_calendars as mcal

warnings.filterwarnings('ignore')

# =========================================================
# ⚙️ [테스트 모드 설정]
# =========================================================
TEST_MODE = False

# =========================================================
# [보안 및 Secrets / 환경변수 자동 로드 - Gemini Key 1, 2 연동]
# =========================================================
try:
    from google.colab import userdata
    GEMINI_API_KEY_1 = userdata.get('GEMINI_API_REPORT') or userdata.get('GEMINI_API_KEY')
    GEMINI_API_KEY_2 = userdata.get('GEMINI_API_REPORT2') or userdata.get('GEMINI_API_KEY2')
    GROQ_API_KEY_1 = userdata.get('GROQ_API_KEY')
    GROQ_API_KEY_2 = userdata.get('GROQ_API_KEY2')
    GITHUB_TOKEN = userdata.get('GH_TOKEN')
    TOSS_CLIENT_ID = userdata.get('TOSS_CLIENT_ID')
    TOSS_CLIENT_SECRET = userdata.get('TOSS_CLIENT_SECRET')
    FIXIE_URL = userdata.get('FIXIE_URL')
except Exception:
    GEMINI_API_KEY_1 = os.environ.get("GEMINI_API_REPORT") or os.environ.get("GEMINI_API_KEY", "")
    GEMINI_API_KEY_2 = os.environ.get("GEMINI_API_REPORT2") or os.environ.get("GEMINI_API_KEY2", "")
    GROQ_API_KEY_1 = os.environ.get("GROQ_API_KEY", "")
    GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY2", "")
    GITHUB_TOKEN = os.environ.get("GH_TOKEN", "")
    TOSS_CLIENT_ID = os.environ.get("TOSS_CLIENT_ID", "")
    TOSS_CLIENT_SECRET = os.environ.get("TOSS_CLIENT_SECRET", "")
    FIXIE_URL = os.environ.get("FIXIE_URL", "")

GITHUB_REPO_NAME = os.environ.get("GITHUB_REPOSITORY", "dhlee090512-arch/report")
CACHE_FILE_NAME = "ai_cache.json"

# =========================================================
# 🌐 [프록시 풀 설정]
# =========================================================
PROXY_POOL = [
    "http://ljdunsqh:ln7u5fsekv2t@142.111.67.146:5611",
    "http://zghmkutu:36itaybf3evk@31.59.20.176:6754"
]
if FIXIE_URL and FIXIE_URL not in PROXY_POOL:
    PROXY_POOL.append(FIXIE_URL)

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

def toss_request_with_proxy_failover(method, url, **kwargs):
    headers_req = kwargs.pop('headers', {})
    data_req = kwargs.pop('data', None)
    timeout_req = kwargs.pop('timeout', 12)

    for i, p_url in enumerate(PROXY_POOL):
        proxies = {"http": p_url, "https": p_url}
        try:
            if method.upper() == "POST":
                res = requests.post(url, headers=headers_req, data=data_req, proxies=proxies, timeout=timeout_req)
            else:
                res = requests.get(url, headers=headers_req, proxies=proxies, timeout=timeout_req)

            if res.status_code == 402:
                print(f"⚠️ 프록시 #{i+1} 대역폭 소진 -> 다음 프록시로 페일오버")
                continue
            return res
        except Exception:
            continue

    if method.upper() == "POST":
        return requests.post(url, headers=headers_req, data=data_req, timeout=timeout_req)
    else:
        return requests.get(url, headers=headers_req, timeout=timeout_req)

# =========================================================
# [유틸리티 함수]
# =========================================================
def fmt_price(val, is_krw=True, show_decimal=False):
    if val is None or pd.isna(val):
        return "0원" if is_krw else "$0.00"
    try:
        f_val = float(val)
    except Exception:
        return "0원" if is_krw else "$0.00"

    if is_krw:
        return f"{f_val:,.2f}원" if show_decimal else f"{int(round(f_val)):,}원"
    else:
        return f"${f_val:,.2f}" if (show_decimal or not f_val.is_integer()) else f"${int(round(f_val)):,}"

def fmt_num(val):
    if val is None or pd.isna(val):
        return "0"
    try:
        f_val = float(val)
        return f"{int(f_val):,}" if f_val.is_integer() else f"{f_val:,.2f}".rstrip('0').rstrip('.')
    except Exception:
        return "0"

def adjust_to_tick_size(price, is_krw=True):
    if price is None or price <= 0:
        return price
    if not is_krw:
        return round(price, 2) if price >= 1.0 else round(price, 4)
    p = float(price)
    if p < 2000: tick = 1
    elif p < 5000: tick = 5
    elif p < 20000: tick = 10
    elif p < 50000: tick = 50
    elif p < 200000: tick = 100
    elif p < 500000: tick = 500
    else: tick = 1000
    return int(round(p / tick) * tick)

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

# =========================================================
# 🏛️ [3단계 AI 다중화 매니저: Gemini 1 -> Gemini 2 -> Groq Pool]
# =========================================================
class MultiLLMManager:
    def __init__(self, gemini_keys, groq_keys):
        self.gemini_keys = [k.strip() for k in gemini_keys if k and k.strip()]
        self.gemini_clients = []
        for i, k in enumerate(self.gemini_keys):
            try:
                c = genai.Client(api_key=k)
                self.gemini_clients.append(c)
                print(f"✅ [Gemini Client #{i+1}] 초기화 성공")
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
                print(f"✅ [3순위 Groq Key #{self.current_groq_index + 1}] 초기화 성공")
            except Exception as e:
                print(f"⚠️ Groq Key #{self.current_groq_index + 1} 초기화 실패: {e}")
                self.groq_client = None

    def switch_to_next_groq(self):
        self.current_groq_index += 1
        if self.current_groq_index < len(self.groq_keys):
            print(f"🔄 Groq Key #{self.current_groq_index + 1}로 자동 전환")
            self._init_groq_client()
            return True
        else:
            print("🚨 모든 Groq Key 소진")
            self.groq_client = None
            return False

    def is_available(self):
        return (len(self.gemini_clients) > 0 or self.groq_client is not None) and not TEST_MODE

    def generate_completion(self, prompt, temperature=0.3, max_tokens=1800):
        if TEST_MODE:
            raise RuntimeError("TEST_MODE 활성화 상태")

        # 1 & 2순위 Gemini 시도
        for idx, client in enumerate(self.gemini_clients):
            for attempt in range(2):
                try:
                    elapsed = time.time() - self.last_gemini_call_time
                    if elapsed < 4.2:
                        time.sleep(4.2 - elapsed)

                    print(f"⚡ [Gemini Key #{idx+1}] 요청 중... (시도 {attempt+1}/2)")
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
                        print(f"⏳ Gemini Key #{idx+1} 503 감지 -> 3초 대기 후 재시도")
                        time.sleep(3)
                        continue
                    print(f"⚠️ Gemini Key #{idx+1} 실패 ({e}) -> 다음 순위 전환")
                    break

        # 3순위 Groq 시도
        groq_model_candidates = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama3-70b-8192", "llama-3.1-8b-instant"]
        while self.groq_client:
            for g_model in groq_model_candidates:
                try:
                    print(f"⚡ [Groq Key #{self.current_groq_index + 1}] ({g_model}) 요청 중...")
                    res = self.groq_client.chat.completions.create(
                        model=g_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return res.choices[0].message.content.strip()
                except RateLimitError:
                    print(f"🔄 Groq Key #{self.current_groq_index + 1} 한도 초과")
                    break
                except Exception as e:
                    if "404" in str(e) or "model_not_found" in str(e):
                        continue
                    break

            if not self.switch_to_next_groq():
                break

        raise RuntimeError("모든 AI 모델(Gemini 1/2 및 Groq Pool) 호출에 실패했습니다.")

llm_mgr = MultiLLMManager(
    gemini_keys=[GEMINI_API_KEY_1, GEMINI_API_KEY_2], 
    groq_keys=[GROQ_API_KEY_1, GROQ_API_KEY_2]
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Chrome/120.0.0.0)',
    'Referer': 'https://finance.naver.com/'
}

kst_timezone = datetime.timezone(datetime.timedelta(hours=9))
now_dt = datetime.datetime.now(kst_timezone)
now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")
today_date = now_dt.date()

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

def save_entire_cache(full_cache_dict):
    try:
        with open(CACHE_FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(full_cache_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 전체 캐시 저장 실패: {e}")

ai_cache_store = load_ai_cache()

def is_cache_valid(cache_key, max_hours):
    if cache_key not in ai_cache_store:
        return False
    cached_data = ai_cache_store[cache_key]
    updated_at_str = cached_data.get('updated_at', '')
    if not updated_at_str:
        return False
    try:
        cached_dt = datetime.datetime.strptime(updated_at_str, "%Y-%m-%d %H:%M KST").replace(tzinfo=kst_timezone)
        return ((now_dt - cached_dt).total_seconds() / 3600.0) < max_hours
    except Exception:
        return False

def should_refresh_daily_pivot(market_type):
    cache_key = f"MARKET_{market_type}"
    if cache_key not in ai_cache_store:
        return True
    cached_data = ai_cache_store[cache_key]
    updated_at_str = cached_data.get('updated_at', '')
    if not updated_at_str:
        return True
    try:
        cached_dt = datetime.datetime.strptime(updated_at_str, "%Y-%m-%d %H:%M KST").replace(tzinfo=kst_timezone)
        target_h, target_m = (8, 30) if ("국장" in market_type or "한국" in market_type) else (22, 0)
        today_pivot = now_dt.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
        if now_dt >= today_pivot:
            return cached_dt < today_pivot
        else:
            return cached_dt < (today_pivot - datetime.timedelta(days=1))
    except Exception:
        return True

# =========================================================
# 🚨 변동성 및 거시 지표 수집
# =========================================================
def get_index_change_rate(ticker_symbol):
    try:
        df = yf.Ticker(ticker_symbol).history(period="2d")
        if df is not None and len(df) >= 2:
            return ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100.0
        elif df is not None and len(df) == 1:
            return ((df['Close'].iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1]) * 100.0
    except Exception:
        pass
    return 0.0

def check_market_volatility_trigger(market_type="KR"):
    if market_type == "KR":
        avg_chg = (get_index_change_rate("^KS11") + get_index_change_rate("^KQ11")) / 2.0
    else:
        avg_chg = (get_index_change_rate("^IXIC") + get_index_change_rate("^DJI")) / 2.0

    state_key = f"VOLATILITY_STATE_{market_type}"
    today_str = today_date.strftime("%Y-%m-%d")
    state = ai_cache_store.get(state_key, {"date": today_str, "status": "NORMAL", "crash_count": 0, "recovery_count": 0})
    if state.get("date") != today_str:
        state = {"date": today_str, "status": "NORMAL", "crash_count": 0, "recovery_count": 0}

    is_emergency_refresh = False
    banner_msg = None
    defense_mode = False

    if avg_chg <= -2.5:
        defense_mode = True
        if state["status"] == "NORMAL" and state["crash_count"] == 0:
            state["status"] = "CRASH_HANDLED"
            state["crash_count"] = 1
            is_emergency_refresh = True
            banner_msg = f"🚨 <b>[시장 급락 경보 ({avg_chg:+.2f}%)]</b> 방어 지지선 및 손절선 풀 업데이트 완료"
        elif state["status"] == "RECOVERY_HANDLED":
            state["status"] = "HIGH_VOLATILITY_LOCKED"
            banner_msg = f"🚨 <b>[초고변동성 롤러코스터 경보 ({avg_chg:+.2f}%)]</b> 신규 진입 주의 및 관망 권장"
    elif avg_chg >= -0.5 and state["status"] == "CRASH_HANDLED" and state["recovery_count"] == 0:
        state["status"] = "RECOVERY_HANDLED"
        state["recovery_count"] = 1
        is_emergency_refresh = True
        banner_msg = f"🟢 <b>[시장 급반등 확인 ({avg_chg:+.2f}%)]</b> 상방 목표가 복구 풀 업데이트 완료"

    ai_cache_store[state_key] = state
    save_ai_cache(state_key, state)
    return is_emergency_refresh, banner_msg, defense_mode, avg_chg

def get_market_open_status(market="KR"):
    if today_date.weekday() in [5, 6]:
        return False, "주말 휴장"
    cal_name = 'XKRX' if market == "KR" else 'NYSE'
    try:
        schedule = mcal.get_calendar(cal_name).schedule(start_date=today_date, end_date=today_date)
        return (not schedule.empty), ("정상 개장일" if not schedule.empty else "증시 공식 휴장일")
    except Exception:
        return True, "개장일"

def get_witching_day_alert(market="KR"):
    alerts = []
    for m_offset in range(2):
        month = today_date.month + m_offset
        year = today_date.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        cal = calendar.monthcalendar(year, month)
        if market == "KR":
            thursdays = [w[3] for w in cal if w[3] != 0]
            if len(thursdays) >= 2:
                target_date = datetime.date(year, month, thursdays[1])
                d_day = (target_date - today_date).days
                if 0 <= d_day <= 7:
                    event = "네 마녀의 날 (선물·옵션 동시 만기일) 🧙‍♀️" if month in [3, 6, 9, 12] else "월별 옵션 만기일"
                    alerts.append(f"🚨 <b>[{'오늘(D-Day)' if d_day==0 else f'D-{d_day}'}]</b> {target_date.strftime('%m/%d')} 한국 {event}")
        else:
            fridays = [w[4] for w in cal if w[4] != 0]
            if len(fridays) >= 3:
                target_date = datetime.date(year, month, fridays[2])
                d_day = (target_date - today_date).days
                if 0 <= d_day <= 7:
                    event = "세/네 마녀의 날 🧙‍♀️" if month in [3, 6, 9, 12] else "월별 옵션 만기일"
                    alerts.append(f"🚨 <b>[{'오늘(D-Day)' if d_day==0 else f'D-{d_day}'}]</b> {target_date.strftime('%m/%d')} 미국 {event}")
    return alerts

def get_economic_calendar_events(market="KR"):
    events_list = []
    try:
        res = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", headers=headers, timeout=5)
        if res.status_code == 200:
            target_country = "USD" if market == "US" else "KRW"
            for item in res.json():
                country = item.get("country", "")
                impact = item.get("impact", "")
                date_str = item.get("date", "")[:10]
                if (country == "USD" and impact == "High") or (country == target_country and impact in ["High", "Medium"]):
                    try:
                        d_diff = (datetime.datetime.strptime(date_str, "%Y-%m-%d").date() - today_date).days
                        if 0 <= d_diff <= 5:
                            events_list.append(f"• <b>[{'오늘' if d_diff==0 else f'D-{d_diff}'}]</b> {date_str[5:]} {item.get('time','')} {country} {item.get('title','')} [중요 🔴]")
                    except Exception:
                        pass
    except Exception:
        pass
    return events_list[:4]

def parse_te_summary_val(url):
    latest_val, date_str = None, ""
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            elem = soup.select_one('#description') or soup.select_one('.panel-body') or soup.select_one('p')
            if elem:
                match = re.search(r'(?:increased|decreased|stood|reached)\s+to\s+([\d,]+\.?\d*).*?in\s+([A-Za-z0-9\s]+)', elem.text.strip(), re.IGNORECASE)
                if match:
                    latest_val = float(match.group(1).replace(',', ''))
                    date_str = f"{match.group(2).strip()} 기준"
    except Exception:
        pass
    return latest_val, date_str

def get_kr_macro_data():
    val_m2, date_m2 = parse_te_summary_val("https://tradingeconomics.com/south-korea/money-supply-m2")
    m2_val = f"{fmt_num(val_m2/1000 if val_m2 and val_m2>1000 else 4210.5)}조 원"
    val_cli, date_cli = parse_te_summary_val("https://tradingeconomics.com/south-korea/leading-economic-index")
    cli_val = f"{fmt_num(val_cli if val_cli else 101.20)} Pts"
    try:
        soup = BeautifulSoup(requests.get("https://finance.naver.com/sise/sise_index.naver?code=VKOSPI", headers=headers, timeout=5).text, 'html.parser')
        vk_val = soup.select_one('#now_value').text.strip()
    except Exception:
        vk_val = "18.50"
    return {
        "m2": m2_val, "m2_date": date_m2 or "최신 기준", "m2_url": "https://tradingeconomics.com/south-korea/money-supply-m2",
        "cli": cli_val, "cli_date": date_cli or "최신 기준", "cli_url": "https://tradingeconomics.com/south-korea/leading-economic-index",
        "vix": f"{vk_val} Pts", "vix_url": "https://finance.naver.com/sise/sise_index.naver?code=VKOSPI"
    }

def get_us_macro_data():
    val_m2, date_m2 = parse_te_summary_val("https://tradingeconomics.com/united-states/money-supply-m2")
    m2_val = f"${fmt_num(val_m2/1000 if val_m2 and val_m2>1000 else 21.40)} Trillion"
    val_cli, date_cli = parse_te_summary_val("https://tradingeconomics.com/united-states/leading-economic-index")
    cli_val = f"{fmt_num(val_cli if val_cli else 102.10)} Pts"
    try:
        vix_tk = yf.Ticker("^VIX").history(period="2d")
        vix_val = f"{float(vix_tk['Close'].iloc[-1]):.2f} Pts"
    except Exception:
        vix_val = "15.20 Pts"
    return {
        "m2": m2_val, "m2_date": date_m2 or "최신 기준", "m2_url": "https://tradingeconomics.com/united-states/money-supply-m2",
        "cli": cli_val, "cli_date": date_cli or "최신 기준", "cli_url": "https://tradingeconomics.com/united-states/leading-economic-index",
        "vix": vix_val, "vix_url": "https://www.tradingview.com/symbols/CBOE-VIX/"
    }

def get_usd_krw_rate():
    try:
        return float(yf.Ticker("KRW=X").history(period="1d")['Close'].iloc[-1])
    except Exception:
        return 1350.0

kr_macro = get_kr_macro_data()
us_macro = get_us_macro_data()
usd_krw_rate = get_usd_krw_rate()

# =========================================================
# 📰 뉴스 수집 및 7일 감성 분석
# =========================================================
def get_naver_7days_news():
    titles = []
    try:
        for i in range(5):
            t_date = (now_dt - datetime.timedelta(days=i)).strftime("%Y%m%d")
            res = requests.get(f"https://finance.naver.com/news/mainnews.naver?date={t_date}", headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            titles.extend([a.text.strip() for a in soup.select('.articleSubject a') if len(a.text.strip()) > 5][:6])
        return "\n".join(list(dict.fromkeys(titles))[:30])
    except Exception:
        return "반도체 및 AI 밸류체인 수급 기대감 유입 | 수출 기업 실적 모멘텀 지속"

def get_yahoo_7days_news():
    titles = []
    try:
        res = requests.get("https://news.google.com/rss/search?q=US+stock+market+when:7d&hl=en-US&gl=US&ceid=US:en", headers=headers, timeout=6)
        root = ET.fromstring(res.content)
        for item in root.findall('.//item/title')[:25]:
            t = item.text.strip()
            if ' - ' in t: t = t.rsplit(' - ', 1)[0]
            titles.append(t)
        return "\n".join(list(dict.fromkeys(titles))[:25])
    except Exception:
        return "Fed Policy Path In Focus | Tech Earnings Growth Momentum"

def sanitize_text(text):
    if not text: return ""
    return re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]', '', str(text)).replace("파싱_", "").strip()

def analyze_7days_news_sentiment(market_type, news_text, force_refresh=False):
    cache_key = f"MARKET_{market_type}"
    if not force_refresh and not should_refresh_daily_pivot(market_type):
        cached = ai_cache_store[cache_key]
        return cached['status'], cached['briefing_html'], cached.get('updated_at', now_str)

    if not llm_mgr.is_available():
        if cache_key in ai_cache_store:
            c = ai_cache_store[cache_key]
            return c['status'], c['briefing_html'], c.get('updated_at', now_str)
        return "보통 🟡", "분석 준비 중", now_str

    prompt = f"""
너는 마켓 분석가이다. {market_type} 최근 뉴스들을 분석하라.
[뉴스]
{news_text}

[출력 양식]
상태: <긍정 OR 보통 OR 부정>
긍정1: <내용>
긍정2: <내용>
긍정3: <내용>
부정1: <내용>
부정2: <내용>
부정3: <내용>
강세테마: <테마명>
감성지수: <+00점 또는 -00점>
[언어 제한] 한자/일본어 절대 금지.
"""
    try:
        content = llm_mgr.generate_completion(prompt, temperature=0.3, max_tokens=600)
        status_m = re.search(r'상태:\s*(.*)', content)
        st = "긍정 🟢" if status_m and "긍정" in status_m.group(1) else ("부정 🔴" if status_m and "부정" in status_m.group(1) else "보통 🟡")
        
        pos = [re.search(rf'긍정{i}:\s*(.*)', content).group(1).strip() for i in range(1, 4) if re.search(rf'긍정{i}:\s*(.*)', content)]
        neg = [re.search(rf'부정{i}:\s*(.*)', content).group(1).strip() for i in range(1, 4) if re.search(rf'부정{i}:\s*(.*)', content)]
        th = re.search(r'강세테마:\s*(.*)', content)
        sc = re.search(r'감성지수:\s*(.*)', content)

        p_html = "<br>".join([f"&nbsp;&nbsp;• {sanitize_text(x)}" for x in pos]) or "호재 미포착"
        n_html = "<br>".join([f"&nbsp;&nbsp;• {sanitize_text(x)}" for x in neg]) or "악재 미포착"

        raw_html = f"""
        🟢 <b>긍정 호재:</b><br>{p_html}<br><br>
        🔴 <b>부정 리스크:</b><br>{n_html}<br><br>
        🚀 <b>주도 테마:</b> <span style="color:#38bdf8; font-weight:bold;">{sanitize_text(th.group(1) if th else '특이 테마 없음')}</span><br>
        📊 <b>뉴스 감성 지수:</b> <span class="highlight-val">{sanitize_text(sc.group(1) if sc else '0점')}</span>
        """
        save_ai_cache(cache_key, {"status": st, "briefing_html": raw_html})
        return st, raw_html, now_str
    except Exception as e:
        return "보통 🟡", f"분석 오류: {e}", now_str

def extract_peaks_and_troughs(df_60, is_krw=True):
    try:
        c = df_60['Close'].values
        peaks = [c[i] for i in range(2, len(c)-2) if c[i]>c[i-1] and c[i]>c[i-2] and c[i]>c[i+1] and c[i]>c[i+2]]
        troughs = [c[i] for i in range(2, len(c)-2) if c[i]<c[i-1] and c[i]<c[i-2] and c[i]<c[i+1] and c[i]<c[i+2]]
        return f"최근 반등 저점({fmt_price(troughs[-1], is_krw)}) -> 저항 고점({fmt_price(peaks[-1], is_krw)})"
    except Exception:
        return "파동 안정화 진행 중"

def parse_price_from_text(text, key_prefix, is_krw=True, current_price=0.0):
    if not text: return None
    try:
        m = re.search(rf'{key_prefix}\s*:\s*([^\n]+)', text)
        if m:
            digits = re.findall(r'[\d\.]+', m.group(1).replace(',', ''))
            if digits and float(digits[0]) > 0:
                return adjust_to_tick_size(float(digits[0]), is_krw)
    except Exception:
        pass
    return None

# =========================================================
# 🤖 일반 종목 분석 + 복합 스코어 (-100 ~ +100)
# =========================================================
def generate_ai_stock_analysis(stock_name, symbol, news_keywords, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, latest_close, ma20_d, ma60_d, ma120_d, atr_val=0.0, supply_type="", currency_symbol="원", force_refresh=False):
    cache_key = f"STOCK_{symbol}"
    is_krw = (currency_symbol in ["원", "KRW"])

    if not force_refresh and is_cache_valid(cache_key, max_hours=4):
        cached = ai_cache_store[cache_key]
        return cached.get('reason', ''), cached.get('basic_report', ''), cached.get('deep_report', ''), cached.get('parsed_prices', {}), cached.get('score_info', {"score": 70, "label": "적극 매수 🔵", "p_reason": "-", "t_reason": "-", "m_reason": "-"}), cached.get('updated_at', now_str)

    if not llm_mgr.is_available():
        if cache_key in ai_cache_store:
            cached = ai_cache_store[cache_key]
            return cached.get('reason', ''), cached.get('basic_report', ''), cached.get('deep_report', ''), cached.get('parsed_prices', {}), cached.get('score_info', {"score": 0, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}), cached.get('updated_at', now_str)
        return "수급/모멘텀 모니터링", "분석 준비 중", "", {"buy": None, "stop": None, "target1": None, "target2": None}, {"score": 0, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}, now_str

    prompt = f"""
너는 퀀트 트레이더이다. [현재 주가가 진입하기에 매력적인 자리인가? (손익비/안전마진)]을 평가하여 -100~+100점의 스코어와 리포트를 도출하라.
[종목] {stock_name} ({symbol}) | 수급: {supply_type}
[현재가] {fmt_price(latest_close, is_krw)} | 20일선({fmt_price(ma20_d, is_krw)}), 60일선({fmt_price(ma60_d, is_krw)})
[지표] RSI({rsi_val}), MACD({macd_status}), 매물대POC({fmt_price(poc_price, is_krw)})
[15일 시세]
{raw_data_str_15days}

[출력 양식 - 필수 준수]
파싱_스코어점수: <+00 또는 -00>
파싱_스코어라벨: <적극 매수 🔵 OR 눌림목 매수 🟢 OR 관망 🟡 OR 비중 축소 🟠 OR 전량 매도 🔴>
파싱_가격매력도근거: <20일선 지지 및 손익비 평가 1줄>
파싱_기술지표근거: <RSI 및 이평선 지표 평가 1줄>
파싱_뉴스수급근거: <수급 및 모멘텀 지속성 1줄>
선정이유: <선정 이유 요약 2줄>
파싱_눌림목가: <{int(latest_close*0.98) if is_krw else round(latest_close*0.98,2)}>
파싱_손절가: <{int(latest_close*0.95) if is_krw else round(latest_close*0.95,2)}>
파싱_1차익절가: <{int(latest_close*1.05) if is_krw else round(latest_close*1.05,2)}>
파싱_2차익절가: <{int(latest_close*1.10) if is_krw else round(latest_close*1.10,2)}>

핵심요약리포트:
📌 [차트 종합 진단]
- 내용 요약 2줄
🟢 [안전 매수/손절 전략]
- 진입 타점 및 손절 근거 요약
🚀 [분할 익절 전략]
- 1차, 2차 목표가 요약

상세리포트:
• 기술적 지표 및 패턴 상세 서술
• 리스크 관리 및 대응 가이드 서술
[언어 제한] 한자/일본어 절대 금지.
"""
    try:
        content = llm_mgr.generate_completion(prompt, temperature=0.3, max_tokens=1400)
        
        reason_m = re.search(r'선정이유:\s*(.*)', content)
        b_m = re.search(r'핵심요약리포트:\s*([\s\S]*?)(?=상세리포트:|$)', content)
        d_m = re.search(r'상세리포트:\s*([\s\S]*)', content)
        
        sc_m = re.search(r'파싱_스코어점수:\s*([+-]?\d+)', content)
        score_val = int(sc_m.group(1)) if sc_m else 70
        lb_m = re.search(r'파싱_스코어라벨:\s*(.*)', content)
        label_val = lb_m.group(1).strip() if lb_m else "적극 매수 🔵"

        p_m = re.search(r'파싱_가격매력도근거:\s*(.*)', content)
        t_m = re.search(r'파싱_기술지표근거:\s*(.*)', content)
        m_m = re.search(r'파싱_뉴스수급근거:\s*(.*)', content)

        score_info = {
            "score": score_val,
            "label": label_val,
            "p_reason": sanitize_text(p_m.group(1) if p_m else "지지선 근접으로 손익비 양호"),
            "t_reason": sanitize_text(t_m.group(1) if t_m else f"Wilder RSI {rsi_val} 탄력 유지"),
            "m_reason": sanitize_text(m_m.group(1) if m_m else "주도 수급 모멘텀 지속")
        }

        ai_buy = parse_price_from_text(content, "파싱_눌림목가", is_krw, latest_close)
        ai_stop = parse_price_from_text(content, "파싱_손절가", is_krw, latest_close)
        ai_target1 = parse_price_from_text(content, "파싱_1차익절가", is_krw, latest_close)
        ai_target2 = parse_price_from_text(content, "파싱_2차익절가", is_krw, latest_close)
        if ai_stop and atr_val > 0 and ai_buy:
            ai_stop = validate_stop_loss_with_atr(ai_buy, ai_stop, atr_val, is_krw)

        parsed_prices = {"buy": ai_buy, "stop": ai_stop, "target1": ai_target1, "target2": ai_target2}

        save_ai_cache(cache_key, {
            "reason": sanitize_text(reason_m.group(1) if reason_m else ""),
            "basic_report": sanitize_text(b_m.group(1) if b_m else content),
            "deep_report": sanitize_text(d_m.group(1) if d_m else ""),
            "parsed_prices": parsed_prices,
            "score_info": score_info
        })
        return sanitize_text(reason_m.group(1) if reason_m else ""), sanitize_text(b_m.group(1) if b_m else content), sanitize_text(d_m.group(1) if d_m else ""), parsed_prices, score_info, now_str
    except Exception as e:
        print(f"⚠️ {stock_name} AI 실패: {e}")
        return "수급 관망", f"AI 호출 오류: {e}", "", {"buy": None, "stop": None, "target1": None, "target2": None}, {"score": 0, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}, now_str

# =========================================================
# 🎯 마이 대시보드 포지션 분석 + 8단계 결론 라벨
# =========================================================
def generate_ai_toss_3line_analysis(stock_name, symbol, avg_price, current_price, return_pct, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, is_krw=True, force_refresh=False):
    cache_key = f"TOSS_MY_{symbol}"
    if not force_refresh and is_cache_valid(cache_key, max_hours=4):
        c = ai_cache_store[cache_key]
        return c.get('deep_report', ''), c.get('stop_price'), c.get('target_price'), c.get('pyramid_price'), c.get('pyramid_type'), c.get('score_info', {"score": 50, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}), c.get('updated_at', now_str)

    if not llm_mgr.is_available():
        if cache_key in ai_cache_store:
            c = ai_cache_store[cache_key]
            return c.get('deep_report', ''), c.get('stop_price'), c.get('target_price'), c.get('pyramid_price'), c.get('pyramid_type'), c.get('score_info', {"score": 50, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}), c.get('updated_at', now_str)
        return "분석 준비 중", None, None, None, None, {"score": 0, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}, now_str

    prompt = f"""
너는 포트폴리오 트레이딩 전문가이다. [실제 계좌 손익 ({return_pct:+.2f}%)]과 차트 지지/저항을 결합하여 포지션 스코어와 8단계 결론을 도출하라.
[종목] {stock_name} ({symbol}) | 내 평단: {fmt_price(avg_price, is_krw)} | 현재가: {fmt_price(current_price, is_krw)}
[지표] RSI({rsi_val}), MACD({macd_status}), POC({fmt_price(poc_price, is_krw)})
[15일 시세]
{raw_data_str_15days}

[8단계 결론 라벨 중 1개 선택]
1. [일부 익절 🟢 - 수익권 중 상단 저항 직면 / 분할 차익 실현]
2. [비중 축소 🟠 - 손실권 중 상단 저항 직면 / 손실 축소 필요]
3. [익절 🟢 - 수익권 중 지지선 이탈 / 이익 실현 필요]
4. [일부 손절 🔴 - 손실권 중 단기 지지 이탈 / 비중 축소 권장]
5. [손절 🔴 - 손실권 중 주요 추세선 붕괴 확인 / 전량 청산 권장]
6. [불타기 고려 🔵 - 상승 추세 및 눌림목 지지 확인]
7. [물타기 고려 🟠 - 장기 지지 및 바닥 반등 확인]
8. [관망 🟡 - 추세 유지 및 박스권 횡보 중 / 포지션 유지]

[출력 양식 - 필수 준수]
파싱_포지션점수: <+00 또는 -00>
파싱_포지션라벨: <선택한 결론 핵심 문구>
파싱_가격매력도근거: <평단 대비 수익률 위치 및 손익비 평가 1줄>
파싱_지지저항근거: <매물대 POC 지지/저항 구조 1줄>
파싱_대응권고: <분할 대응 실행 권고 1줄>
파싱_추매타입: <불타기 OR 물타기 OR 없음>
파싱_추매추천가: <0 또는 가격>
파싱_Trailing손절가: <{int(current_price*0.95) if is_krw else round(current_price*0.95,2)}>
파싱_동적목표가: <{int(current_price*1.08) if is_krw else round(current_price*1.08,2)}>

상세가이드:
결론: [위 8개 표준 라벨 중 1개 정확히 출력]
• [포지션 및 수급 진단] 상세 서술
• [동적 목표가 시나리오] 상세 서술
• [Trailing Stop 전략] 상세 서술
[언어 제한] 한자/일본어 절대 금지.
"""
    try:
        content = llm_mgr.generate_completion(prompt, temperature=0.3, max_tokens=1200)
        stop_val = parse_price_from_text(content, "파싱_Trailing손절가", is_krw, current_price)
        target_val = parse_price_from_text(content, "파싱_동적목표가", is_krw, current_price)
        pyr_val = parse_price_from_text(content, "파싱_추매추천가", is_krw, current_price)
        pyr_type_m = re.search(r'파싱_추매타입:\s*(불타기|물타기)', content)
        pyr_type = pyr_type_m.group(1) if (pyr_type_m and pyr_val and pyr_val > 0) else None

        sc_m = re.search(r'파싱_포지션점수:\s*([+-]?\d+)', content)
        lb_m = re.search(r'파싱_포지션라벨:\s*(.*)', content)
        p_m = re.search(r'파싱_가격매력도근거:\s*(.*)', content)
        t_m = re.search(r'파싱_지지저항근거:\s*(.*)', content)
        m_m = re.search(r'파싱_대응권고:\s*(.*)', content)

        score_info = {
            "score": int(sc_m.group(1)) if sc_m else 50,
            "label": lb_m.group(1).strip() if lb_m else "관망 🟡",
            "p_reason": sanitize_text(p_m.group(1) if p_m else f"평단 대비 {return_pct:+.2f}% 위치"),
            "t_reason": sanitize_text(t_m.group(1) if t_m else "매물대 지지 테스트 중"),
            "m_reason": sanitize_text(m_m.group(1) if m_m else "주요 이탈선 준수")
        }

        d_m = re.search(r'상세가이드:\s*([\s\S]*)', content)
        deep_text = d_m.group(1).strip() if d_m else content

        save_ai_cache(cache_key, {
            "deep_report": sanitize_text(deep_text),
            "stop_price": stop_val,
            "target_price": target_val,
            "pyramid_price": pyr_val,
            "pyramid_type": pyr_type,
            "score_info": score_info
        })
        return sanitize_text(deep_text), stop_val, target_val, pyr_val, pyr_type, score_info, now_str
    except Exception as e:
        print(f"⚠️ {stock_name} 마이 대시보드 AI 실패: {e}")
        return f"분석 오류: {e}", None, None, None, None, {"score": 0, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}, now_str

def update_and_get_consecutive_days(symbol, is_new_pivot_cycle):
    tracker_key = "RECOMMEND_DAYS_TRACKER"
    tracker = ai_cache_store.get(tracker_key, {})
    item_info = tracker.get(symbol, {"days": 1, "last_pivot_date": ""})
    today_str = today_date.strftime("%Y-%m-%d")
    
    if is_new_pivot_cycle:
        last_date = item_info.get("last_pivot_date", "")
        if last_date:
            try:
                diff = (today_date - datetime.datetime.strptime(last_date, "%Y-%m-%d").date()).days
                item_info["days"] = (item_info.get("days", 1) + 1) if 1 <= diff <= 3 else 1
            except Exception:
                item_info["days"] = 1
        else:
            item_info["days"] = 1
        item_info["last_pivot_date"] = today_str
        tracker[symbol] = item_info
        ai_cache_store[tracker_key] = tracker
        save_ai_cache(tracker_key, tracker)

    cnt = item_info.get("days", 1)
    return f'<span class="badge-item" style="background:#dc2626;">{cnt}일 연속 추천 🔥</span>' if cnt > 1 else '<span class="badge-item" style="background:#2563eb;">1일차 🆕</span>'

def get_crash_defense_badge(stock_daily_chg, market_avg_chg, defense_mode):
    if not defense_mode: return ""
    if stock_daily_chg >= (market_avg_chg + 1.5):
        return '<span class="badge-item" style="background:#15803d;">지수 방어 양호 🛡️</span>'
    elif stock_daily_chg <= (market_avg_chg - 2.0):
        return '<span class="badge-item" style="background:#b91c1c;">고변동성 하락 ⚠️</span>'
    return ""

# =========================================================
# PART 1: 🇰🇷 국장(index.html) 스캔 및 렌더링
# =========================================================
print("\n" + "="*60)
print("🇰🇷 [PART 1] 한국 증시 스캔 & 스코어 리포트 생성 중...")
print("="*60)

kr_is_open, kr_open_msg = get_market_open_status("KR")
kr_witching_alerts = get_witching_day_alert("KR")
kr_econ_events = get_economic_calendar_events("KR")
kr_emergency, kr_vol_banner, kr_defense_mode, kr_avg_chg = check_market_volatility_trigger("KR")
kr_7d_news = get_naver_7days_news()
kr_market_status, kr_sentiment_briefing, kr_briefing_time = analyze_7days_news_sentiment("대한민국 주식시장(국장)", kr_7d_news, force_refresh=kr_emergency)

kr_banner_items = kr_witching_alerts + kr_econ_events
if kr_vol_banner: kr_banner_items.insert(0, kr_vol_banner)
if not kr_is_open: kr_banner_items.insert(0, f"<b>[오늘 휴장일]</b> {kr_open_msg}")

kr_banner_html = f"""
<div class="event-banner">
    <div class="event-banner-title">🚨 [시장 변동성 주의] 주요 일정 & 만기일 캘린더</div>
    <div class="event-banner-content">{'<br>'.join(kr_banner_items)}</div>
</div>
""" if kr_banner_items else ""

kr_needs_refresh = should_refresh_daily_pivot("대한민국 주식시장(국장)") or kr_emergency
kr_selected_cache_key = "SELECTED_KR_TARGETS"

if not kr_needs_refresh and kr_selected_cache_key in ai_cache_store:
    selected_kr_targets = ai_cache_store[kr_selected_cache_key].get("targets", {})
else:
    def get_naver_multi_sise():
        cmap = {}
        for biz, sosok in [("dealForeign", "0"), ("dealForeign", "1"), ("dealOrgan", "0"), ("dealOrgan", "1"), ("topAmount", "0"), ("topAmount", "1")]:
            try:
                res = requests.get(f"https://m.stock.naver.com/api/json/sise/siseListJson.nhn?bizType={biz}&sosok={sosok}", headers=headers, timeout=6)
                if res.status_code == 200:
                    for item in res.json().get('result', {}).get('itemList', [])[:20]:
                        name, cd = item.get('nm'), item.get('cd')
                        if name and cd and name not in cmap:
                            cmap[name] = f"{cd}.KS" if sosok == "0" else f"{cd}.KQ"
            except Exception:
                pass
        return cmap

    raw_kr_candidates = get_naver_multi_sise()
    scored_kr_stocks = []
    for name, symbol in list(raw_kr_candidates.items())[:50]:
        try:
            df_hist = yf.Ticker(symbol).history(period="3mo", interval="1d")
            if df_hist is None or len(df_hist) < 25: continue
            last_c = float(df_hist['Close'].iloc[-1])
            if last_c < 1000 or (df_hist['Volume'].tail(5).mean() * last_c) < 10_000_000_000: continue
            
            ma5 = df_hist['Close'].rolling(5).mean().iloc[-1]
            ma20 = df_hist['Close'].rolling(20).mean().iloc[-1]
            ma60 = df_hist['Close'].rolling(60).mean().iloc[-1]
            score = 0
            if last_c >= ma20: score += 30
            if ma5 >= ma20 >= ma60: score += 30
            if (df_hist['Volume'].iloc[-1] / (df_hist['Volume'].tail(20).mean() + 1e-9)) >= 1.5: score += 40
            scored_kr_stocks.append((score, name, symbol, "수급 주도 및 이평선 정배열"))
        except Exception:
            continue

    scored_kr_stocks.sort(key=lambda x: x[0], reverse=True)
    selected_kr_targets = {item[1]: (item[2], item[3]) for item in scored_kr_stocks[:5]}
    backup_kr = [("한미반도체", "042700.KS", "AI 반도체 수급 주도주"), ("알테오젠", "196170.KQ", "바이오 플랫폼 기관 매수세"), ("효성중공업", "298040.KS", "전력기기 인프라"), ("HD현대일렉트릭", "267260.KS", "전력망 성장주"), ("레인보우로보틱스", "277810.KQ", "로봇 모멘텀")]
    for b_n, b_s, b_f in backup_kr:
        if len(selected_kr_targets) >= 5: break
        if b_n not in selected_kr_targets: selected_kr_targets[b_n] = (b_s, b_f)
    save_ai_cache(kr_selected_cache_key, {"targets": selected_kr_targets})

stock_cards_kr_html = ""
for stock_name, (symbol, supply_type) in selected_kr_targets.items():
    try:
        pure_code = symbol.split('.')[0]
        df_daily = yf.Ticker(symbol).history(period="1y", interval="1d")
        if df_daily is None or len(df_daily) < 1: continue

        badge_html = update_and_get_consecutive_days(symbol, kr_needs_refresh)
        kr_stock_chg = ((df_daily['Close'].iloc[-1] - df_daily['Close'].iloc[-2]) / df_daily['Close'].iloc[-2]) * 100.0 if len(df_daily)>=2 else 0.0
        defense_badge = get_crash_defense_badge(kr_stock_chg, kr_avg_chg, kr_defense_mode)

        df_daily['MA20'] = df_daily['Close'].rolling(20, min_periods=1).mean()
        df_daily['MA60'] = df_daily['Close'].rolling(60, min_periods=1).mean()
        df_daily['MA120'] = df_daily['Close'].rolling(120, min_periods=1).mean()
        std20 = df_daily['Close'].rolling(20, min_periods=1).std().fillna(0)
        df_daily['BB_Upper'] = df_daily['MA20'] + (std20 * 2)
        df_daily['BB_Lower'] = df_daily['MA20'] - (std20 * 2)

        df_daily['RSI'], df_daily['RSI_Signal'] = calculate_wilder_rsi(df_daily['Close'], period=14, signal_period=9)
        rsi_val = round(float(df_daily['RSI'].iloc[-1]), 2)
        rsi_signal_val = round(float(df_daily['RSI_Signal'].iloc[-1]), 2)
        rsi_cross = "RSI 상향 돌파 📈" if rsi_val > rsi_signal_val else "RSI 조정 📉"

        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_status = "골든크로스 📈" if df_daily['MACD'].iloc[-1] > df_daily['Signal'].iloc[-1] else "데드크로스 📉"

        latest_close = int(df_daily['Close'].iloc[-1])
        ma20_d = int(df_daily['MA20'].iloc[-1])
        ma60_d = int(df_daily['MA60'].iloc[-1])
        ma120_d = int(df_daily['MA120'].iloc[-1])

        poc_price = int(df_daily['Close'].tail(120).mean())
        atr_val = calculate_atr(df_daily, period=14)

        df_recent15 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15)
        raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:{int(r['Open']):,}원 | High:{int(r['High']):,}원 | Low:{int(r['Low']):,}원 | Close:{int(r['Close']):,}원 | Vol:{int(r['Volume']):,}" for idx, r in df_recent15.iterrows()]
        raw_data_str_15days = "\n".join(raw_lines)

        pick_reason, basic_ai_report, deep_ai_report, ai_prices, score_info, stock_ai_time = generate_ai_stock_analysis(
            stock_name, symbol, kr_7d_news, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross, macd_status, "정배열" if ma20_d>ma60_d>ma120_d else "혼조세", "밴드 내", "구름대 위", poc_price, int(df_daily['High'].tail(120).max()), int(df_daily['Low'].tail(120).min()), extract_peaks_and_troughs(df_daily.tail(60), True), latest_close, ma20_d, ma60_d, ma120_d, atr_val, supply_type, "원", force_refresh=kr_emergency
        )

        buy_price_str = f"{int(round(ai_prices['buy'])):,}원" if ai_prices.get('buy') else "진입가 산출 대기"
        stop_loss_str = f"{int(round(ai_prices['stop'])):,}원" if ai_prices.get('stop') else "손절가 산출 대기"
        target_price_str = f"{int(round(ai_prices['target1'])):,}원" if ai_prices.get('target1') else "목표가 산출 대기"

        df_chart = df_daily.tail(120)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.25, 0.2])
        fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='주가'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='orange', width=1.2), name='20일선'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA60'], line=dict(color='purple', width=1.2), name='60일선'), row=1, col=1)
        fig.add_hline(y=poc_price, line_dash="dot", line_color="#facc15", annotation_text=f"POC: {fmt_price(poc_price, True)}", row=1, col=1)
        colors = ['#f87171' if c < o else '#4ade80' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='거래량'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#38bdf8', width=1.2), name='RSI'), row=3, col=1)
        fig.update_layout(height=480, margin=dict(l=5, r=5, t=30, b=20), xaxis_rangeslider_visible=False, template="plotly_dark")
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

        score_color = "#38bdf8" if score_info["score"] >= 70 else ("#4ade80" if score_info["score"] >= 30 else ("#facc15" if score_info["score"] >= -29 else "#f87171"))

        stock_cards_kr_html += f"""
        <div class="card">
            <div class="console-report">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div class="report-header">{stock_name} ({pure_code}) {badge_html} {defense_badge}</div>
                    <a href="https://www.tradingview.com/symbols/KRX-{pure_code}/" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
                </div>
                
                <div style="background:#111827; border:1px solid #374151; border-radius:8px; padding:10px 14px; margin:10px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:13px; color:#94a3b8; font-weight:bold;">🎯 AI 복합 매매 스코어</span>
                        <span style="font-size:15px; font-weight:bold; color:{score_color};">{score_info['score']:+d}점 [{score_info['label']}]</span>
                    </div>
                    <div style="font-size:12px; color:#cbd5e1; line-height:1.6; border-top:1px dashed #374151; padding-top:6px;">
                        • <b>가격 매력도:</b> {score_info['p_reason']}<br>
                        • <b>기술 지표:</b> {score_info['t_reason']}<br>
                        • <b>뉴스/수급:</b> {score_info['m_reason']}
                    </div>
                </div>

                <div class="stock-reason-box">💡 <b>선정 이유:</b><br>{pick_reason}</div>
                <div class="report-divider"></div>
                <div class="report-line">• 종가 기준 현재가 : <span class="highlight-val">{fmt_price(latest_close, True)}</span></div>
                <div class="report-line">• RSI / MACD : {rsi_val} / {macd_status}</div>
                <div class="report-line" style="color:#38bdf8; font-weight:bold;">🎯 AI 추천 진입가 : {buy_price_str}</div>
                <div class="report-line text-red">🛑 AI 산출 손절가 : {stop_loss_str}</div>
                <div class="report-line text-green">🚀 AI 산출 1차 익절가 : {target_price_str}</div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ AI 핵심 매매 전략 <span class="sub-desc">({stock_ai_time})</span></div>
                <div class="ai-content" style="white-space: pre-line;">{basic_ai_report}</div>
                {f'<details class="deep-report-accordion"><summary class="deep-report-btn">🔍 AI 심도 분석 더보기 ▼</summary><div class="deep-report-content" style="white-space: pre-line;">{deep_ai_report}</div></details>' if deep_ai_report else ''}
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e:
        print(f"⚠️ {stock_name} 생성 실패: {e}")

# =========================================================
# PART 2: 🇺🇸 미장(us_index.html) 스캔 및 3중 상폐 차단
# =========================================================
print("\n" + "="*60)
print("🇺🇸 [PART 2] 미국 증시 스캔 & 3중 상폐 필터링...")
print("="*60)

us_is_open, us_open_msg = get_market_open_status("US")
us_witching_alerts = get_witching_day_alert("US")
us_econ_events = get_economic_calendar_events("US")
us_emergency, us_vol_banner, us_defense_mode, us_avg_chg = check_market_volatility_trigger("US")
us_7d_news = get_yahoo_7days_news()
us_market_status, us_sentiment_briefing, us_briefing_time = analyze_7days_news_sentiment("미국 주식시장(미장)", us_7d_news, force_refresh=us_emergency)

us_banner_items = us_witching_alerts + us_econ_events
if us_vol_banner: us_banner_items.insert(0, us_vol_banner)
if not us_is_open: us_banner_items.insert(0, f"<b>[오늘 휴장일]</b> {us_open_msg}")

us_banner_html = f"""
<div class="event-banner">
    <div class="event-banner-title">🚨 [시장 변동성 주의] 주요 일정 & 만기일 캘린더</div>
    <div class="event-banner-content">{'<br>'.join(us_banner_items)}</div>
</div>
""" if us_banner_items else ""

us_needs_refresh = should_refresh_daily_pivot("미국 주식시장(미장)") or us_emergency
us_selected_cache_key = "SELECTED_US_TARGETS"

if not us_needs_refresh and us_selected_cache_key in ai_cache_store:
    selected_us_targets = ai_cache_store[us_selected_cache_key].get("targets", {})
else:
    def get_us_candidates():
        scanned = []
        for url in ["https://finance.yahoo.com/markets/stocks/most-active/", "https://finance.yahoo.com/markets/stocks/gainers/"]:
            try:
                soup = BeautifulSoup(requests.get(url, headers=headers, timeout=8).text, 'html.parser')
                for a in soup.find_all('a', href=True):
                    if '/quote/' in a['href']:
                        sym = a['href'].split('/quote/')[1].split('?')[0].split('/')[0].upper()
                        if sym.isalpha() and len(sym) <= 5 and sym not in scanned: scanned.append(sym)
            except Exception:
                pass
        for b in ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'AMZN', 'GOOGL', 'META', 'AVGO', 'PLTR', 'COST', 'NFLX']:
            if b not in scanned: scanned.append(b)
        return scanned

    selected_us_targets = {}
    for sym in get_us_candidates():
        if len(selected_us_targets) >= 10: break
        try:
            tk = yf.Ticker(sym)
            df_c = tk.history(period="5d", interval="1d")
            # 🛡️ 3중 상폐 차단 필터
            if df_c is None or len(df_c) < 2: continue
            if float(df_c['Volume'].iloc[-1]) <= 1000: continue
            if (today_date - df_c.index[-1].date()).days > 7: continue
            if float(df_c['Close'].iloc[-1]) <= 0.5: continue
            if tk.info.get('marketCap', 0) >= 10_000_000_000:
                selected_us_targets[tk.info.get('shortName', sym)] = (sym, "🔥 Wall Street 거래대금 상위 및 빅테크 주도주")
        except Exception:
            continue
    save_ai_cache(us_selected_cache_key, {"targets": selected_us_targets})

stock_cards_us_html = ""
for stock_name, (symbol, supply_type) in selected_us_targets.items():
    try:
        df_daily = yf.Ticker(symbol).history(period="1y", interval="1d")
        if df_daily is None or len(df_daily) < 1: continue

        badge_html = update_and_get_consecutive_days(symbol, us_needs_refresh)
        us_stock_chg = ((df_daily['Close'].iloc[-1] - df_daily['Close'].iloc[-2]) / df_daily['Close'].iloc[-2]) * 100.0 if len(df_daily)>=2 else 0.0
        defense_badge = get_crash_defense_badge(us_stock_chg, us_avg_chg, us_defense_mode)

        df_daily['MA20'] = df_daily['Close'].rolling(20, min_periods=1).mean()
        df_daily['MA60'] = df_daily['Close'].rolling(60, min_periods=1).mean()
        df_daily['MA120'] = df_daily['Close'].rolling(120, min_periods=1).mean()
        df_daily['RSI'], df_daily['RSI_Signal'] = calculate_wilder_rsi(df_daily['Close'], period=14, signal_period=9)
        rsi_val = round(float(df_daily['RSI'].iloc[-1]), 2)
        rsi_signal_val = round(float(df_daily['RSI_Signal'].iloc[-1]), 2)

        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_status = "골든크로스 📈" if df_daily['MACD'].iloc[-1] > df_daily['Signal'].iloc[-1] else "데드크로스 📉"

        latest_close = round(float(df_daily['Close'].iloc[-1]), 2)
        ma20_d = round(float(df_daily['MA20'].iloc[-1]), 2)
        ma60_d = round(float(df_daily['MA60'].iloc[-1]), 2)
        ma120_d = round(float(df_daily['MA120'].iloc[-1]), 2)
        poc_price = round(float(df_daily['Close'].tail(120).mean()), 2)
        atr_val = calculate_atr(df_daily, period=14)

        df_recent15 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15)
        raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:${r['Open']:.2f} | High:${r['High']:.2f} | Low:${r['Low']:.2f} | Close:${r['Close']:.2f} | Vol:{int(r['Volume']):,}" for idx, r in df_recent15.iterrows()]
        raw_data_str_15days = "\n".join(raw_lines)

        pick_reason, basic_ai_report, deep_ai_report, ai_prices, score_info, stock_ai_time = generate_ai_stock_analysis(
            stock_name, symbol, us_7d_news, raw_data_str_15days, rsi_val, rsi_signal_val, "RSI 유지", macd_status, "정배열" if ma20_d>ma60_d>ma120_d else "혼조세", "밴드 내", "구름대 위", poc_price, round(float(df_daily['High'].tail(120).max()),2), round(float(df_daily['Low'].tail(120).min()),2), extract_peaks_and_troughs(df_daily.tail(60), False), latest_close, ma20_d, ma60_d, ma120_d, atr_val, supply_type, "$", force_refresh=us_emergency
        )

        buy_price_str = f"${ai_prices['buy']:.2f}" if ai_prices.get('buy') else "진입가 산출 대기"
        stop_loss_str = f"${ai_prices['stop']:.2f}" if ai_prices.get('stop') else "손절가 산출 대기"
        target_price_str = f"${ai_prices['target1']:.2f}" if ai_prices.get('target1') else "목표가 산출 대기"

        df_chart = df_daily.tail(120)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.25, 0.2])
        fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='주가'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='orange', width=1.2), name='20일선'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA60'], line=dict(color='purple', width=1.2), name='60일선'), row=1, col=1)
        fig.add_hline(y=poc_price, line_dash="dot", line_color="#facc15", annotation_text=f"POC: {fmt_price(poc_price, False)}", row=1, col=1)
        colors = ['#f87171' if c < o else '#4ade80' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='거래량'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#38bdf8', width=1.2), name='RSI'), row=3, col=1)
        fig.update_layout(height=480, margin=dict(l=5, r=5, t=30, b=20), xaxis_rangeslider_visible=False, template="plotly_dark")
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

        score_color = "#38bdf8" if score_info["score"] >= 70 else ("#4ade80" if score_info["score"] >= 30 else ("#facc15" if score_info["score"] >= -29 else "#f87171"))

        stock_cards_us_html += f"""
        <div class="card">
            <div class="console-report">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div class="report-header">{stock_name} ({symbol}) {badge_html} {defense_badge}</div>
                    <a href="https://www.tradingview.com/symbols/{symbol}/" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
                </div>

                <div style="background:#111827; border:1px solid #374151; border-radius:8px; padding:10px 14px; margin:10px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:13px; color:#94a3b8; font-weight:bold;">🎯 AI 복합 매매 스코어</span>
                        <span style="font-size:15px; font-weight:bold; color:{score_color};">{score_info['score']:+d}점 [{score_info['label']}]</span>
                    </div>
                    <div style="font-size:12px; color:#cbd5e1; line-height:1.6; border-top:1px dashed #374151; padding-top:6px;">
                        • <b>가격 매력도:</b> {score_info['p_reason']}<br>
                        • <b>기술 지표:</b> {score_info['t_reason']}<br>
                        • <b>뉴스/수급:</b> {score_info['m_reason']}
                    </div>
                </div>

                <div class="stock-reason-box">💡 <b>선정 이유:</b><br>{pick_reason}</div>
                <div class="report-divider"></div>
                <div class="report-line">• 종가 기준 현재가 : <span class="highlight-val">{fmt_price(latest_close, False)}</span></div>
                <div class="report-line">• RSI / MACD : {rsi_val} / {macd_status}</div>
                <div class="report-line" style="color:#38bdf8; font-weight:bold;">🎯 AI 추천 진입가 : {buy_price_str}</div>
                <div class="report-line text-red">🛑 AI 산출 손절가 : {stop_loss_str}</div>
                <div class="report-line text-green">🚀 AI 산출 1차 익절가 : {target_price_str}</div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ AI 핵심 매매 전략 <span class="sub-desc">({stock_ai_time})</span></div>
                <div class="ai-content" style="white-space: pre-line;">{basic_ai_report}</div>
                {f'<details class="deep-report-accordion"><summary class="deep-report-btn">🔍 AI 심도 분석 더보기 ▼</summary><div class="deep-report-content" style="white-space: pre-line;">{deep_ai_report}</div></details>' if deep_ai_report else ''}
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e:
        print(f"⚠️ {stock_name} 생성 실패: {e}")

# =========================================================
# PART 3: 🎯 마이 대시보드(index3.html) - 토스 실계좌 잔고
# =========================================================
print("\n" + "="*60)
print("🎯 [PART 3] 토스 실계좌 잔고 & 8단계 라벨 포지션 스코어...")
print("="*60)

def get_toss_holdings():
    if not TOSS_CLIENT_ID or not TOSS_CLIENT_SECRET:
        return [
            {"ticker": "005930.KS", "name": "삼성전자", "avg_price": 72000, "current_price": 74500, "eval_amount": 3725000, "profit_loss": 125000, "return_pct": 3.47, "quantity": 50, "market": "KR", "currency": "KRW"},
            {"ticker": "NVDA", "name": "NVIDIA", "avg_price": 115.0, "current_price": 128.5, "eval_amount": 1927.5, "profit_loss": 202.5, "return_pct": 11.74, "quantity": 15, "market": "US", "currency": "USD"}
        ], False

    try:
        t_res = toss_request_with_proxy_failover("POST", "https://openapi.tossinvest.com/oauth2/token", data={"grant_type": "client_credentials", "client_id": TOSS_CLIENT_ID, "client_secret": TOSS_CLIENT_SECRET}, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=12)
        if t_res.status_code == 200:
            token = t_res.json().get("access_token")
            b_heads = {"Authorization": f"Bearer {token}", "x-api-key": TOSS_CLIENT_ID, "Content-Type": "application/json"}
            a_res = toss_request_with_proxy_failover("GET", "https://openapi.tossinvest.com/api/v1/accounts", headers=b_heads, timeout=12)
            seq = a_res.json().get("result", [{}])[0].get("accountSeq", 1) if a_res.status_code == 200 else 1
            
            b_heads["X-Tossinvest-Account"] = str(seq)
            h_res = toss_request_with_proxy_failover("GET", "https://openapi.tossinvest.com/api/v1/holdings", headers=b_heads, timeout=12)
            if h_res.status_code == 200:
                items = h_res.json().get("result", {}).get("items", [])
                holdings = []
                for it in items:
                    sym = str(it.get("symbol") or it.get("stockCode") or "").strip()
                    qty = float(it.get("quantity") or it.get("holdingQuantity") or 0)
                    if qty > 0:
                        holdings.append({
                            "ticker": sym,
                            "name": str(it.get("name") or it.get("stockName") or sym).strip(),
                            "avg_price": float(it.get("averagePurchasePrice") or it.get("avgPrice") or 0),
                            "current_price": float(it.get("lastPrice") or 0),
                            "eval_amount": float((it.get("marketValue") or {}).get("amountAfterCost") or (it.get("lastPrice",0)*qty)),
                            "profit_loss": float((it.get("profitLoss") or {}).get("amountAfterCost") or 0),
                            "return_pct": float((it.get("profitLoss") or {}).get("rateAfterCost") or 0.0) * 100.0,
                            "quantity": qty,
                            "market": str(it.get("marketCountry", "KR")).upper(),
                            "currency": str(it.get("currency", "KRW")).upper()
                        })
                if holdings: return holdings, True
    except Exception as e:
        print(f"⚠️ 토스 API 예외: {e}")
    return [], False

toss_holdings, is_real_toss = get_toss_holdings()

if is_real_toss:
    held = set(h['ticker'] for h in toss_holdings)
    for k in list(ai_cache_store.keys()):
        if k.startswith("TOSS_MY_") and k.replace("TOSS_MY_", "") not in held:
            del ai_cache_store[k]
    save_entire_cache(ai_cache_store)

my_stock_cards_html = ""
total_eval_my, total_profit_my = 0.0, 0.0

for h in toss_holdings:
    try:
        ticker = h['ticker']
        is_krw = (h['market'] == 'KR' or h['currency'] == 'KRW')
        fx = usd_krw_rate if not is_krw else 1.0
        eval_krw = h['eval_amount'] * fx
        profit_krw = h['profit_loss'] * fx
        pure_code = ticker.split('.')[0]

        yf_sym = f"{pure_code}.KS" if (is_krw and not ticker.endswith((".KS", ".KQ"))) else pure_code
        df_daily = yf.Ticker(yf_sym).history(period="6mo", interval="1d")
        if (df_daily is None or len(df_daily) < 1) and is_krw:
            df_daily = yf.Ticker(f"{pure_code}.KQ").history(period="6mo", interval="1d")

        if df_daily is not None and len(df_daily) > 0:
            df_daily['RSI'], df_daily['RSI_Signal'] = calculate_wilder_rsi(df_daily['Close'])
            rsi_v = round(float(df_daily['RSI'].iloc[-1]), 1)
            poc_v = float(df_daily['Close'].mean())
            raw_lines = [f"{idx.strftime('%m-%d')} Close:{r['Close']}" for idx, r in df_daily.tail(10).iterrows()]
            r_str = "\n".join(raw_lines)
        else:
            rsi_v, poc_v, r_str = 50.0, h['current_price'], "시세 수신 대기"

        deep_ai_guide, my_stop_val, my_target_val, my_pyr_val, my_pyr_type, my_score_info, my_guide_time = generate_ai_toss_3line_analysis(
            h['name'], ticker, h['avg_price'], h['current_price'], h['return_pct'], r_str, rsi_v, rsi_v, "유지", "안정", "정배열", "안정", "유효", poc_v, h['current_price']*1.2, h['current_price']*0.8, "마디점", is_krw, force_refresh=(kr_emergency if is_krw else us_emergency)
        )

        my_score_color = "#38bdf8" if my_score_info["score"] >= 70 else ("#4ade80" if my_score_info["score"] >= 30 else ("#facc15" if my_score_info["score"] >= -29 else "#f87171"))
        pyr_html = f'<div class="report-line" style="color:#38bdf8; font-weight:bold;">🎯 추천 추매가({my_pyr_type}) : {fmt_price(my_pyr_val, is_krw)}</div>' if (my_pyr_val and my_pyr_type) else ''

        my_stock_cards_html += f"""
        <div class="card">
            <div class="console-report">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div class="report-header">{'🇰🇷' if is_krw else '🇺🇸'} {h['name']} ({pure_code}) - {fmt_num(h['quantity'])}주</div>
                    <a href="https://www.tradingview.com/symbols/{f'KRX-{pure_code}' if is_krw else ticker}/" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
                </div>
                <div style="font-size:20px; font-weight:bold; margin-top:6px; color:#f8fafc;">
                    {fmt_price(eval_krw, True)} <span class="{'text-green' if profit_krw>=0 else 'text-red'}">({profit_krw:+,.0f}원)</span>
                </div>

                <div style="background:#111827; border:1px solid #374151; border-radius:8px; padding:10px 14px; margin:10px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:13px; color:#94a3b8; font-weight:bold;">🎯 포지션 파워 스코어</span>
                        <span style="font-size:15px; font-weight:bold; color:{my_score_color};">{my_score_info['score']:+d}점 [{my_score_info['label']}]</span>
                    </div>
                    <div style="font-size:12px; color:#cbd5e1; line-height:1.6; border-top:1px dashed #374151; padding-top:6px;">
                        • <b>가격 매력도:</b> {my_score_info['p_reason']}<br>
                        • <b>저항/지지:</b> {my_score_info['t_reason']}<br>
                        • <b>대응 권고:</b> {my_score_info['m_reason']}
                    </div>
                </div>

                <div class="report-divider"></div>
                <div class="report-line">평단가 : <span class="highlight-val">{fmt_price(h['avg_price'], is_krw)}</span> (<span class="{'text-green' if h['return_pct']>=0 else 'text-red'}">{h['return_pct']:+.2f}%</span>) &nbsp;|&nbsp; 현재가 : <span class="highlight-val">{fmt_price(h['current_price'], is_krw)}</span></div>
                {pyr_html}
                <div class="report-line text-red">🛑 손절선(Trailing) : {fmt_price(my_stop_val, is_krw) if my_stop_val else '산출 대기'}</div>
                <div class="report-line text-green">🚀 목표가 : {fmt_price(my_target_val, is_krw) if my_target_val else '산출 대기'}</div>
            </div>
            <details class="deep-report-accordion">
                <summary class="deep-report-btn">🔍 AI 포트폴리오 심도 대응 전략 자세히 보기 ▼</summary>
                <div class="deep-report-content" style="white-space: pre-line;">
                    <div style="font-size:14px; font-weight:bold; color:#4ade80; margin-bottom:8px;">⚡ AI 포지션 대응 분석 <span class="sub-desc">({my_guide_time})</span></div>
                    {deep_ai_guide}
                </div>
            </details>
        </div>
        """
        total_eval_my += eval_krw
        total_profit_my += profit_krw
    except Exception as e:
        print(f"⚠️ {h.get('name')} 처리 실패: {e}")

tot_cost = total_eval_my - total_profit_my
total_ret_pct = (total_profit_my / tot_cost * 100) if tot_cost > 0 else 0

# =========================================================
# PART 4: 📰 뉴스 인텔리전스 (index4.html) - SaveTicker 파싱 보강
# =========================================================
print("\n" + "="*60)
print("📰 [PART 4] SaveTicker 뉴스 인텔리전스 & 다중 자산 선별...")
print("="*60)

def scrape_saveticker_news():
    news_items = []
    try:
        soup = BeautifulSoup(requests.get("https://www.saveticker.com/news", headers=headers, timeout=8).text, 'html.parser')
        cards = soup.select('article') or soup.select('.news-card') or soup.select('li')
        for c in cards[:30]:
            t_elem = c.select_one('h2') or c.select_one('h3') or c.select_one('.title') or c.select_one('a')
            if not t_elem or len(t_elem.text.strip()) < 8: continue
            d_elem = c.select_one('p') or c.select_one('.desc') or c.select_one('.summary')
            time_elem = c.select_one('time') or c.select_one('.date') or c.select_one('.time')
            news_items.append({
                "title": t_elem.text.strip(),
                "desc": (d_elem.text.strip()[:180] if d_elem else t_elem.text.strip()),
                "time": (time_elem.text.strip() if time_elem else "최근 24시간 내")
            })
    except Exception as e:
        print(f"⚠️ SaveTicker 웹 파싱 예외: {e}")

    if len(news_items) < 5:
        for t in get_yahoo_7days_news().split('\n')[:20]:
            if len(t) > 8: news_items.append({"title": t, "desc": t, "time": "최근 12시간 내"})
    return news_items[:30]

saveticker_news_list = scrape_saveticker_news()
news_digest_text = "\n".join([f"[{it['time']}] {it['title']} - {it['desc']}" for it in saveticker_news_list])

def analyze_saveticker_macro_and_assets(news_text, force_refresh=False):
    cache_key = "SAVETICKER_INTEL_REPORT"
    if not force_refresh and is_cache_valid(cache_key, max_hours=3):
        return ai_cache_store[cache_key]

    if not llm_mgr.is_available():
        if cache_key in ai_cache_store: return ai_cache_store[cache_key]
        return {
            "verdict": "중립 🟡", "score": 5, "core_drivers": ["글로벌 거시 지표 방향성 탐색 중"],
            "asset_hedges": [{"name": "Invesco QQQ Trust", "ticker": "QQQ", "type": "지수 ETF", "score": 65, "p_reason": "20일선 지지", "reason": "빅테크 실적 견인", "strategy": "분할 매수"}],
            "continuation_stocks": [{"name": "NVIDIA", "ticker": "NVDA", "score": 80, "p_reason": "AI 반도체 수요 견고", "reason": "공급 우위", "strategy": "눌림목 매수"}]
        }

    prompt = f"""
너는 글로벌 매크로 헤지펀드 전략가이다. 아래 실시간 뉴스들을 분석하라.
[뉴스]
{news_text}

[규칙]
1. 하락장 우세 시 반드시 인버스 ETF(SQQQ, SH 등)를 추천할 것.
2. 원자재(금 GLD, 은 SLV, 원유 USO) 및 테마 ETF(SOXX, XLE 등)를 유연하게 배정할 것.
3. 각 자산/종목별로 -100 ~ +100점의 [score]와 [p_reason](손익비/가격 매력도)를 작성할 것.

[출력 양식 - 반드시 아래 규격의 유효한 JSON만 반환하라. 마크다운 따옴표(```json) 없이 순수 JSON 텍스트만 출력하라]
{{
  "verdict": "강한 상승 🟢 또는 완만한 상승 🟢 또는 중립 🟡 또는 하락 조정 🔴 또는 시장 급락 위험 🔴",
  "score": 0,
  "core_drivers": ["동인1", "동인2", "동인3"],
  "asset_hedges": [
    {{"name": "상품명", "ticker": "티커", "type": "자산분류", "score": 75, "p_reason": "가격 매력도 1줄", "reason": "근거 1줄", "strategy": "전략 1줄"}}
  ],
  "continuation_stocks": [
    {{"name": "기업명", "ticker": "티커", "score": 80, "p_reason": "가격 매력도 1줄", "reason": "근거 1줄", "strategy": "전략 1줄"}}
  ]
}}
"""
    try:
        raw_res = llm_mgr.generate_completion(prompt, temperature=0.2, max_tokens=1500)
        # 안전한 JSON 정규식 추출기 적용
        m_json = re.search(r'(\{[\s\S]*\})', raw_res)
        data = json.loads(m_json.group(1) if m_json else raw_res)
        save_ai_cache(cache_key, data)
        return data
    except Exception as e:
        print(f"⚠️ SaveTicker AI 파싱 예외 ({e}) -> 안전 폴백 데이터 적용")
        return {
            "verdict": "중립 🟡", "score": 5, "core_drivers": ["시장 거시 지표 관망 및 테마별 차별화"],
            "asset_hedges": [
                {"name": "Invesco QQQ Trust", "ticker": "QQQ", "type": "나스닥 100", "score": 65, "p_reason": "주요 지지선 안착", "reason": "빅테크 하방 경직성", "strategy": "20일선 분할 매수"},
                {"name": "SPDR Gold Shares", "ticker": "GLD", "type": "금 안전자산", "score": 75, "p_reason": "안전자산 수요 유효", "reason": "지정학 리스크 헷지", "strategy": "박스권 하단 분할 매수"}
            ],
            "continuation_stocks": [
                {"name": "NVIDIA", "ticker": "NVDA", "score": 82, "p_reason": "눌림목 지지선 형성", "reason": "AI 인프라 수주 지속", "strategy": "눌림목 분할 진입"}
            ]
        }

saveticker_intel = analyze_saveticker_macro_and_assets(news_digest_text, force_refresh=us_emergency)

asset_cards_html = ""
for item in saveticker_intel.get("asset_hedges", [])[:4]:
    sym = item.get("ticker", "").upper()
    try:
        df_h = yf.Ticker(sym).history(period="1mo", interval="1d")
        last_p = f"${df_h['Close'].iloc[-1]:.2f}" if df_h is not None and len(df_h)>0 else "-"
    except Exception:
        last_p = "-"
    sc_val = item.get('score', 70)
    sc_col = "#38bdf8" if sc_val >= 70 else ("#4ade80" if sc_val >= 30 else "#facc15")

    asset_cards_html += f"""
    <div class="card">
        <div class="console-report">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div class="report-header">{item.get('name')} ({sym}) - <span style="color:#a855f7;">{item.get('type')}</span></div>
                <a href="[https://www.tradingview.com/symbols/](https://www.tradingview.com/symbols/){sym}/" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
            </div>
            <div style="background:#111827; border:1px solid #374151; border-radius:8px; padding:10px 14px; margin:10px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-size:13px; color:#94a3b8; font-weight:bold;">🎯 뉴스 헷지 파워 스코어</span>
                    <span style="font-size:15px; font-weight:bold; color:{sc_col};">{sc_val:+d}점</span>
                </div>
                <div style="font-size:12px; color:#cbd5e1; line-height:1.6; border-top:1px dashed #374151; padding-top:6px;">
                    • <b>가격 매력도:</b> {item.get('p_reason', '지지선 형성 완료')}<br>
                    • <b>재료 지속성:</b> {item.get('reason')}
                </div>
            </div>
            <div class="report-divider"></div>
            <div class="report-line">• 현재가: <span class="highlight-val">{last_p}</span> &nbsp;|&nbsp; 🎯 대응 전략: {item.get('strategy')}</div>
        </div>
    </div>
    """

stock_picks_html = ""
for item in saveticker_intel.get("continuation_stocks", [])[:4]:
    sym = item.get("ticker", "").upper()
    try:
        df_h = yf.Ticker(sym).history(period="1mo", interval="1d")
        last_p = f"${df_h['Close'].iloc[-1]:.2f}" if df_h is not None and len(df_h)>0 else "-"
    except Exception:
        last_p = "-"
    sc_val = item.get('score', 80)
    sc_col = "#38bdf8" if sc_val >= 70 else ("#4ade80" if sc_val >= 30 else "#facc15")

    stock_picks_html += f"""
    <div class="card">
        <div class="console-report">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div class="report-header">{item.get('name')} ({sym})</div>
                <a href="[https://www.tradingview.com/symbols/](https://www.tradingview.com/symbols/){sym}/" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
            </div>
            <div style="background:#111827; border:1px solid #374151; border-radius:8px; padding:10px 14px; margin:10px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-size:13px; color:#94a3b8; font-weight:bold;">🎯 뉴스 모멘텀 스코어</span>
                    <span style="font-size:15px; font-weight:bold; color:{sc_col};">{sc_val:+d}점</span>
                </div>
                <div style="font-size:12px; color:#cbd5e1; line-height:1.6; border-top:1px dashed #374151; padding-top:6px;">
                    • <b>가격 매력도:</b> {item.get('p_reason', '눌림목 지지선 메리트')}<br>
                    • <b>재료 지속성:</b> {item.get('reason')}
                </div>
            </div>
            <div class="report-divider"></div>
            <div class="report-line">• 현재가: <span class="highlight-val">{last_p}</span> &nbsp;|&nbsp; 🎯 진입 전략: {item.get('strategy')}</div>
        </div>
    </div>
    """

news_feed_html = "".join([f'<div style="margin-bottom:10px; border-bottom:1px dashed #334155; padding-bottom:6px;"><span style="color:#38bdf8; font-size:12px;">[{n["time"]}]</span> <b>{n["title"]}</b><div style="color:#94a3b8; font-size:12.5px;">{n["desc"]}</div></div>' for n in saveticker_news_list])
drivers_html = "<br>".join([f"&nbsp;&nbsp;• {d}" for d in saveticker_intel.get("core_drivers", [])])

# =========================================================
# PART 5: HTML 공통 스타일 및 페이지 빌드
# =========================================================
html_style = """
<style>
    * { box-sizing: border-box; }
    body { font-family: 'Consolas', -apple-system, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
    .container { max-width: 950px; margin: 0 auto; }
    .nav-bar { display: flex; justify-content: center; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
    .nav-btn { padding: 8px 14px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 13.5px; }
    .btn-active { background: #2563eb; color: #ffffff; }
    .btn-inactive { background: #334155; color: #94a3b8; }
    .header { background: #1e293b; color: #38bdf8; padding: 18px; border-radius: 12px; margin-bottom: 20px; text-align: center; border: 1px solid #334155; }
    .event-banner { background: #3b0764; border: 1px solid #a855f7; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px; font-size: 13.5px; line-height: 1.7; color: #f3e8ff; }
    .event-banner-title { font-weight: bold; color: #facc15; font-size: 15px; margin-bottom: 6px; }
    .macro-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 20px; }
    .macro-card { background: #182232; border: 1px solid #334155; border-radius: 10px; padding: 14px 12px; text-align: center; text-decoration: none; color: inherit; display: block; }
    .macro-title { font-size: 13px; color: #94a3b8; font-weight: bold; margin-bottom: 6px; }
    .macro-value { font-size: 19px; font-weight: bold; color: #38bdf8; margin-bottom: 4px; }
    .macro-sub { font-size: 11.5px; color: #4ade80; margin-top: 2px; }
    .news-briefing-card { background: #182232; border: 1px solid #38bdf8; border-radius: 12px; padding: 16px; margin-bottom: 24px; line-height: 1.75; font-size: 14px; }
    .news-title { font-size: 15.5px; font-weight: bold; color: #38bdf8; margin-bottom: 10px; border-bottom: 1px dashed #334155; padding-bottom: 6px; }
    .card { background: #1e293b; padding: 18px; border-radius: 12px; margin-bottom: 26px; border: 1px solid #334155; }
    .console-report { background: #090d16; padding: 16px; border-radius: 8px; border: 1px solid #334155; font-size: 14.5px; line-height: 1.7; }
    .stock-reason-box { background: #1e1b4b; border-left: 4px solid #818cf8; padding: 11px; border-radius: 4px; margin: 10px 0; font-size: 13.5px; color: #e0e7ff; line-height: 1.6; }
    .report-header { font-size: 17px; font-weight: bold; color: #38bdf8; }
    .report-divider { border-top: 1px dashed #475569; margin: 10px 0; }
    .report-line { margin: 4px 0; }
    .highlight-val { color: #facc15; font-weight: bold; }
    .text-red { color: #f87171; font-weight: bold; }
    .text-green { color: #4ade80; font-weight: bold; }
    .tv-link-btn { background: #2563eb; color: #ffffff; padding: 4px 9px; border-radius: 4px; text-decoration: none; font-size: 12px; font-weight: bold; }
    .sub-desc { font-size: 12px; color: #94a3b8; }
    .badge-item { color: #ffffff; font-size: 11.5px; padding: 2px 7px; border-radius: 12px; margin-left: 6px; font-weight: bold; display: inline-block; }
    .ai-opinion-box { background: #062a1c; border: 1px solid #22c55e; border-radius: 8px; padding: 16px; margin-top: 14px; }
    .ai-title { font-size: 14.5px; font-weight: bold; color: #4ade80; margin-bottom: 8px; }
    .ai-content { font-size: 14px; color: #f1f5f9; line-height: 1.7; }
    .deep-report-accordion { margin-top: 12px; border-top: 1px dashed rgba(34, 197, 94, 0.4); padding-top: 10px; }
    .deep-report-btn { background: #0d4a32; color: #86efac; padding: 9px 12px; border-radius: 6px; font-size: 13px; font-weight: bold; cursor: pointer; border: 1px solid #15803d; list-style: none; text-align: center; }
    .deep-report-content { background: #041f15; border-radius: 6px; padding: 13px; margin-top: 8px; font-size: 13.5px; color: #e2e8f0; line-height: 1.75; border: 1px solid #166534; }
    .chart-container { margin-top: 16px; border-radius: 8px; overflow: hidden; }
    #btn-back-to-top { position: fixed; bottom: 25px; right: 25px; display: none; width: 46px; height: 46px; border-radius: 50%; background: #2563eb; color: #fff; border: none; cursor: pointer; font-size: 20px; font-weight: bold; z-index: 9999; align-items: center; justify-content: center; }
    @media (max-width: 768px) { .macro-grid { grid-template-columns: 1fr; } .nav-btn { flex: 1 1 45%; text-align: center; } }
</style>
"""

top_button_component = """
<button id="btn-back-to-top" title="최상단으로">▲</button>
<script>
    const topBtn = document.getElementById("btn-back-to-top");
    window.addEventListener("scroll", () => { topBtn.style.display = (window.scrollY > 300) ? "flex" : "none"; });
    topBtn.addEventListener("click", () => { window.scrollTo({ top: 0, behavior: "smooth" }); });
</script>
"""

# ✅ 변수 바인딩 오타 전면 제거 및 안전 치환
macro_html_kr = f"""
<div class="macro-grid">
    <a href="{kr_macro['m2_url']}" target="_blank" class="macro-card">
        <div class="macro-title">💵 원화 통화량 (M2) ↗</div>
        <div class="macro-value">{kr_macro['m2']}</div>
        <div class="macro-sub">{kr_macro['m2_date']}</div>
    </a>
    <a href="{kr_macro['cli_url']}" target="_blank" class="macro-card">
        <div class="macro-title">🌐 한국 경기선행지수 (CLI) ↗</div>
        <div class="macro-value">{kr_macro['cli']}</div>
        <div class="macro-sub">{kr_macro['cli_date']}</div>
    </a>
    <a href="{kr_macro['vix_url']}" target="_blank" class="macro-card">
        <div class="macro-title">⚡ 한국 VKOSPI ↗</div>
        <div class="macro-value" style="color:#facc15;">{kr_macro['vix']}</div>
        <div class="macro-sub">네이버 금융 원본 연동</div>
    </a>
</div>
"""

macro_html_us = f"""
<div class="macro-grid">
    <a href="{us_macro['m2_url']}" target="_blank" class="macro-card">
        <div class="macro-title">💵 달러 통화량 (US M2) ↗</div>
        <div class="macro-value">{us_macro['m2']}</div>
        <div class="macro-sub">{us_macro['m2_date']}</div>
    </a>
    <a href="{us_macro['cli_url']}" target="_blank" class="macro-card">
        <div class="macro-title">🌐 미국 경기선행지수 (CLI) ↗</div>
        <div class="macro-value">{us_macro['cli']}</div>
        <div class="macro-sub">{us_macro['cli_date']}</div>
    </a>
    <a href="{us_macro['vix_url']}" target="_blank" class="macro-card">
        <div class="macro-title">⚡ 미국 VIX 지수 ↗</div>
        <div class="macro-value" style="color:#facc15;">{us_macro['vix']}</div>
        <div class="macro-sub">TradingView 원본 연동</div>
    </a>
</div>
"""

full_html_kr = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>🇰🇷 AI 국장 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-active">🇰🇷 국장</a><a href="us_index.html" class="nav-btn btn-inactive">🇺🇸 미장</a><a href="index3.html" class="nav-btn btn-inactive">🎯 마이</a><a href="index4.html" class="nav-btn btn-inactive">📰 뉴스</a></div><div class="header"><h1>📊 AI 국장 주도주 대시보드 [{kr_market_status}]</h1><p style="margin:0; color:#94a3b8; font-size:13px;">{kr_open_msg} | {now_str}</p></div>{kr_banner_html}{macro_html_kr}<div class="news-briefing-card"><div class="news-title">📰 [최근 7일간 뉴스 AI 종합 분석] ({kr_briefing_time})</div>{kr_sentiment_briefing}</div>{stock_cards_kr_html}</div>{top_button_component}</body></html>"""

full_html_us = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>🇺🇸 AI 미장 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-inactive">🇰🇷 국장</a><a href="us_index.html" class="nav-btn btn-active">🇺🇸 미장</a><a href="index3.html" class="nav-btn btn-inactive">🎯 마이</a><a href="index4.html" class="nav-btn btn-inactive">📰 뉴스</a></div><div class="header"><h1>🇺🇸 AI US Stock 주도주 대시보드 [{us_market_status}]</h1><p style="margin:0; color:#94a3b8; font-size:13px;">{us_open_msg} | {now_str}</p></div>{us_banner_html}{macro_html_us}<div class="news-briefing-card"><div class="news-title">📰 [최근 7일간 뉴스 AI 종합 분석] ({us_briefing_time})</div>{us_sentiment_briefing}</div>{stock_cards_us_html}</div>{top_button_component}</body></html>"""

full_html_my = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>🎯 마이 포트폴리오</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-inactive">🇰🇷 국장</a><a href="us_index.html" class="nav-btn btn-inactive">🇺🇸 미장</a><a href="index3.html" class="nav-btn btn-active">🎯 마이</a><a href="index4.html" class="nav-btn btn-inactive">📰 뉴스</a></div><div class="header"><h1>🎯 마이 포트폴리오 실계좌 대시보드</h1><p style="margin:0; color:#94a3b8; font-size:13px;">{now_str}</p></div><div style="background:#1e293b; padding:16px; border-radius:12px; margin-bottom:22px; display:flex; justify-content:space-around; text-align:center; border:1px solid #334155;"><div><div style="font-size:12px; color:#94a3b8;">총 평가 금액</div><div style="font-size:22px; font-weight:bold;">{fmt_price(total_eval_my, True)}</div></div><div><div style="font-size:12px; color:#94a3b8;">총 손익</div><div style="font-size:22px; font-weight:bold; color:{'#4ade80' if total_profit_my>=0 else '#f87171'};">{total_profit_my:+,.0f}원</div></div><div><div style="font-size:12px; color:#94a3b8;">수익률</div><div style="font-size:22px; font-weight:bold; color:{'#4ade80' if total_ret_pct>=0 else '#f87171'};">{total_ret_pct:+.2f}%</div></div></div>{my_stock_cards_html}</div>{top_button_component}</body></html>"""

full_html_news = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>📰 SaveTicker 뉴스 인텔리전스</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-inactive">🇰🇷 국장</a><a href="us_index.html" class="nav-btn btn-inactive">🇺🇸 미장</a><a href="index3.html" class="nav-btn btn-inactive">🎯 마이</a><a href="index4.html" class="nav-btn btn-active">📰 뉴스</a></div><div class="header"><h1>📰 SaveTicker 뉴스 인텔리전스 대시보드</h1><p style="margin:0; color:#94a3b8; font-size:13px;">{now_str}</p></div><div class="card"><div class="console-report"><div class="report-header">🧭 글로벌 거시 시장 나침반 : <span class="highlight-val">{saveticker_intel.get('verdict','중립')}</span> ({saveticker_intel.get('score', 0):+d}점)</div><div class="report-divider"></div><div class="report-line"><b>📌 시장 핵심 동인:</b><br>{drivers_html}</div></div></div><h2 style="font-size:1.15rem; color:#38bdf8; margin-bottom:12px;">🛡️ 시장 대응 자산 (지수 / 인버스 / 원자재 ETF)</h2>{asset_cards_html}<h2 style="font-size:1.15rem; color:#4ade80; margin-bottom:12px;">🚀 뉴스 모멘텀 유망주 (시간차 방어)</h2>{stock_picks_html}<details class="deep-report-accordion" style="margin-top:20px;"><summary class="deep-report-btn">📋 SaveTicker 원문 피드 ({len(saveticker_news_list)}건) ▼</summary><div class="deep-report-content">{news_feed_html}</div></details></div>{top_button_component}</body></html>"""

# =========================================================
# PART 6: GitHub 배포 실행
# =========================================================
def upload_to_github_safely(repo, file_path, commit_message, content):
    try:
        file_obj = repo.get_contents(file_path)
        repo.update_file(path=file_path, message=commit_message, content=content, sha=file_obj.sha)
        print(f"✅ {file_path} 업데이트 성공")
    except UnknownObjectException:
        repo.create_file(path=file_path, message=commit_message, content=content)
        print(f"✅ {file_path} 신규 생성 성공")
    except Exception as e:
        print(f"🚨 {file_path} 배포 실패: {e}")

print("\n🌐 [PART 6] GitHub 배포 중...")
try:
    if not GITHUB_TOKEN: raise ValueError("GH_TOKEN 누락")
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO_NAME)
    
    upload_to_github_safely(repo, "index.html", f"Deploy KR: {now_str}", full_html_kr)
    upload_to_github_safely(repo, "us_index.html", f"Deploy US: {now_str}", full_html_us)
    upload_to_github_safely(repo, "index3.html", f"Deploy My: {now_str}", full_html_my)
    upload_to_github_safely(repo, "index4.html", f"Deploy News: {now_str}", full_html_news)
    
    if os.path.exists(CACHE_FILE_NAME):
        with open(CACHE_FILE_NAME, "r", encoding="utf-8") as f:
            c_json = f.read()
        upload_to_github_safely(repo, "ai_cache.json", f"Update Cache: {now_str}", c_json)

    print("\n🎉 [최종 완료] 4개 대시보드가 정상적으로 배포되었습니다.")
except Exception as e:
    print(f"🚨 GitHub 연결 실패: {e}")
