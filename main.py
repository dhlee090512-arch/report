# 1. 필수 라이브러리 설치 (Colab / 로컬 전용 - GitHub Actions 사용 시 주석 처리)
#!pip install groq yfinance "pandas==2.2.2" beautifulsoup4 plotly requests PyGithub -q

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
import xml.etree.ElementTree as ET
import datetime
import time
import re
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# ⚙️ [테스트 모드 설정 - 기본 TEST_MODE = True 고정]
# True : Groq AI 및 네이버 뉴스 크롤링 스킵 (Groq 쿼터 100% 보존)
#        ★ 토스증권 API 연동은 TEST_MODE와 상관없이 항상 실제 데이터를 받아옵니다.
# False: 실제 뉴스 크롤링, 주간 테마/감성 분석, 토스증권 API, Groq AI 실시간 종목 분석 전체 가동
# =========================================================
TEST_MODE = False

# =========================================================
# [보안 및 Secrets / 환경변수 자동 로드]
# =========================================================
try:
    from google.colab import userdata
    GROQ_API_KEY_1 = userdata.get('GROQ_API_KEY')
    GROQ_API_KEY_2 = userdata.get('GROQ_API_KEY2')
    GITHUB_TOKEN = userdata.get('GH_TOKEN')
    TOSS_CLIENT_ID = userdata.get('TOSS_CLIENT_ID')
    TOSS_CLIENT_SECRET = userdata.get('TOSS_CLIENT_SECRET')
    FIXIE_URL = userdata.get('FIXIE_URL')
except ImportError:
    GROQ_API_KEY_1 = os.environ.get("GROQ_API_KEY", "")
    GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY2", "")
    GITHUB_TOKEN = os.environ.get("GH_TOKEN", "")
    TOSS_CLIENT_ID = os.environ.get("TOSS_CLIENT_ID", "")
    TOSS_CLIENT_SECRET = os.environ.get("TOSS_CLIENT_SECRET", "")
    FIXIE_URL = os.environ.get("FIXIE_URL", "")

GITHUB_REPO_NAME = os.environ.get("GITHUB_REPOSITORY", "dhlee090512-arch/report")
CACHE_FILE_NAME = "ai_cache.json"

# Webshare 고정 IP 프록시 연동
DEFAULT_PROXY_URL = "http://rhjkraof:8k6vhgbj4i2h@142.111.67.146:5611"
PROXY_URL = FIXIE_URL if FIXIE_URL else DEFAULT_PROXY_URL
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
# [멀티 GROQ API Key 관리 클래스]
# =========================================================
class GroqKeyManager:
    def __init__(self, key1, key2):
        self.keys = [k.strip() for k in [key1, key2] if k and k.strip()]
        self.current_index = 0
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.keys and self.current_index < len(self.keys):
            try:
                self.client = Groq(api_key=self.keys[self.current_index])
                print(f"✅ Groq AI 클라이언트 초기화 성공! (현재 {self.current_index + 1}번 API Key 사용 중)")
            except Exception as e:
                print(f"⚠️ {self.current_index + 1}번 키 초기화 실패: {e}")
                self.client = None
        else:
            self.client = None

    def switch_to_next_key(self):
        self.current_index += 1
        if self.current_index < len(self.keys):
            print(f"🔄 429 한도 초과 감지 ➔ {self.current_index + 1}번 Groq API Key로 자동 전환합니다...")
            self._init_client()
            return True
        else:
            print("🚨 등록된 모든 Groq API Key의 일일 한도가 초과되었습니다.")
            self.client = None
            return False

    def is_available(self):
        return self.client is not None and not TEST_MODE

groq_mgr = GroqKeyManager(GROQ_API_KEY_1, GROQ_API_KEY_2)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Chrome/120.0.0.0)',
    'Referer': 'https://finance.naver.com/'
}

kst_timezone = datetime.timezone(datetime.timedelta(hours=9))
now_dt = datetime.datetime.now(kst_timezone)
now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")

# =========================================================
# 💾 AI 캐시 매니저 (JSON 파일 읽기/쓰기 함수)
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
    if groq_mgr.current_index >= len(groq_mgr.keys):
        return "모든 Groq API Key (1번 & 2번) 일일 호출 한도 초과(Rate Limit Exceeded)"
    if not groq_mgr.keys:
        return "Groq API Key 미설정"
    return "AI 통신 응답 지연 및 일시적 서버 오류"

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
                
                # 수치 + 단위 + 수치 바로 뒤의 기준 월(in Month [of Year])을 정밀하게 한 번에 포착
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
        if val_m2 > 1000000:
            m2_trillion = val_m2 / 1000
        elif val_m2 > 1000:
            m2_trillion = val_m2 / 1000 if "BILLION" in unit_m2 else val_m2
        else:
            m2_trillion = val_m2
        m2_val_str = f"{fmt_num(m2_trillion)}조 원"
    else:
        m2_val_str = "4,210.5조 원"
    
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
    else:
        m2_val_str = "$21.40 Trillion"
        
    m2_date_str = date_m2 if date_m2 else "최신 발표 기준 🟢"

    val_cli, unit_cli, date_cli = parse_te_summary_val(US_CLI_URL)
    cli_val_str = f"{fmt_num(val_cli)} Pts" if val_cli else "102.10 Pts"
    cli_date_str = date_cli if date_cli else "최신 발표 기준 🟢"

    try:
        vix_tk = yf.Ticker("^VIX").history(period="5d")
        val_vix = float(vix_tk['Close'].iloc[-1]) if (vix_tk is not None and len(vix_tk) > 0) else None
    except Exception:
        val_vix = None
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

    if len(titles) < 10:
        try:
            url = "https://finance.yahoo.com/news/"
            res = requests.get(url, headers=headers, timeout=7)
            soup = BeautifulSoup(res.text, 'html.parser')
            h3_tags = [h3.text.strip() for h3 in soup.find_all(['h3', 'h2']) if len(h3.text.strip()) > 10]
            titles.extend(h3_tags)
        except Exception: pass

    titles = list(dict.fromkeys(titles))
    return "\n".join(titles[:40]) if titles else "Fed Rate Policy | Tech Earnings Rally | AI Demand Momentum"

def sanitize_text(text):
    cleaned = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]', '', text)
    return cleaned.strip()

def analyze_7days_news_sentiment(market_type, news_text):
    cache_key = f"MARKET_{market_type}"

    if not groq_mgr.is_available():
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
    
    [분석 요구사항]
    1. 시장 전체 분위기를 진단하여 종합 상태 [긍정 🟢 OR 보통 🟡 OR 부정 🔴] 중 하나를 선택하라.
    2. 뉴스 헤드라인에서 확인되는 **🟢 긍정 호재 요소(최대 5개)**를 추출하여 각각 한 줄로 요약하라.
    3. 뉴스 헤드라인에서 확인되는 **🔴 부정 악재/리스크 요소(최대 5개)**를 추출하여 각각 한 줄로 요약하라.
    4. 현재 시장을 강하게 이끌고 있는 **🚀 강세/주도 테마(주요 업종 및 섹터)**를 작성하라.
    5. 7일 누적 뉴스 감성 지수(-100점 ~ +100점)를 산출하라.
    
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
        res = groq_mgr.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "너는 한자와 일본어를 절대 사용하지 않고 오직 한글과 영문으로만 뉴스를 분석하는 금융 분석가이다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, max_tokens=800
        )
        content = res.choices[0].message.content.strip()
        
        status_val = "보통 🟡"
        status_match = re.search(r'상태:\s*(.*)', content)
        if status_match: 
            status_val = status_match.group(1).strip()

        extracted_pos = []
        for i in range(1, 6):
            m = re.search(rf'긍정{i}:\s*(.*)', content)
            if m:
                text = m.group(1).strip()
                if text and not text.startswith('<') and len(text) > 2:
                    extracted_pos.append(text)

        extracted_neg = []
        for i in range(1, 6):
            m = re.search(rf'부정{i}:\s*(.*)', content)
            if m:
                text = m.group(1).strip()
                if text and not text.startswith('<') and len(text) > 2:
                    extracted_neg.append(text)

        theme_val = "특이 테마 미포착"
        score_val = "0점 (중립)"
        
        theme_match = re.search(r'강세테마:\s*(.*)', content)
        score_match = re.search(r'감성지수:\s*(.*)', content)
        if theme_match and len(theme_match.group(1).strip()) > 1: 
            theme_val = theme_match.group(1).strip()
        if score_match and len(score_match.group(1).strip()) > 1: 
            score_val = score_match.group(1).strip()

        pos_html = "<br>".join([f"&nbsp;&nbsp;• {sanitize_text(item)}" for item in extracted_pos]) if extracted_pos else "&nbsp;&nbsp;• 특이 긍정 호재 미포착"
        neg_html = "<br>".join([f"&nbsp;&nbsp;• {sanitize_text(item)}" for item in extracted_neg]) if extracted_neg else "&nbsp;&nbsp;• 특이 부정 리스크 미포착"

        raw_briefing_html = f"""
        🟢 <b>지난 7일 긍정 호재 ({len(extracted_pos)}개):</b><br>{pos_html}<br><br>
        🔴 <b>지난 7일 부정 리스크 ({len(extracted_neg)}개):</b><br>{neg_html}<br><br>
        🚀 <b>시장 주도/강세 테마:</b> <span style="color:#38bdf8; font-weight:bold;">{sanitize_text(theme_val)}</span><br>
        📊 <b>7일 누적 뉴스 감성 지수:</b> <span class="highlight-val">{sanitize_text(score_val)}</span>
        """

        save_ai_cache(cache_key, {
            "status": sanitize_text(status_val),
            "briefing_html": raw_briefing_html
        })

        return sanitize_text(status_val), raw_briefing_html

    except RateLimitError:
        if groq_mgr.switch_to_next_key():
            return analyze_7days_news_sentiment(market_type, news_text)
        else:
            return analyze_7days_news_sentiment(market_type, news_text)
    except Exception as e:
        return "보통 🟡", f"뉴스 분석 생성 안내: {e}"

# =========================================================
# 🛠️ 타이트한 손절가 기반 손익비(Risk-Reward) 우위 가격 산정 유틸리티
# =========================================================
def calculate_advanced_tech_levels(latest_close, ma20, kijun, bb_low, bb_up, cloud_top, poc_price, atr, is_krw=True):
    supports = [val for val in [ma20, kijun, bb_low, poc_price] if val < latest_close and val > 0]
    if supports:
        primary_support = max(supports)
        stop_loss = primary_support - (0.3 * atr) # 리스크 폭 최소화
    else:
        stop_loss = latest_close - (1.0 * atr)

    # 1차 익절가는 최소 손절폭 대비 1.5배 이상 구간 설정
    risk_range = latest_close - stop_loss
    min_target = latest_close + (risk_range * 1.5)

    resistances = [val for val in [bb_up, cloud_top, poc_price] if val > min_target]
    if resistances:
        target_price = min(resistances)
    else:
        target_price = max(min_target, latest_close + (1.2 * atr))

    if is_krw:
        stop_loss = int(round(stop_loss))
        target_price = int(round(target_price))
    else:
        stop_loss = round(stop_loss, 2)
        target_price = round(target_price, 2)

    return stop_loss, target_price

# =========================================================
# 🤖 일반 종목 AI 정밀 리포트 (손익비 1:1.5 이상 강제 전략)
# =========================================================
def generate_ai_stock_analysis(stock_name, symbol, news_keywords, raw_data_str, rsi_val, macd_status, ma_status, bb_status, cloud_status, poc_price, target_price, stop_loss, supply_type="", currency_symbol="원"):
    cache_key = f"STOCK_{symbol}"
    is_krw = True if currency_symbol in ["원", "KRW"] else False

    if not groq_mgr.is_available():
        if cache_key in ai_cache_store:
            cached = ai_cache_store[cache_key]
            updated_at = cached.get('updated_at', '일자 미상')
            reason_msg = get_fallback_reason()
            reason_str = f"⚠️ [비실시간 백업] {cached['reason']}"
            report_str = f"⚠️ [비실시간 백업 리포트 - 생성일: {updated_at} | 📌 사유: {reason_msg}]\n" + cached['report']
            return reason_str, report_str
        else:
            return f"수급/모멘텀 모니터링 포착 종목", "AI 상세 전략 리포트 준비 중입니다."

    prompt = f"""
    너는 20년 경력의 수석 기술적 분석 및 트레이딩 전략가이다. {stock_name}({symbol}) 종목에 대해 손익비(Risk-Reward Ratio) 최소 1:1.5 이상이 확보되는 현실적이고 안전한 매매 전략 리포트를 작성하라.

    [종목 기본 & 기술 데이터]
    - 수급/테마 특징: {supply_type}
    - 최근 시장 뉴스 이슈/호재:
    {news_keywords}
    - 최근 10일 OHLCV 원본 데이터:
    {raw_data_str}
    - 보조지표: RSI({rsi_val}), MACD({macd_status}), 이평선 배열({ma_status})
    - 차트 구조: 볼린저 밴드({bb_status}), 일목 구름대({cloud_status}), 매물대 POC({fmt_price(poc_price, is_krw)})
    - 복합 기술적 참조가: 1차 목표가 {fmt_price(target_price, is_krw)} / 손절가 {fmt_price(stop_loss, is_krw)}

    [작성 요구사항 & 출력 양식]
    아래 양식을 엄격히 준수하여 작성하라. 손절폭 대비 익절폭이 최소 1:1.5 이상 확보되도록 눌림목 매수 추천가, 손절가, 목표가를 정확하게 제시할 것.

    선정이유: <외인/기관 수급 동향, 최근 뉴스 이슈/호재, 주도 테마/섹터의 강세 배경, 그리고 차트의 기술적 모멘텀을 종합하여 2~3줄로 입체적으로 요약 작성할 것>
    상세리포트:
    📌 [차트 구조 & 지표 종합 진단]
    - <이동평균선 배열, 볼린저밴드 위치, 일목 구름대 상태, 매물대(POC), RSI/MACD 보조지표 종합 진단 2~3줄>

    🟢 [안전 매수 & 리스크 관리 전략 (손익비 타겟 1:1.5 이상)]
    - 눌림목 매수 추천가 : <가격: 000{currency_symbol}> / 최종 하단 지지선 : <가격: 000{currency_symbol}> / 타이트 손절가 : <가격: 000{currency_symbol}>
    - <지지가 확인되는 눌림목 매수 타점 근거 작성>. <주요 지지선 직하단으로 손절가를 타이트하게 설정해 리스크 폭을 최소화한 이유 서술>.

    🚀 [현실적 분할 익절 전략]
    - 1차 안전 익절가 : <가격: 000{currency_symbol}> (손익비 1:1.5 이상 달성 지점 / 물량 50% 익절)
    - 2차 추세 익절가 : <가격: 000{currency_symbol}> (손익비 1:2.5~3.0 및 전고점 저항 지점 / 잔량 50% 추세 대응)
    - <1차 목표가와 2차 목표가까지 상승 가능한 지표적 근거 및 손익비 우위 관점 기술>.

    [언어 제한] 한자(漢字) 및 일본어 절대 금지. 오직 순수 한글, 영문, 숫자만 사용할 것.
    """
    try:
        res = groq_mgr.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1000
        )
        content = res.choices[0].message.content.strip()
        
        reason_val = f"{supply_type} 모멘텀과 기술적 지지선 반등이 강화되는 종목입니다."
        report_val = content

        reason_match = re.search(r'선정이유:\s*(.*)', content)
        report_match = re.search(r'상세리포트:\s*([\s\S]*)', content)

        if reason_match: reason_val = reason_match.group(1).strip()
        if report_match: report_val = report_match.group(1).strip()

        save_ai_cache(cache_key, {
            "reason": sanitize_text(reason_val),
            "report": sanitize_text(report_val)
        })

        return sanitize_text(reason_val), sanitize_text(report_val)

    except RateLimitError:
        if groq_mgr.switch_to_next_key():
            return generate_ai_stock_analysis(stock_name, symbol, news_keywords, raw_data_str, rsi_val, macd_status, ma_status, bb_status, cloud_status, poc_price, target_price, stop_loss, supply_type, currency_symbol)
        else:
            return generate_ai_stock_analysis(stock_name, symbol, news_keywords, raw_data_str, rsi_val, macd_status, ma_status, bb_status, cloud_status, poc_price, target_price, stop_loss, supply_type, currency_symbol)
    except Exception as e:
        return "수급 유입 및 기술적 지지 종목", f"상세 전략 리포트 생성 안내: {e}"

# =========================================================
# 🎯 [토스증권 마이 대시보드 전용] AI 냉정한 수익 확정(익절) 및 리스크 관리 프롬프트
# =========================================================
def generate_ai_toss_3line_analysis(stock_name, symbol, avg_price, current_price, return_pct, raw_data_str, rsi_val, macd_status, ma_status, bb_status, cloud_status, poc_price, target_price, stop_loss, is_krw=True):
    cache_key = f"TOSS_MY_{symbol}"

    if not groq_mgr.is_available():
        return "[테스트 모드] Groq AI 연동 미사용 상태입니다."

    prompt = f"""
    너는 20년 경력의 냉정한 포트폴리오 리스크 관리 전문가이다. 
    사용자의 [내 보유 평단가, 현재 수익률({return_pct:+.2f}%)]과 [차트 기술적 지표]를 바탕으로, 감정을 배제하고 익절 및 손절 타이밍을 명확히 판단하라.

    [보유 종목 & 차트 데이터]
    - 종목명: {stock_name} ({symbol})
    - 내 보유 평단가: {fmt_price(avg_price, is_krw, show_decimal=is_krw)} (현재 수익률: {return_pct:+.2f}%)
    - 현재가: {fmt_price(current_price, is_krw)}
    - 보조지표: RSI({rsi_val}), MACD({macd_status}), 이평선 배열({ma_status})
    - 차트 구조: 볼린저 밴드({bb_status}), 일목 구름대({cloud_status}), 매물대 POC({fmt_price(poc_price, is_krw)})
    - 참조 가격: 1차 목표가 {fmt_price(target_price, is_krw)} / 손절가 {fmt_price(stop_loss, is_krw)}

    [판단 원칙 - 엄격 준수]
    1. **수익 구간인 경우**: 차트 추세가 좋아 보여도 단기 과열(RSI 65 이상, 볼린저 상한 근접) 조짐이 있거나 상단 저항선 도달 시 무분별한 매수 시그널을 절대 내지 말고 **[결론: 일부 매도 🔴]** 또는 **[결론: 관망 후 익절 준비 🟡]**를 선택하여 수익 확정을 우선시하라.
    2. **손실 구간인 경우**: 무작정 물타기(추가 매수)를 권하지 말고, 지지선 이탈 위험 시 **[결론: 손절/비중축소 🔴]**를 명확히 권유하라.

    [출력 양식]
    결론: [매수 🟢 / 관망 🟡 / 일부 매도 🔴 / 손절 🔴] 중 하나 명시
    - <내 평단가 및 현재 수익률 대비 차트 위치(과열/안정/위험) 진단>
    - <수익 중이면 '일부 분할 매도(익절) 가격대', 손실 중이면 '손절/비중축소 기준 가격' 구체적 제시>
    - <내 평단가 위로 손절가를 높여 이미 얻은 수익을 보존하는 트레일링 스탑(Trailing Stop) 가이드 작성>
    """
    try:
        res = groq_mgr.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=400
        )
        content = res.choices[0].message.content.strip()
        save_ai_cache(cache_key, {"report": sanitize_text(content)})
        return sanitize_text(content)

    except RateLimitError:
        if groq_mgr.switch_to_next_key():
            return generate_ai_toss_3line_analysis(stock_name, symbol, avg_price, current_price, return_pct, raw_data_str, rsi_val, macd_status, ma_status, bb_status, cloud_status, poc_price, target_price, stop_loss, is_krw)
        else:
            return generate_ai_toss_3line_analysis(stock_name, symbol, avg_price, current_price, return_pct, raw_data_str, rsi_val, macd_status, ma_status, bb_status, cloud_status, poc_price, target_price, stop_loss, is_krw)
    except Exception as e:
        return "[테스트 모드] Groq AI 연동 미사용 상태입니다."

# =========================================================
# PART 1: 🇰🇷 국장(index.html) 분석 (5개 종목)
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
    try:
        res = requests.get(url, headers=headers, timeout=10).json()
        return res.get('result', {}).get('itemList', [])[:15]
    except Exception:
        return []

foreign_items = get_naver_sise_list(url_foreign)
organ_items = get_naver_sise_list(url_organ)
quant_items = get_naver_sise_list(url_quant)

foreign_names = [item.get('nm') for item in foreign_items if item.get('nm')]
organ_names = [item.get('nm') for item in organ_items if item.get('nm')]
both_names = [name for name in foreign_names if name in organ_names]

all_items_kr = {}
for item in foreign_items + organ_items + quant_items:
    if item.get('nm') and item.get('cd'):
        all_items_kr[item.get('nm')] = item.get('cd')

selected_kr_targets = {}

for name in both_names:
    if len(selected_kr_targets) >= 2: break
    code = all_items_kr.get(name)
    if code:
        selected_kr_targets[name] = (f"{code}.KS", "🔥 외인·기관 동시 순매수(양매수) 주도주")

for item in quant_items:
    if len(selected_kr_targets) >= 4: break
    name = item.get('nm')
    code = item.get('cd')
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
        
        if df_daily is None or df_daily.empty or len(df_daily) < 60:
            symbol = f"{pure_code}.KQ"
            ticker = yf.Ticker(symbol)
            df_daily = ticker.history(period="1y", interval="1d")

        if df_daily is None or df_daily.empty or len(df_daily) < 60: continue

        df_daily['MA20'] = df_daily['Close'].rolling(20).mean()
        df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
        df_daily['MA120'] = df_daily['Close'].rolling(120).mean()
        
        std20 = df_daily['Close'].rolling(20).std()
        df_daily['BB_Upper'] = df_daily['MA20'] + (std20 * 2)
        df_daily['BB_Lower'] = df_daily['MA20'] - (std20 * 2)
        
        high9 = df_daily['High'].rolling(9).max()
        low9 = df_daily['Low'].rolling(9).min()
        df_daily['Tenkan'] = (high9 + low9) / 2
        high26 = df_daily['High'].rolling(26).max()
        low26 = df_daily['Low'].rolling(26).min()
        df_daily['Kijun'] = (high26 + low26) / 2
        high52 = df_daily['High'].rolling(52).max()
        low52 = df_daily['Low'].rolling(52).min()
        senkou_b = (high52 + low52) / 2
        senkou_a = (df_daily['Tenkan'] + df_daily['Kijun']) / 2
        
        df_daily['Senkou_A'] = senkou_a.shift(26)
        df_daily['Senkou_B'] = senkou_b.shift(26)

        df_recent120 = df_daily.tail(120).copy()
        price_bins = pd.cut(df_recent120['Close'], bins=15)
        vol_by_price = df_recent120.groupby(price_bins)['Volume'].sum()
        poc_bin = vol_by_price.idxmax() if not vol_by_price.empty else None
        poc_price = int(poc_bin.mid) if poc_bin is not None and pd.notnull(poc_bin) else int(df_recent120['Close'].mean())

        df_daily = df_daily.bfill().ffill().dropna()
        if df_daily.empty or len(df_daily) == 0: continue

        delta = df_daily['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df_daily['RSI'] = 100 - (100 / (1 + (gain / loss)))
        rsi_val = round(float(np.nan_to_num(df_daily['RSI'].values[-1], nan=50.0)), 2)
        
        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_val = float(np.nan_to_num(df_daily['MACD'].values[-1])) if len(df_daily['MACD'])>0 else 0.0
        signal_val = float(np.nan_to_num(df_daily['Signal'].values[-1])) if len(df_daily['Signal'])>0 else 0.0
        
        tr = pd.concat([df_daily['High']-df_daily['Low'], np.abs(df_daily['High']-df_daily['Close'].shift()), np.abs(df_daily['Low']-df_daily['Close'].shift())], axis=1).max(axis=1)
        atr = float(np.nan_to_num(tr.rolling(14).mean().values[-1], nan=1000.0))
        
        latest_close = int(np.nan_to_num(df_daily['Close'].values[-1]))
        ma20_d = int(np.nan_to_num(df_daily['MA20'].values[-1]))
        ma60_d = int(np.nan_to_num(df_daily['MA60'].values[-1]))
        ma120_d = int(np.nan_to_num(df_daily['MA120'].values[-1]))
        bb_up = int(np.nan_to_num(df_daily['BB_Upper'].values[-1]))
        bb_low = int(np.nan_to_num(df_daily['BB_Lower'].values[-1]))
        cloud_a = int(np.nan_to_num(df_daily['Senkou_A'].values[-1]))
        cloud_b = int(np.nan_to_num(df_daily['Senkou_B'].values[-1]))
        kijun_d = int(np.nan_to_num(df_daily['Kijun'].values[-1]))
        cloud_top = max(cloud_a, cloud_b)

        stop_loss, target_price_1 = calculate_advanced_tech_levels(latest_close, ma20_d, kijun_d, bb_low, bb_up, cloud_top, poc_price, atr, is_krw=True)

        short_trend = "단기 상승 추세 📈" if latest_close >= ma20_d else "단기 하락 추세 📉"
        mid_trend = "중기 상승 추세 📈" if latest_close >= ma60_d else "중기 하락 추세 📉"
        bb_status = "상한선 돌파/근접 🚀" if latest_close >= bb_up * 0.99 else ("하한선 근접/지지 🟢" if latest_close <= bb_low * 1.01 else "밴드 내 안정 ⚖️")
        cloud_status = "구름대 위 상승 국면 🟢" if latest_close > cloud_top else "구름대 내부/하단 돌파 시도 🟡"

        df_recent10 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:{int(row['Open']):,}원 | High:{int(row['High']):,}원 | Low:{int(row['Low']):,}원 | Close:{int(row['Close']):,}원 | Vol:{int(row['Volume']):,}" for idx, row in df_recent10.iterrows()]
        raw_data_str = "\n".join(raw_lines)

        rsi_status = f"과매수 ({rsi_val}) ⚠️" if rsi_val >= 70 else (f"과매도 ({rsi_val}) 🟢" if rsi_val <= 30 else f"중립 ({rsi_val}) ⚖️")
        macd_status = "골든크로스 📈" if macd_val > signal_val else "데드크로스 📉"
        ma_status = f"정배열 (20>60>120) 🟢" if (ma20_d > ma60_d and ma60_d > ma120_d) else "역배열/혼조세 🔴"

        tradingview_url = f"https://www.tradingview.com/symbols/KRX-{pure_code}/"

        print(f"  ⚡ [국장] {stock_name} AI 리포트 처리 중...")
        pick_reason, ai_comment = generate_ai_stock_analysis(stock_name, symbol, kr_7d_news, raw_data_str, rsi_val, macd_status, ma_status, bb_status, cloud_status, poc_price, target_price_1, stop_loss, supply_type, "원")

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
        fig.add_hline(y=target_price_1, line_dash="dash", line_color="green", annotation_text=f"목표가: {fmt_price(target_price_1, True)}", row=1, col=1)
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"손절가: {fmt_price(stop_loss, True)}", row=1, col=1)
        
        colors = ['#f87171' if c < o else '#4ade80' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='거래량'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#38bdf8', width=1.2), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        
        fig.update_layout(height=540, margin=dict(l=10, r=10, t=10, b=40), xaxis_rangeslider_visible=False, template="plotly_dark")
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # 선정 이유 박스를 종목명 헤더 바로 아래 배치
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
                <div class="report-line text-red">🛑 손절가 : {fmt_price(stop_loss, True)} <span style="font-size:12px; color:#94a3b8; font-weight:normal;">(산정기준: 주요 지지선 하단 - 0.3×ATR 타이트 손절)</span></div>
                <div class="report-line text-green">🚀 현실적 1차 익절가 : {fmt_price(target_price_1, True)} <span style="font-size:12px; color:#94a3b8; font-weight:normal;">(산정기준: 손익비 1:1.5 이상 타겟 저항선)</span></div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ Groq AI 상세 리포트 & 입체 매매 전략</div>
                <div class="ai-content" style="white-space: pre-line;">{ai_comment}</div>
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e: print(f"🚨 {stock_name} 생성 오류: {e}")

# =========================================================
# PART 2: 🇺🇸 미장(us_index.html) 분석 (10개 종목)
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
            stype = "🔥 Wall Street 거래대금 상위 및 빅테크/AI 핵심주"
            selected_us_targets[info.get('shortName', sym)] = (sym, stype)
    except Exception: continue

stock_cards_us_html = ""
for stock_name, (symbol, supply_type) in selected_us_targets.items():
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period="1y", interval="1d")
        df_weekly = ticker.history(period="2y", interval="1wk")
        
        if df_daily is None or df_weekly is None or len(df_daily) < 120 or len(df_weekly) == 0: continue

        df_daily['MA20'] = df_daily['Close'].rolling(20).mean()
        df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
        df_daily['MA120'] = df_daily['Close'].rolling(120).mean()
        
        std20 = df_daily['Close'].rolling(20).std()
        df_daily['BB_Upper'] = df_daily['MA20'] + (std20 * 2)
        df_daily['BB_Lower'] = df_daily['MA20'] - (std20 * 2)
        
        high9 = df_daily['High'].rolling(9).max()
        low9 = df_daily['Low'].rolling(9).min()
        df_daily['Tenkan'] = (high9 + low9) / 2
        high26 = df_daily['High'].rolling(26).max()
        low26 = df_daily['Low'].rolling(26).min()
        df_daily['Kijun'] = (high26 + low26) / 2
        high52 = df_daily['High'].rolling(52).max()
        low52 = df_daily['Low'].rolling(52).min()
        senkou_b = (high52 + low52) / 2
        senkou_a = (df_daily['Tenkan'] + df_daily['Kijun']) / 2
        
        df_daily['Senkou_A'] = senkou_a.shift(26)
        df_daily['Senkou_B'] = senkou_b.shift(26)

        df_recent120 = df_daily.tail(120).copy()
        price_bins = pd.cut(df_recent120['Close'], bins=15)
        vol_by_price = df_recent120.groupby(price_bins)['Volume'].sum()
        poc_bin = vol_by_price.idxmax()
        poc_price = round(float(poc_bin.mid), 2) if pd.notnull(poc_bin) else round(float(df_recent120['Close'].mean()), 2)

        df_daily = df_daily.bfill().ffill().dropna()
        if df_daily.empty or len(df_daily) == 0: continue

        delta = df_daily['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df_daily['RSI'] = 100 - (100 / (1 + (gain / loss)))
        rsi_val = round(float(np.nan_to_num(df_daily['RSI'].values[-1], nan=50.0)), 2)
        
        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_val = float(np.nan_to_num(df_daily['MACD'].values[-1]))
        signal_val = float(np.nan_to_num(df_daily['Signal'].values[-1]))
        
        tr = pd.concat([df_daily['High']-df_daily['Low'], np.abs(df_daily['High']-df_daily['Close'].shift()), np.abs(df_daily['Low']-df_daily['Close'].shift())], axis=1).max(axis=1)
        atr = float(np.nan_to_num(tr.rolling(14).mean().values[-1], nan=1.0))
        
        latest_close = round(float(np.nan_to_num(df_daily['Close'].values[-1])), 2)
        ma20_d = round(float(np.nan_to_num(df_daily['MA20'].values[-1])), 2)
        ma60_d = round(float(np.nan_to_num(df_daily['MA60'].values[-1])), 2)
        ma120_d = round(float(np.nan_to_num(df_daily['MA120'].values[-1])), 2)
        bb_up = round(float(np.nan_to_num(df_daily['BB_Upper'].values[-1])), 2)
        bb_low = round(float(np.nan_to_num(df_daily['BB_Lower'].values[-1])), 2)
        cloud_a = round(float(np.nan_to_num(df_daily['Senkou_A'].values[-1])), 2)
        cloud_b = round(float(np.nan_to_num(df_daily['Senkou_B'].values[-1])), 2)
        kijun_d = round(float(np.nan_to_num(df_daily['Kijun'].values[-1])), 2)
        cloud_top = max(cloud_a, cloud_b)

        stop_loss, target_price_1 = calculate_advanced_tech_levels(latest_close, ma20_d, kijun_d, bb_low, bb_up, cloud_top, poc_price, atr, is_krw=False)

        short_trend = "단기 상승 추세 📈" if latest_close >= ma20_d else "단기 하락 추세 📉"
        mid_trend = "중기 상승 추세 📈" if latest_close >= ma60_d else "중기 하락 추세 📉"
        bb_status = "상한선 돌파/근접 🚀" if latest_close >= bb_up * 0.99 else ("하한선 근접/지지 🟢" if latest_close <= bb_low * 1.01 else "밴드 내 안정 ⚖️")
        cloud_status = "구름대 위 상승 국면 🟢" if latest_close > cloud_top else "구름대 내부/하단 돌파 시도 🟡"

        df_recent10 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:${row['Open']:.2f} | High:${row['High']:.2f} | Low:${row['Low']:.2f} | Close:${row['Close']:.2f} | Vol:{int(row['Volume']):,}" for idx, row in df_recent10.iterrows()]
        raw_data_str = "\n".join(raw_lines)

        rsi_status = f"과매수 ({rsi_val}) ⚠️" if rsi_val >= 70 else (f"과매도 ({rsi_val}) 🟢" if rsi_val <= 30 else f"중립 ({rsi_val}) ⚖️")
        macd_status = "골든크로스 📈" if macd_val > signal_val else "데드크로스 📉"
        ma_status = f"정배열 (20>60>120) 🟢" if (ma20_d > ma60_d and ma60_d > ma120_d) else "역배열/혼조세 🔴"

        tradingview_url = f"https://www.tradingview.com/symbols/{symbol}/"

        print(f"  ⚡ [미장] {stock_name} AI 리포트 처리 중...")
        pick_reason, ai_comment = generate_ai_stock_analysis(stock_name, symbol, us_7d_news, raw_data_str, rsi_val, macd_status, ma_status, bb_status, cloud_status, poc_price, target_price_1, stop_loss, supply_type, "$")

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
        fig.add_hline(y=target_price_1, line_dash="dash", line_color="green", annotation_text=f"목표가: {fmt_price(target_price_1, False)}", row=1, col=1)
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"손절가: {fmt_price(stop_loss, False)}", row=1, col=1)
        
        colors = ['#f87171' if c < o else '#4ade80' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='거래량'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#38bdf8', width=1.2), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        
        fig.update_layout(height=540, margin=dict(l=10, r=10, t=10, b=40), xaxis_rangeslider_visible=False, template="plotly_dark")
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # 선정 이유 박스를 종목명 헤더 바로 아래 배치
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
                <div class="report-line text-red">🛑 손절가 : {fmt_price(stop_loss, False)} <span style="font-size:12px; color:#94a3b8; font-weight:normal;">(산정기준: 주요 지지선 하단 - 0.3×ATR 타이트 손절)</span></div>
                <div class="report-line text-green">🚀 현실적 1차 익절가 : {fmt_price(target_price_1, False)} <span style="font-size:12px; color:#94a3b8; font-weight:normal;">(산정기준: 손익비 1:1.5 이상 타겟 저항선)</span></div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ Groq AI 상세 리포트 & 입체 매매 전략</div>
                <div class="ai-content" style="white-space: pre-line;">{ai_comment}</div>
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e: print(f"🚨 {stock_name} 생성 오류: {e}")

# =========================================================
# PART 3: 🎯 마이 대시보드(index3.html) 분석 (요청 레이아웃 100% 적용)
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
        
        yf_ticker = ticker
        if is_krw:
            if not ticker.endswith(".KS") and not ticker.endswith(".KQ"):
                yf_ticker = f"{pure_code}.KS"
        else:
            yf_ticker = pure_code
                
        df_daily = None
        try:
            stock = yf.Ticker(yf_ticker)
            df_daily = stock.history(period="1y", interval="1d")
            
            if (df_daily is None or df_daily.empty or len(df_daily) == 0) and is_krw and yf_ticker.endswith(".KS"):
                yf_ticker = f"{pure_code}.KQ"
                df_daily = yf.Ticker(yf_ticker).history(period="1y", interval="1d")
        except Exception:
            df_daily = None

        if df_daily is None or df_daily.empty or len(df_daily) == 0:
            current_price = avg_price
            return_pct = 0.0
            eval_amount_krw = avg_price * quantity * fx
            profit_loss_krw = 0.0
            
            short_trend = "단기 데이터 미수집 ⚖️"
            mid_trend = "중기 데이터 미수집 ⚖️"
            bb_status = "밴드 미산출 ⚖️"
            cloud_status = "구름대 미산출 🟡"
            poc_price = avg_price
            rsi_status = "중립 (50) ⚖️"
            macd_status = "중립 ⚖️"
            stop_loss = avg_price * 0.9
            target_price_1 = avg_price * 1.1
            rsi_val = 50.0
            raw_data_str = "최근 데이터 미수집"
        else:
            latest_close = float(df_daily['Close'].values[-1])
            current_price = latest_close
            return_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
            eval_amount_krw = (current_price * quantity) * fx
            profit_loss_krw = ((current_price - avg_price) * quantity) * fx

            df_daily['MA20'] = df_daily['Close'].rolling(20).mean()
            df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
            df_daily['MA120'] = df_daily['Close'].rolling(120).mean()
            
            std20 = df_daily['Close'].rolling(20).std()
            df_daily['BB_Upper'] = df_daily['MA20'] + (std20 * 2)
            df_daily['BB_Lower'] = df_daily['MA20'] - (std20 * 2)

            high9 = df_daily['High'].rolling(9).max()
            low9 = df_daily['Low'].rolling(9).min()
            df_daily['Tenkan'] = (high9 + low9) / 2
            high26 = df_daily['High'].rolling(26).max()
            low26 = df_daily['Low'].rolling(26).min()
            df_daily['Kijun'] = (high26 + low26) / 2
            high52 = df_daily['High'].rolling(52).max()
            low52 = df_daily['Low'].rolling(52).min()
            senkou_b = (high52 + low52) / 2
            senkou_a = (df_daily['Tenkan'] + df_daily['Kijun']) / 2
            
            df_daily['Senkou_A'] = senkou_a.shift(26)
            df_daily['Senkou_B'] = senkou_b.shift(26)

            df_recent120 = df_daily.tail(120).copy()
            price_bins = pd.cut(df_recent120['Close'], bins=15)
            vol_by_price = df_recent120.groupby(price_bins)['Volume'].sum()
            poc_bin = vol_by_price.idxmax() if not vol_by_price.empty else None
            poc_price = float(poc_bin.mid) if poc_bin is not None and pd.notnull(poc_bin) else float(df_recent120['Close'].mean())

            df_daily = df_daily.bfill().ffill().dropna()

            delta = df_daily['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df_daily['RSI'] = 100 - (100 / (1 + (gain / loss)))
            rsi_val = round(float(np.nan_to_num(df_daily['RSI'].values[-1], nan=50.0)), 2) if ('RSI' in df_daily and len(df_daily['RSI']) > 0) else 50.0
            
            exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
            df_daily['MACD'] = exp1 - exp2
            df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
            macd_val = float(np.nan_to_num(df_daily['MACD'].values[-1])) if ('MACD' in df_daily and len(df_daily['MACD']) > 0) else 0.0
            signal_val = float(np.nan_to_num(df_daily['Signal'].values[-1])) if ('Signal' in df_daily and len(df_daily['Signal']) > 0) else 0.0
            
            tr = pd.concat([df_daily['High']-df_daily['Low'], np.abs(df_daily['High']-df_daily['Close'].shift()), np.abs(df_daily['Low']-df_daily['Close'].shift())], axis=1).max(axis=1)
            atr = float(np.nan_to_num(tr.rolling(14).mean().values[-1], nan=1000.0 if is_krw else 1.0)) if len(tr) > 0 else (1000.0 if is_krw else 1.0)

            ma20_d = float(df_daily['MA20'].values[-1]) if ('MA20' in df_daily and len(df_daily['MA20']) > 0 and not pd.isna(df_daily['MA20'].values[-1])) else latest_close
            ma60_d = float(df_daily['MA60'].values[-1]) if ('MA60' in df_daily and len(df_daily['MA60']) > 0 and not pd.isna(df_daily['MA60'].values[-1])) else latest_close
            bb_up = float(df_daily['BB_Upper'].values[-1]) if ('BB_Upper' in df_daily and len(df_daily['BB_Upper']) > 0 and not pd.isna(df_daily['BB_Upper'].values[-1])) else latest_close * 1.05
            bb_low = float(df_daily['BB_Lower'].values[-1]) if ('BB_Lower' in df_daily and len(df_daily['BB_Lower']) > 0 and not pd.isna(df_daily['BB_Lower'].values[-1])) else latest_close * 0.95
            cloud_a = float(df_daily['Senkou_A'].values[-1]) if ('Senkou_A' in df_daily and len(df_daily['Senkou_A']) > 0 and not pd.isna(df_daily['Senkou_A'].values[-1])) else latest_close
            cloud_b = float(df_daily['Senkou_B'].values[-1]) if ('Senkou_B' in df_daily and len(df_daily['Senkou_B']) > 0 and not pd.isna(df_daily['Senkou_B'].values[-1])) else latest_close
            kijun_d = float(df_daily['Kijun'].values[-1]) if ('Kijun' in df_daily and len(df_daily['Kijun']) > 0 and not pd.isna(df_daily['Kijun'].values[-1])) else latest_close
            cloud_top = max(cloud_a, cloud_b)

            stop_loss, target_price_1 = calculate_advanced_tech_levels(latest_close, ma20_d, kijun_d, bb_low, bb_up, cloud_top, poc_price, atr, is_krw=is_krw)

            short_trend = "단기 상승 추세 📈" if latest_close >= ma20_d else "단기 하락 추세 📉"
            mid_trend = "중기 상승 추세 📈" if latest_close >= ma60_d else "중기 하락 추세 📉"
            bb_status = "상한선 돌파/근접 🚀" if latest_close >= bb_up * 0.99 else ("하한선 근접/지지 🟢" if latest_close <= bb_low * 1.01 else "밴드 내 안정 ⚖️")
            cloud_status = "구름대 위 상승 국면 🟢" if latest_close > cloud_top else "구름대 내부/하단 돌파 시도 🟡"

            df_recent10 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
            if is_krw:
                raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:{int(row['Open']):,}원 | High:{int(row['High']):,}원 | Low:{int(row['Low']):,}원 | Close:{int(row['Close']):,}원 | Vol:{int(row['Volume']):,}" for idx, row in df_recent10.iterrows()]
            else:
                raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:${row['Open']:.2f} | High:${row['High']:.2f} | Low:${row['Low']:.2f} | Close:${row['Close']:.2f} | Vol:{int(row['Volume']):,}" for idx, row in df_recent10.iterrows()]
            raw_data_str = "\n".join(raw_lines)

            rsi_status = f"과매수 ({fmt_num(rsi_val)}) ⚠️" if rsi_val >= 70 else (f"과매도 ({fmt_num(rsi_val)}) 🟢" if rsi_val <= 30 else f"중립 ({fmt_num(rsi_val)}) ⚖️")
            macd_status = "골든크로스 📈" if macd_val > signal_val else "데드크로스 📉"
            ma_status = f"정배열 지지 🟢" if latest_close >= ma20_d else "역배열/혼조세 🔴"

        tv_prefix = f"KRX-{pure_code}" if is_krw else ticker
        tradingview_url = f"https://www.tradingview.com/symbols/{tv_prefix}/"

        print(f"  ⚡ [마이] {stock_name} 3줄 요약 AI 가이드 처리 중...")
        ai_3line_comment = generate_ai_toss_3line_analysis(
            stock_name, ticker, avg_price, current_price, return_pct, raw_data_str, rsi_val, macd_status, ma_status, bb_status, cloud_status, poc_price, target_price_1, stop_loss, is_krw
        )

        eval_formatted = f"{int(round(eval_amount_krw)):,}원"
        profit_formatted = f"({profit_loss_krw:+,.0f}원)"
        
        avg_price_formatted = fmt_price(avg_price, is_krw, show_decimal=is_krw)
        current_price_formatted = fmt_price(current_price, is_krw)
        poc_formatted = fmt_price(poc_price, is_krw)

        # ★ [수정 포인트] 사용자 요청 2열 가로 배치 레이아웃 (불릿 제거, 구분자 | 적용) ★
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
                <div class="ai-title">⚡ Groq AI 포트폴리오 3줄 대응 가이드</div>
                <div class="ai-content" style="white-space: pre-line;">{ai_3line_comment}</div>
            </div>
        </div>
        """

        total_eval_my += eval_amount_krw
        total_profit_my += profit_loss_krw

    except Exception as e:
        print(f"⚠️ {stock_name} 처리 중 예외: {e}")

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
    .macro-card:hover { 
        transform: translateY(-3px); 
        border-color: #38bdf8; 
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2); 
    }
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
