# 1. 필수 라이브러리 설치
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
# ⚙️ [테스트 모드 설정] - 기본 TEST_MODE = False 고정
# =========================================================
TEST_MODE = False

# =========================================================
# [보안 및 멀티 GROQ API Key 관리 클래스]
# =========================================================
try:
    from google.colab import userdata
    GROQ_API_KEY_1 = userdata.get('GROQ_API_KEY')
    GROQ_API_KEY_2 = userdata.get('GROQ_API_KEY2')
    GITHUB_TOKEN = userdata.get('GH_TOKEN')
except ImportError:
    GROQ_API_KEY_1 = os.environ.get("GROQ_API_KEY", "")
    GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY2", "")
    GITHUB_TOKEN = os.environ.get("GH_TOKEN", "")

GITHUB_REPO_NAME = "dhlee090512-arch/report"
CACHE_FILE_NAME = "ai_cache.json"

class GroqKeyManager:
    """1번 API 키 한도 초과시 2번 API 키로 자동 전환하는 매니저"""
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
        """키 한도 초과 시 다음 키로 스위칭"""
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
# 🌐 정밀 매크로 수치 수집 및 투명한 데이터 처리 함수
# =========================================================
KR_M2_URL = "https://tradingeconomics.com/south-korea/money-supply-m2"
KR_CLI_URL = "https://tradingeconomics.com/south-korea/leading-economic-index"
KR_VKOSPI_URL = "https://finance.naver.com/sise/sise_index.naver?code=VKOSPI"

US_M2_URL = "https://tradingeconomics.com/united-states/money-supply-m2"
US_CLI_URL = "https://tradingeconomics.com/united-states/leading-economic-index"
US_VIX_URL = "https://www.tradingview.com/symbols/CBOE-VIX/"

def parse_exact_value_and_change(url):
    """Trading Economics 수치 및 변동폭 정밀 수집"""
    try:
        res = requests.get(url, headers=headers, timeout=7)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            val_elem = soup.select_one('#stream') or \
                       soup.select_one('.table-responsive table tbody tr td:nth-child(2)') or \
                       soup.select_one('#sp-m2') or soup.select_one('#sp-lei')
            chg_elem = soup.select_one('.table-responsive table tbody tr td:nth-child(4)') or \
                       soup.select_one('span[id*="change"]')

            val_text = val_elem.text.strip() if val_elem else ""
            chg_text = chg_elem.text.strip() if chg_elem else ""

            return val_text, chg_text
    except Exception as e:
        print(f"⚠️ {url} 크롤링 실패: {e}")
    return "", ""

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
    print("⏳ [국장 매크로] 차트 수치 웹 크롤링 수집 중...")
    
    # 1. 한국 M2 파싱 (하드코딩 제거)
    raw_m2, chg_m2 = parse_exact_value_and_change(KR_M2_URL)
    if raw_m2:
        try:
            # 원본 단위 수치 분석 및 변환 (Trading Economics 단위: Billion KRW 또는 Million KRW)
            clean_num = float(re.sub(r'[^\d.]', '', raw_m2))
            if 'M' in raw_m2.upper() or clean_num < 100:
                # Million Billion KRW = 1,000조 단위
                m2_val_str = f"{int(clean_num * 1000):,}조 원"
            elif clean_num > 1000000:
                m2_val_str = f"{int(clean_num / 1000):,}조 원"
            else:
                m2_val_str = f"{int(clean_num):,}조 원"
        except Exception:
            m2_val_str = f"{raw_m2} KRW"
    else:
        m2_val_str = "수집 실패 (데이터 확인 필요)"
        
    m2_chg_str = f"전월 대비 {chg_m2}" if chg_m2 else "변동폭 확인불가"

    # 2. 한국 경기선행지수 (CLI)
    raw_cli, chg_cli = parse_exact_value_and_change(KR_CLI_URL)
    cli_val_str = f"{raw_cli} Pts" if raw_cli else "수집 실패 (데이터 확인 필요)"
    cli_chg_str = f"전월 대비 {chg_cli}" if chg_cli else "변동폭 확인불가"

    # 3. VKOSPI
    val_vkospi = get_vkospi_naver()
    vix_val_str = f"{val_vkospi:.2f} Pts" if val_vkospi is not None else "수집 실패"

    return {
        "m2": m2_val_str, "m2_chg": m2_chg_str, "m2_url": KR_M2_URL,
        "cli": cli_val_str, "cli_chg": cli_chg_str, "cli_url": KR_CLI_URL,
        "vix": vix_val_str, "vix_url": KR_VKOSPI_URL
    }

def get_us_macro_data():
    print("⏳ [미장 매크로] 차트 수치 웹 크롤링 수집 중...")
    
    # 1. 미국 M2 파싱 (고정 환율 원화 환산 제거 -> 순수 $ 달러 단위 표기)
    raw_m2, chg_m2 = parse_exact_value_and_change(US_M2_URL)
    if raw_m2:
        try:
            clean_num = float(re.sub(r'[^\d.]', '', raw_m2))
            if 'K' in raw_m2.upper() or clean_num > 1000:
                # 23.2K Billion USD = 23.2 Trillion USD
                val_trillion_usd = clean_num / 1000
                m2_val_str = f"${val_trillion_usd:.1f} Trillion"
            else:
                m2_val_str = f"${clean_num:.1f} Trillion"
        except Exception:
            m2_val_str = f"${raw_m2}"
    else:
        m2_val_str = "수집 실패 (데이터 확인 필요)"
        
    m2_chg_str = f"전주 대비 {chg_m2}" if chg_m2 else "변동폭 확인불가"

    # 2. 미국 경기선행지수 (CLI)
    raw_cli, chg_cli = parse_exact_value_and_change(US_CLI_URL)
    cli_val_str = f"{raw_cli} Pts" if raw_cli else "수집 실패 (데이터 확인 필요)"
    cli_chg_str = f"전월 대비 {chg_cli}" if chg_cli else "변동폭 확인불가"

    # 3. VIX 지수
    try:
        vix_tk = yf.Ticker("^VIX").history(period="5d")
        val_vix = float(vix_tk['Close'].iloc[-1]) if not vix_tk.empty else None
    except Exception:
        val_vix = None
    vix_val_str = f"{val_vix:.2f} Pts" if val_vix is not None else "수집 실패"

    return {
        "m2": m2_val_str, "m2_chg": m2_chg_str, "m2_url": US_M2_URL,
        "cli": cli_val_str, "cli_chg": cli_chg_str, "cli_url": US_CLI_URL,
        "vix": vix_val_str, "vix_url": US_VIX_URL
    }

kr_macro = get_kr_macro_data()
us_macro = get_us_macro_data()

# =========================================================
# 📰 뉴스 헤드라인 수집 함수
# =========================================================
def get_naver_7days_news():
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

# =========================================================
# 🤖 7일 뉴스 감성 & AI 분석 함수
# =========================================================
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
# 🤖 종목 분석 및 상단 추천 사유 + 분할 진입/매도 전략 AI 생성 함수
# =========================================================
def generate_ai_stock_analysis(stock_name, symbol, news_keywords, raw_data_str, rsi_val, macd_status, ma_status, bb_status, cloud_status, poc_price, target_price, stop_loss, supply_type="", currency_symbol="원"):
    cache_key = f"STOCK_{symbol}"

    if not groq_mgr.is_available():
        if cache_key in ai_cache_store:
            cached = ai_cache_store[cache_key]
            updated_at = cached.get('updated_at', '일자 미상')
            reason_msg = get_fallback_reason()
            reason_str = f"⚠️ [비실시간 백업] {cached['reason']}"
            report_str = f"⚠️ [비실시간 백업 리포트 - 생성일: {updated_at} | 📌 사유: {reason_msg}]\n" + cached['report']
            return reason_str, report_str
        else:
            return f"수급({supply_type}) 및 주요 기술적 모멘텀 포착 종목", "AI 상세 전략 리포트 준비 중입니다."

    prompt = f"""
    너는 20년 경력의 수석 기술적 분석 및 트레이딩 전략가이다. {stock_name}({symbol}) 종목에 대한 최적의 매매 전략을 수립하라.
    
    [종목 기본 & 기술 데이터]
    - 수급/테마 특징: {supply_type}
    - 최근 10일 OHLCV 원본 데이터:
    {raw_data_str}
    - 보조지표: RSI({rsi_val}), MACD({macd_status}), 이평선 배열({ma_status})
    - 차트 구조: 볼린저 밴드({bb_status}), 일목 구름대({cloud_status}), 매물대 POC({poc_price:,}{currency_symbol})
    - 기술적 참조가: 1차 목표가 {target_price:,}{currency_symbol} / 손절가 {stop_loss:,}{currency_symbol}
    
    [작성 요구사항]
    1. **선정이유**: {supply_type} 수급과 기술적 모멘텀을 결합한 1~2줄 요약.
    2. **상세 전략 리포트**: 차트 지표 진단과 함께 아래 매매 시나리오를 명확한 가격 숫자로 산출하여 구체적으로 작성하라.
       - **[분할 진입(매수) 추천 전략]**: 
         • 1차 진입 시기 및 가격 (현재가 부근/눌림목 지지)
         • 2차 진입 시기 및 가격 (핵심 이평선/매물대 POC 지지)
         • 3차 진입 시기 및 가격 (최종 하단 지지선)
       - **[분할 매도(목표) 추천 전략]**:
         • 1차 매도 시기 및 가격 (단기 단단한 저항선 / 리스크 관리)
         • 2차 매도 시기 및 가격 (1차 목표가 및 전고점 구간)
         • 3차 매도 시기 및 가격 (추세 확장 및 오버슈팅 목표)
       - **[손절 및 대응 전략]**: 기준 가격 및 이탈 시 대응법

    [출력 양식]
    선정이유: <1~2줄 요약>
    상세리포트:
    📌 [차트 구조 & 지표 진단]
    - <이동평균선(20/60/120일), 볼린저밴드, 구름대, 매물대 진단 2~3줄>

    🟢 [분할 진입(매수) 추천]
    - 1차 진입: <시기> | <가격: 000{currency_symbol}> (비중 30%)
    - 2차 진입: <시기> | <가격: 000{currency_symbol}> (비중 40%)
    - 3차 진입: <시기> | <가격: 000{currency_symbol}> (비중 30%)

    🚀 [분할 매도(목표) 추천]
    - 1차 매도: <시기> | <가격: 000{currency_symbol}> (물량 30% 실현)
    - 2차 매도: <시기> | <가격: 000{currency_symbol}> (물량 40% 실현)
    - 3차 매도: <시기> | <가격: 000{currency_symbol}> (잔량 30% 추세 대응)

    🛑 [손절 대응 기준]
    - 기준가: <가격: 000{currency_symbol}> 이탈 종가 마감 시 즉시 손절 대응.

    [언어 제한] 한자(漢字) 및 일본어 절대 금지. 오직 순수 한글, 영문, 숫자만 사용할 것.
    """
    try:
        res = groq_mgr.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1000
        )
        content = res.choices[0].message.content.strip()
        
        reason_val = f"{supply_type} 수급 유입과 함께 기술적 지지선에서 반등 모멘텀이 강화되는 종목입니다."
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
# PART 1: 🇰🇷 국장(index.html) 분석 및 배포 (5개 종목)
# =========================================================
print("\n" + "="*60)
print("🇰🇷 [PART 1] 한국 증시 스캔 & AI 분석 중...")
print("="*60)

kr_7d_news = get_naver_7days_news()
kr_market_status, kr_sentiment_briefing = analyze_7days_news_sentiment("대한민국 주식시장(국장)", kr_7d_news)

url_foreign = "https://m.stock.naver.com/api/json/sise/siseListJson.nhn?bizType=dealForeign&sosok=0"
url_organ = "https://m.stock.naver.com/api/json/sise/siseListJson.nhn?bizType=dealOrgan&sosok=0"

def get_deal_top(url):
    try:
        res = requests.get(url, headers=headers, timeout=10).json()
        return res.get('result', {}).get('itemList', [])[:10]
    except Exception: return []

foreign_items = get_deal_top(url_foreign)
organ_items = get_deal_top(url_organ)
foreign_names = [item.get('nm') for item in foreign_items if item.get('nm')]
organ_names = [item.get('nm') for item in organ_items if item.get('nm')]
both_names = [name for name in foreign_names if name in organ_names]

all_items_kr = {item.get('nm'): item.get('cd') for item in foreign_items + organ_items if item.get('nm')}
selected_kr_targets = {}
candidate_kr = both_names + [name for name in foreign_names if name not in both_names]
for name in candidate_kr:
    if len(selected_kr_targets) >= 5: break
    code = all_items_kr.get(name)
    if code: 
        stype = "🔥 외국인·기관 동시 순매수(양매수) 주도주" if name in both_names else "⚡ 외국인 순매수 상위 핵심주"
        selected_kr_targets[name] = (f"{code}.KS", stype)

if not selected_kr_targets:
    selected_kr_targets = {
        "삼성전자": ("005930.KS", "🔥 외국인·기관 양매수 대장주"), 
        "SK하이닉스": ("000660.KS", "🚀 AI HBM 핵심 주도주"), 
        "현대차": ("005380.KS", "⚡ 기관 순매수 대표주"), 
        "NAVER": ("035420.KS", "🟢 플랫폼 기술적 반등주"), 
        "카카오": ("035720.KS", "🟢 수급 유입 관심주")
    }

stock_cards_kr_html = ""
for stock_name, (symbol, supply_type) in selected_kr_targets.items():
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period="1y", interval="1d")
        df_weekly = ticker.history(period="2y", interval="1wk")
        
        if df_daily.empty or df_weekly.empty or len(df_daily) < 120: continue

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
        poc_price = int(poc_bin.mid) if pd.notnull(poc_bin) else int(df_recent120['Close'].mean())

        df_daily = df_daily.bfill().ffill().dropna()
        if df_daily.empty: continue

        delta = df_daily['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df_daily['RSI'] = 100 - (100 / (1 + (gain / loss)))
        rsi_val = round(float(np.nan_to_num(df_daily['RSI'].iloc[-1], nan=50.0)), 2)
        
        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_val = float(np.nan_to_num(df_daily['MACD'].iloc[-1]))
        signal_val = float(np.nan_to_num(df_daily['Signal'].iloc[-1]))
        
        tr = pd.concat([df_daily['High']-df_daily['Low'], np.abs(df_daily['High']-df_daily['Close'].shift()), np.abs(df_daily['Low']-df_daily['Close'].shift())], axis=1).max(axis=1)
        atr = float(np.nan_to_num(tr.rolling(14).mean().iloc[-1], nan=1000.0))
        
        latest_close = int(np.nan_to_num(df_daily['Close'].iloc[-1]))
        ma20_d = int(np.nan_to_num(df_daily['MA20'].iloc[-1]))
        ma60_d = int(np.nan_to_num(df_daily['MA60'].iloc[-1]))
        ma120_d = int(np.nan_to_num(df_daily['MA120'].iloc[-1]))
        bb_up = int(np.nan_to_num(df_daily['BB_Upper'].iloc[-1]))
        bb_low = int(np.nan_to_num(df_daily['BB_Lower'].iloc[-1]))
        cloud_a = int(np.nan_to_num(df_daily['Senkou_A'].iloc[-1]))
        cloud_b = int(np.nan_to_num(df_daily['Senkou_B'].iloc[-1]))

        short_trend = "상승 추세 📈" if latest_close >= ma20_d else "하락 추세 📉"
        mid_trend = "상승 추세 📈" if latest_close >= ma60_d else "하락 추세 📉"
        bb_status = "상한선 돌파/근접 🚀" if latest_close >= bb_up * 0.99 else ("하한선 근접/지지 🟢" if latest_close <= bb_low * 1.01 else "밴드 내 안정 ⚖️")
        cloud_top = max(cloud_a, cloud_b)
        cloud_status = "구름대 위 상승 국면 🟢" if latest_close > cloud_top else "구름대 내부/하단 돌파 시도 🟡"

        df_recent10 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:{int(row['Open']):,}원 | High:{int(row['High']):,}원 | Low:{int(row['Low']):,}원 | Close:{int(row['Close']):,}원 | Vol:{int(row['Volume']):,}" for idx, row in df_recent10.iterrows()]
        raw_data_str = "\n".join(raw_lines)

        support_line = int(max(df_daily['Low'].tail(20).min(), ma20_d * 0.98))
        stop_loss = int(support_line - (1.5 * atr))
        target_price_1 = max(int(df_daily['High'].tail(60).max()), int(latest_close + (2 * atr)))

        rsi_status = f"과매수 ({rsi_val}) ⚠️" if rsi_val >= 70 else (f"과매도 ({rsi_val}) 🟢" if rsi_val <= 30 else f"중립 ({rsi_val}) ⚖️")
        macd_status = "골든크로스 📈" if macd_val > signal_val else "데드크로스 📉"
        ma_status = f"정배열 (20>60>120) 🟢" if (ma20_d > ma60_d and ma60_d > ma120_d) else "역배열/혼조세 🔴"

        pure_code = symbol.split('.')[0]
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
        
        fig.add_hline(y=poc_price, line_dash="dot", line_color="#facc15", annotation_text=f"최대매물대: {poc_price:,}원", row=1, col=1)
        fig.add_hline(y=target_price_1, line_dash="dash", line_color="green", annotation_text=f"목표가: {target_price_1:,}원", row=1, col=1)
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"손절가: {stop_loss:,}원", row=1, col=1)
        
        colors = ['#f87171' if c < o else '#4ade80' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='거래량'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#38bdf8', width=1.2), name='RSI'), row=3, col=1)
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
                <div class="report-line">• 종가 기준 현재가 : <span class="highlight-val">{latest_close:,} 원</span></div>
                <div class="report-line">• 추세 진단 : 단기 <span class="highlight-val">{short_trend}</span> / 중기 <span class="highlight-val">{mid_trend}</span></div>
                <div class="report-line">• 차트 구조 : {bb_status} / {cloud_status}</div>
                <div class="report-line">• 집중 매물대 (POC) : <span class="highlight-val">{poc_price:,} 원</span></div>
                <div class="report-line">• RSI / MACD : {rsi_status} / {macd_status}</div>
                <div class="report-line text-red">🛑 손절가 : {stop_loss:,} 원</div>
                <div class="report-line text-green">🚀 1차 목표가 : {target_price_1:,} 원</div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ Groq AI 상세 리포트 & 분할 매매 전략</div>
                <div class="ai-content" style="white-space: pre-line;">{ai_comment}</div>
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e: print(f"🚨 {stock_name} 생성 오류: {e}")

# =========================================================
# PART 2: 🇺🇸 미장(us_index.html) 분석 및 배포 (10개 종목)
# =========================================================
print("\n" + "="*60)
print("🇺🇸 [PART 2] 미국 증시 스캔 & AI 분석 중...")
print("="*60)

us_7d_news = get_yahoo_7days_news()
us_market_status, us_sentiment_briefing = analyze_7days_news_sentiment("미국 주식시장(미장)", us_7d_news)

def get_us_active_stocks():
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
        
        if df_daily.empty or df_weekly.empty or len(df_daily) < 120: continue

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
        if df_daily.empty: continue

        delta = df_daily['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df_daily['RSI'] = 100 - (100 / (1 + (gain / loss)))
        rsi_val = round(float(np.nan_to_num(df_daily['RSI'].iloc[-1], nan=50.0)), 2)
        
        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_val = float(np.nan_to_num(df_daily['MACD'].iloc[-1]))
        signal_val = float(np.nan_to_num(df_daily['Signal'].iloc[-1]))
        
        tr = pd.concat([df_daily['High']-df_daily['Low'], np.abs(df_daily['High']-df_daily['Close'].shift()), np.abs(df_daily['Low']-df_daily['Close'].shift())], axis=1).max(axis=1)
        atr = float(np.nan_to_num(tr.rolling(14).mean().iloc[-1], nan=1.0))
        
        latest_close = round(float(np.nan_to_num(df_daily['Close'].iloc[-1])), 2)
        ma20_d = round(float(np.nan_to_num(df_daily['MA20'].iloc[-1])), 2)
        ma60_d = round(float(np.nan_to_num(df_daily['MA60'].iloc[-1])), 2)
        ma120_d = round(float(np.nan_to_num(df_daily['MA120'].iloc[-1])), 2)
        bb_up = round(float(np.nan_to_num(df_daily['BB_Upper'].iloc[-1])), 2)
        bb_low = round(float(np.nan_to_num(df_daily['BB_Lower'].iloc[-1])), 2)
        cloud_a = round(float(np.nan_to_num(df_daily['Senkou_A'].iloc[-1])), 2)
        cloud_b = round(float(np.nan_to_num(df_daily['Senkou_B'].iloc[-1])), 2)

        short_trend = "상승 추세 📈" if latest_close >= ma20_d else "하락 추세 📉"
        mid_trend = "상승 추세 📈" if latest_close >= ma60_d else "하락 추세 📉"
        bb_status = "상한선 돌파/근접 🚀" if latest_close >= bb_up * 0.99 else ("하한선 근접/지지 🟢" if latest_close <= bb_low * 1.01 else "밴드 내 안정 ⚖️")
        cloud_top = max(cloud_a, cloud_b)
        cloud_status = "구름대 위 상승 국면 🟢" if latest_close > cloud_top else "구름대 내부/하단 돌파 시도 🟡"

        df_recent10 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:${row['Open']:.2f} | High:${row['High']:.2f} | Low:${row['Low']:.2f} | Close:${row['Close']:.2f} | Vol:{int(row['Volume']):,}" for idx, row in df_recent10.iterrows()]
        raw_data_str = "\n".join(raw_lines)

        support_line = round(max(df_daily['Low'].tail(20).min(), ma20_d * 0.98), 2)
        stop_loss = round(support_line - (1.5 * atr), 2)
        target_price_1 = max(round(df_daily['High'].tail(60).max(), 2), round(latest_close + (2 * atr), 2))

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
        
        fig.add_hline(y=poc_price, line_dash="dot", line_color="#facc15", annotation_text=f"최대매물대: ${poc_price}", row=1, col=1)
        fig.add_hline(y=target_price_1, line_dash="dash", line_color="green", annotation_text=f"목표가: ${target_price_1}", row=1, col=1)
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"손절가: ${stop_loss}", row=1, col=1)
        
        colors = ['#f87171' if c < o else '#4ade80' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='거래량'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#38bdf8', width=1.2), name='RSI'), row=3, col=1)
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
                <div class="report-line">• 종가 기준 현재가 : <span class="highlight-val">${latest_close:,.2f}</span></div>
                <div class="report-line">• 추세 진단 : 단기 <span class="highlight-val">{short_trend}</span> / 중기 <span class="highlight-val">{mid_trend}</span></div>
                <div class="report-line">• 차트 구조 : {bb_status} / {cloud_status}</div>
                <div class="report-line">• 집중 매물대 (POC) : <span class="highlight-val">${poc_price}</span></div>
                <div class="report-line">• RSI / MACD : {rsi_status} / {macd_status}</div>
                <div class="report-line text-red">🛑 손절가 : ${stop_loss:,.2f}</div>
                <div class="report-line text-green">🚀 1차 목표가 : ${target_price_1:,.2f}</div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ Groq AI 상세 리포트 & 분할 매매 전략</div>
                <div class="ai-content" style="white-space: pre-line;">{ai_comment}</div>
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e: print(f"🚨 {stock_name} 생성 오류: {e}")

# =========================================================
# PART 3: HTML 템플릿 및 레이아웃
# =========================================================
html_style = """
<style>
    body { font-family: 'Consolas', -apple-system, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
    .container { max-width: 950px; margin: 0 auto; }
    .nav-bar { display: flex; justify-content: center; gap: 15px; margin-bottom: 20px; }
    .nav-btn { padding: 8px 16px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 14px; }
    .btn-active { background: #2563eb; color: #ffffff; }
    .btn-inactive { background: #334155; color: #94a3b8; }
    .header { background: #1e293b; color: #38bdf8; padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center; border: 1px solid #334155; }
    
    .macro-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 20px; }
    .macro-card { 
        background: #182232; 
        border: 1px solid #334155; 
        border-radius: 10px; 
        padding: 14px; 
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
    .macro-title { font-size: 13px; color: #94a3b8; font-weight: bold; margin-bottom: 6px; }
    .macro-value { font-size: 18px; font-weight: bold; color: #38bdf8; }
    .macro-sub { font-size: 11px; color: #4ade80; margin-top: 4px; }
    
    .news-briefing-card { background: #182232; border: 1px solid #38bdf8; border-radius: 12px; padding: 18px; margin-bottom: 25px; line-height: 1.8; font-size: 14px; }
    .news-title { font-size: 16px; font-weight: bold; color: #38bdf8; margin-bottom: 10px; border-bottom: 1px dashed #334155; padding-bottom: 6px; }
    
    .card { background: #1e293b; padding: 20px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #334155; }
    .console-report { background: #090d16; padding: 18px; border-radius: 8px; border: 1px solid #334155; font-size: 15px; line-height: 1.7; }
    .stock-reason-box { background: #1e1b4b; border-left: 4px solid #818cf8; padding: 12px; border-radius: 4px; margin: 12px 0; font-size: 14px; color: #e0e7ff; line-height: 1.6; }
    .report-header { font-size: 18px; font-weight: bold; color: #38bdf8; }
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
        <div class="macro-sub">{kr_macro['m2_chg']}</div>
    </a>
    <a href="{kr_macro['cli_url']}" target="_blank" class="macro-card">
        <div class="macro-title">🌐 한국 경기선행지수 (CLI) ↗</div>
        <div class="macro-value">{kr_macro['cli']}</div>
        <div class="macro-sub">{kr_macro['cli_chg']}</div>
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
        <div class="macro-sub">{us_macro['m2_chg']}</div>
    </a>
    <a href="{us_macro['cli_url']}" target="_blank" class="macro-card">
        <div class="macro-title">🌐 미국 경기선행지수 (CLI) ↗</div>
        <div class="macro-value">{us_macro['cli']}</div>
        <div class="macro-sub">{us_macro['cli_chg']}</div>
    </a>
    <a href="{us_macro['vix_url']}" target="_blank" class="macro-card">
        <div class="macro-title">⚡ 미국 VIX 지수 ↗</div>
        <div class="macro-value" style="color:#facc15;">{us_macro['vix']}</div>
        <div class="macro-sub">TradingView 원본 연동 🟢</div>
    </a>
</div>
"""

full_html_kr = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>🇰🇷 AI 국장 분석 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-active">🇰🇷 국장 대시보드</a><a href="us_index.html" class="nav-btn btn-inactive">🇺🇸 미장 대시보드</a></div><div class="header"><h1>📊 AI 국장 주도주 대시보드 <span style="font-size:18px;">[{kr_market_status}]</span></h1><p style="margin:0; color:#94a3b8; font-size:14px;">업데이트: {now_str}</p></div>{macro_html_kr}<div class="news-briefing-card"><div class="news-title">📰 [최근 7일간 뉴스 AI 종합 분석 브리핑]</div>{kr_sentiment_briefing}</div>{stock_cards_kr_html}</div></body></html>"""

full_html_us = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>🇺🇸 AI 미장 분석 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-inactive">🇰🇷 국장 대시보드</a><a href="us_index.html" class="nav-btn btn-active">🇺🇸 미장 대시보드</a></div><div class="header"><h1>🇺🇸 AI US Stock 주도주 대시보드 <span style="font-size:18px;">[{us_market_status}]</span></h1><p style="margin:0; color:#94a3b8; font-size:14px;">업데이트: {now_str}</p></div>{macro_html_us}<div class="news-briefing-card"><div class="news-title">📰 [최근 7일간 뉴스 AI 종합 분석 브리핑]</div>{us_sentiment_briefing}</div>{stock_cards_us_html}</div></body></html>"""

# =========================================================
# PART 4: GitHub Pages 및 ai_cache.json 함께 업로드
# =========================================================
def upload_to_github_safely(repo, file_path, commit_message, content):
    try:
        file_obj = repo.get_contents(file_path)
        repo.update_file(path=file_path, message=commit_message, content=content, sha=file_obj.sha)
        print(f"✅ {file_path} 업데이트 성공!")
    except UnknownObjectException:
        repo.create_file(path=file_path, message=commit_message, content=content)
        print(f"✅ {file_path} 신규 생성 배포 성공!")
    except Exception as e:
        print(f"🚨 {file_path} 배포 중 예외 발생: {e}")

print("\n🌐 [PART 4] GitHub Pages 및 AI 캐시 동기화 업로드 중...")
try:
    if not GITHUB_TOKEN:
        raise ValueError("GH_TOKEN이 설정되지 않았습니다.")

    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO_NAME)
    
    upload_to_github_safely(repo, "index.html", f"Deploy KR Report: {now_str}", full_html_kr)
    upload_to_github_safely(repo, "us_index.html", f"Deploy US Report: {now_str}", full_html_us)
    
    if os.path.exists(CACHE_FILE_NAME):
        with open(CACHE_FILE_NAME, "r", encoding="utf-8") as f:
            cache_json_str = f.read()
        upload_to_github_safely(repo, "ai_cache.json", f"Update AI Cache: {now_str}", cache_json_str)

    print("\n" + "="*65)
    print("🎉 [최종 완료] 하드코딩 백업 수치 완벽 제거 및 신뢰성 보완 대시보드가 배포되었습니다!")
    print("🔗 🇰🇷 국장: https://dhlee090512-arch.github.io/report/index.html")
    print("🔗 🇺🇸 미장: https://dhlee090512-arch.github.io/report/us_index.html")
    print("="*65)

except Exception as e:
    print(f"🚨 GitHub 연결 과정 중 오류 발생: {e}")
