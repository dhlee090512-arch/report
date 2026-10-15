# market_data.py
import re
import datetime
import calendar
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import yfinance as yf
import pandas_market_calendars as mcal
from config import headers, today_date, TEST_MODE
from llm_manager import fmt_num

KR_M2_URL = "https://tradingeconomics.com/south-korea/money-supply-m2"
KR_CLI_URL = "https://tradingeconomics.com/south-korea/leading-economic-index"
KR_VKOSPI_URL = "https://finance.naver.com/sise/sise_index.naver?code=VKOSPI"

US_M2_URL = "https://tradingeconomics.com/united-states/money-supply-m2"
US_CLI_URL = "https://tradingeconomics.com/united-states/leading-economic-index"
US_VIX_URL = "https://www.tradingview.com/symbols/CBOE-VIX/"

# =========================================================
# 📅 [증시 휴장일 판별 모듈]
# =========================================================
def get_market_open_status(market="KR"):
    if today_date.weekday() == 5:
        return False, "주말 휴장 (토요일)"
    elif today_date.weekday() == 6:
        return False, "주말 휴장 (일요일)"

    cal_name = 'XKRX' if market == "KR" else 'NYSE'
    try:
        exchange_cal = mcal.get_calendar(cal_name)
        schedule = exchange_cal.schedule(start_date=today_date, end_date=today_date)
        if schedule.empty:
            return False, "증시 공식 휴장일"
        return True, "정상 개장일"
    except Exception:
        return True, "개장일"

# =========================================================
# 🧙‍♀️ [만기일 D-Day 연산 모듈]
# =========================================================
def get_witching_day_alert(market="KR"):
    alerts = []
    for m_offset in range(2):
        month = today_date.month + m_offset
        year = today_date.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        
        cal = calendar.monthcalendar(year, month)
        
        if market == "KR":
            thursdays = [week[3] for week in cal if week[3] != 0]
            if len(thursdays) >= 2:
                target_day = thursdays[1]
                target_date = datetime.date(year, month, target_day)
                d_day = (target_date - today_date).days
                
                is_quad = month in [3, 6, 9, 12]
                event_name = "네 마녀의 날 (선물·옵션 동시 만기일) 🧙‍♀️" if is_quad else "월별 옵션 만기일"
                
                if 0 <= d_day <= 7:
                    d_str = "오늘(D-Day)" if d_day == 0 else f"D-{d_day}"
                    alerts.append(f"🚨 <b>[{d_str}]</b> {target_date.strftime('%m/%d')} 한국 {event_name} - 동시호가 변동성 주의")

        elif market == "US":
            fridays = [week[4] for week in cal if week[4] != 0]
            if len(fridays) >= 3:
                target_day = fridays[2]
                target_date = datetime.date(year, month, target_day)
                d_day = (target_date - today_date).days
                
                is_triple = month in [3, 6, 9, 12]
                event_name = "세/네 마녀의 날 (Triple/Quadruple Witching Day) 🧙‍♀️" if is_triple else "월별 옵션 만기일"
                
                if 0 <= d_day <= 7:
                    d_str = "오늘(D-Day)" if d_day == 0 else f"D-{d_day}"
                    alerts.append(f"🚨 <b>[{d_str}]</b> {target_date.strftime('%m/%d')} 미국 {event_name} - 파생상품 결제 변동성 주의")

    return alerts

# =========================================================
# 📅 [글로벌 경제 캘린더 실시간 수집 모듈]
# =========================================================
def get_economic_calendar_events(market="KR"):
    events_list = []
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            target_country = "USD" if market == "US" else "KRW"
            
            for item in data:
                country = item.get("country", "")
                impact = item.get("impact", "")
                title = item.get("title", "")
                date_str = item.get("date", "")[:10]
                time_str = item.get("time", "")
                
                if (country == "USD" and impact == "High") or (country == target_country and impact in ["High", "Medium"]):
                    try:
                        event_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                        d_diff = (event_date - today_date).days
                        if 0 <= d_diff <= 5:
                            d_tag = "오늘" if d_diff == 0 else f"D-{d_diff}"
                            events_list.append(f"• <b>[{d_tag}]</b> {date_str[5:]} {time_str} {country} {title} [중요 🔴]")
                    except Exception:
                        pass
    except Exception as e:
        print(f"⚠️ 경제 캘린더 수집 예외: {e}")
        
    return events_list[:4]

# =========================================================
# 🌐 [거시 지표 수치 파싱]
# =========================================================
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
    
    m2_date_str = date_m2 if date_m2 else "최신 발표 기준"
    val_cli, unit_cli, date_cli = parse_te_summary_val(KR_CLI_URL)
    cli_val_str = f"{fmt_num(val_cli)} Pts" if val_cli else "101.20 Pts"
    cli_date_str = date_cli if date_cli else "최신 발표 기준"
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
        
    m2_date_str = date_m2 if date_m2 else "최신 발표 기준"
    val_cli, unit_cli, date_cli = parse_te_summary_val(US_CLI_URL)
    cli_val_str = f"{fmt_num(val_cli)} Pts" if val_cli else "102.10 Pts"
    cli_date_str = date_cli if date_cli else "최신 발표 기준"

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

# =========================================================
# 📰 [뉴스 크롤링]
# =========================================================
def get_naver_7days_news(now_dt):
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

def scrape_saveticker_news():
    news_items = []
    try:
        url = "https://www.saveticker.com/news"
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            cards = soup.select('article') or soup.select('.news-card') or soup.select('li')
            for c in cards[:40]:
                title_elem = c.select_one('h2') or c.select_one('h3') or c.select_one('.title') or c.select_one('a')
                if not title_elem: continue
                title = title_elem.text.strip()
                if len(title) < 8: continue

                desc_elem = c.select_one('p') or c.select_one('.desc') or c.select_one('.summary')
                desc = desc_elem.text.strip()[:200] if desc_elem else title

                time_elem = c.select_one('time') or c.select_one('.date') or c.select_one('.time')
                time_tag = time_elem.text.strip() if time_elem else "최근 24시간 내"

                news_items.append({"title": title, "desc": desc, "time": time_tag})
    except Exception as e:
        print(f"⚠️ SaveTicker 스크래핑 예외: {e}")

    if len(news_items) < 5:
        yahoo_raw = get_yahoo_7days_news().split('\n')
        for y_title in yahoo_raw[:25]:
            if len(y_title) > 8:
                news_items.append({"title": y_title, "desc": y_title, "time": "최근 12시간 내"})

    return news_items[:35]
