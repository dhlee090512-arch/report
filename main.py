import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup
from github import Github
from groq import Groq
import datetime
import time
import re
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# [설정] 깃허브 토큰 / 저장소 / Groq API 키
# =========================================================
GITHUB_TOKEN = "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"
GITHUB_REPO_NAME = "dhlee090512-arch/report"
GROQ_API_KEY = "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"

try:
    groq_client = Groq(api_key=GROQ_API_KEY.strip())
    print("✅ Groq AI 클라이언트 초기화 성공!")
except Exception as e:
    groq_client = None
    print(f"⚠️ Groq 클라이언트 설정 오류: {e}")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Chrome/120.0.0.0)',
    'Referer': 'https://finance.naver.com/'
}

kst_timezone = datetime.timezone(datetime.timedelta(hours=9))
now_str = datetime.datetime.now(kst_timezone).strftime("%Y-%m-%d %H:%M KST")

# =========================================================
# 📰 뉴스 헤드라인 키워드 수집 함수
# =========================================================
def get_naver_news_keywords():
    try:
        url = "https://finance.naver.com/news/mainnews.naver"
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        titles = [a.text.strip() for a in soup.select('.articleSubject a')]
        return " | ".join(titles[:10])
    except Exception:
        return "반도체 | AI | 금리 | 원전 | 실적발표 | 밸류업"

def get_yahoo_news_keywords():
    try:
        url = "https://finance.yahoo.com/news/"
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        titles = [h3.text.strip() for h3 in soup.find_all('h3') if len(h3.text.strip()) > 10]
        return " | ".join(titles[:10])
    except Exception:
        return "Fed Interest Rate | AI Tech | Big Tech Earnings | Inflation | NVIDIA | Semiconductor"

# =========================================================
# 🤖 AI 프롬프트 및 재검토(Self-Correction) 함수
# =========================================================
def sanitize_text(text):
    """파이썬 정규식을 활용한 한자/일본어 완전 제거 및 보정"""
    # 한자(CJK) 및 가나(일본어) 문자 제거
    cleaned = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]', '', text)
    return cleaned.strip()

def review_and_correct_text(draft_text, currency_name):
    """2단계: 작성된 코멘트를 재검토하여 한자/일본어를 완전히 제거하고 한글/영문으로 다듬기"""
    if not groq_client or not draft_text:
        return sanitize_text(draft_text)
    
    review_prompt = f"""
    아래 작성된 초안 텍스트를 검토하여 '오직 순수 한글(한국어)과 영문, 숫자, 문장부호'로만 재구성하라.
    
    [초안 텍스트]
    {draft_text}
    
    [재검토 및 수정 지침]
    1. 초안에 한자(漢字)나 일본어(가나) 문자가 단 한 글자라도 포함되어 있다면 모두 삭제하거나 한글 표기로 바꿀 것.
    2. 전문적인 주식 트레이더 톤앤매너와 원본의 분석 내용, 문장 맥락을 그대로 유지할 것.
    3. 통화 단주는 반드시 '{currency_name}' 단위 표기를 유지할 것.
    4. 검토 완료된 최종 결과 문장만 바로 출력할 것.
    """
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "너는 한자 및 일본어를 절대 사용하지 않고 오직 한글과 영문으로만 문장을 교정하는 감사관이다."},
                {"role": "user", "content": review_prompt}
            ],
            temperature=0.2, max_tokens=800
        )
        final_text = res.choices[0].message.content.strip()
        return sanitize_text(final_text)
    except Exception:
        return sanitize_text(draft_text)

def generate_ai_market_status(market_type, macro_data, news_keywords):
    """정량 지표 + AI 맥락 평가를 결합한 시장 상태(긍정/보통/부정) 판단"""
    if not groq_client: 
        return "보통 🟡", "시장 지표 데이터 관망 필요"
    
    prompt = f"""
    너는 글로벌 마켓 리서치 헤드이자 수석 애널리스트이다.
    아래 제공된 {market_type}의 매크로 지표, 지수 위치, 수급 상황 및 최근 주요 뉴스 헤드라인 키워드를 종합 분석하라.
    
    [정량 데이터 & 뉴스 이슈]
    - 지수 및 매크로: {macro_data}
    - 최근 주요 뉴스 키워드: {news_keywords}
    
    [판단 기준]
    - 단 하루의 기술적 반등이나 일시적 호재만으로 '긍정'을 부여하지 말 것.
    - 주요 이평선이 역배열이거나 수급 연속성이 약하면 '부정' 또는 '보통'으로 엄격하게 평가할 것.
    - 시장 상태 평가 단어는 반드시 [긍정 🟢 / 보통 🟡 / 부정 🔴] 중 하나만 선택할 것.
    
    [출력 양식]
    상태: <긍정 🟢 OR 보통 🟡 OR 부정 🔴>
    사유: <단기 급등/하락 여부, 이평선 배열, 주요 뉴스 이슈 및 수급 상황을 종합한 2~3문장의 판단 사유>
    
    [언어 제한] 한자(漢字) 및 일본어 절대 금지. 오직 순수 한글과 영문만 사용할 것.
    """
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "너는 한자와 일본어를 사용하지 않고 한국어와 영문으로만 답하는 주식 애널리스트이다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, max_tokens=300
        )
        content = res.choices[0].message.content.strip()
        status_line = "보통 🟡"
        reason_text = "시장 지표 및 추세 관망 구간입니다."
        for line in content.split("\n"):
            if line.startswith("상태:"): status_line = line.replace("상태:", "").strip()
            elif line.startswith("사유:"): reason_text = line.replace("사유:", "").strip()
        
        status_line = sanitize_text(status_line)
        reason_text = sanitize_text(reason_text)
        return status_line, reason_text
    except Exception as e:
        return "보통 🟡", f"시황 판단 생성 안내: {e}"

def generate_ai_stock_reason(stock_name, news_keywords, technical_summary):
    """추천 종목 고른 이유 1~2줄 요약 생성"""
    if not groq_client: 
        return "주요 수급 유입 및 기술적 이평선 정배열 모멘텀 포착으로 선정되었습니다."
    
    prompt = f"""
    너는 주식 분석 전문가이다. {stock_name} 종목이 오늘 주요 추천 종목으로 선정된 이유를 1~2줄(2문장 이내)로 깔끔하게 작성하라.
    
    [참고 데이터]
    - 최근 뉴스/정책/이슈 키워드: {news_keywords}
    - 종목 기술적/수급 요약: {technical_summary}
    
    [작성 요구사항]
    1. 단순 주가 상승 외에 최근 뉴스/정책 이슈와의 연관성 및 외국인/기관 수급 결합 요인을 강조할 것.
    2. 오직 순수 한글과 영문만 사용할 것. 한자 및 일본어 절대 금지.
    """
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "너는 한자를 절대 사용하지 않는 주식 큐레이터이다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4, max_tokens=150
        )
        draft = res.choices[0].message.content.strip()
        return sanitize_text(draft)
    except Exception:
        return "최근 뉴스 이슈 테마 유입과 함께 기관/외국인 수급 결합 및 이평선 지지가 확인되어 추천 종목으로 선정되었습니다."

def generate_ai_detailed_10line_analysis(stock_name, raw_data_str, rsi, macd_status, ma_status, target, stop, currency_symbol="원"):
    """종목별 단기 & 중장기 다각도 상세 분석 코멘트 (10줄 내외) + 2단계 재검토 적용"""
    if not groq_client: return "AI 상세 분석을 불러올 수 없습니다."
    
    currency_name = "원" if currency_symbol == "원" else "달러($)"
    
    # 1단계: 깊이 있는 다각도 분석 초안 생성
    prompt = f"""
    너는 20년 경력의 수석 기술적 분석 트레이더이다.
    제공된 {stock_name} 종목의 차트 Raw Data(OHLCV 원본) 및 핵심 보조지표를 바탕으로 
    **단기 및 중장기 관점에서의 다각도 기술적 분석 종합 리포트(약 10줄 내외)**를 작성하라.
    
    [📊 {stock_name} 차트 데이터]
    - 통화 단위: {currency_name} ({currency_symbol})
    - 최근 10 거래일 차트 Raw Data (날짜 | 시가 | 고가 | 저가 | 종가 | 거래량):
    {raw_data_str}
    - 보조 지표: RSI(14) = {rsi} / MACD 신호 = {macd_status} / 이동평균선 배열 = {ma_status}
    - 제시 목표가: {target:,}{currency_symbol} / 손절가: {stop:,}{currency_symbol}
    
    [필수 분석 지표 및 관점 - 아래 항목을 모두 조합하여 작성]
    1. **단기 관점:** 일봉 캔들 모형(위꼬리/아래꼬리), 최근 5~10일 거래량 수급 변화, RSI 과매수/과매도, MACD 단기 크로스, 단기 지지/저항선.
    2. **중장기 관점:** 주봉 추세 방향성, 20일/60일 이동평균선 정배열/역배열 구조, 중장기 주요 매물대.
    3. **차트 패턴 및 전반적 형태:** 눌림목 지지, 전고점 돌파 시도, 박스권 횡보 등 전체적인 차트 패턴 진단.
    4. **실전 대응 전략:** 목표가({target:,}{currency_symbol}) 및 손절가({stop:,}{currency_symbol})를 고려한 분할 매수 타점과 리스크 관리 지침.
    
    [엄격한 언어 제약]
    - **오직 한글(한국어)과 영문, 숫자만 사용할 것. 한자(漢字) 및 일본어 문자는 절대 포함하지 말 것.**
    - 통화 단위는 반드시 **{currency_symbol} ({currency_name})**만 표기할 것.
    - 줄바꿈을 적절히 활용하여 약 10줄 내외의 체계적인 본문으로 작성할 것. (서론/인사말 금지)
    """
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"너는 한자와 일본어를 쓰지 않고, 오직 한글과 영문으로만 주식 차트를 다각도로 종합 분석하는 전문가이다. 통화는 {currency_name}만 사용한다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, max_tokens=850
        )
        draft_comment = res.choices[0].message.content.strip()
        
        # 2단계: 텍스트 재검토 및 교정 (Self-Correction)
        final_comment = review_and_correct_text(draft_comment, currency_name)
        return final_comment
    except Exception as e:
        return f"상세 분석 생성 중 오류 발생: {e}"

# =========================================================
# PART 1: 🇰🇷 국장(index.html) 분석 및 배포
# =========================================================
print("\n" + "="*60)
print("🇰🇷 [PART 1] 한국 증시(국장) 뉴스 스캔 & AI 융합 분석 중...")
print("="*60)

kr_news_keywords = get_naver_news_keywords()

API_KEY = "RROFC9J3FLRFSHGO7VXC"
url_macro = f"https://ecos.bok.or.kr/api/KeyStatisticList/{API_KEY}/json/kr/1/100"
rate_val, m2_val, cci_val = "2.75", "4,184,079.8", "100.2"
try:
    res = requests.get(url_macro, timeout=10).json()
    if "KeyStatisticList" in res:
        rows = res["KeyStatisticList"]["row"]
        df_macro = pd.DataFrame(rows)
        m2_row = df_macro[df_macro['KEYSTAT_NAME'].str.contains('M2', na=False)].iloc[0]
        rate_row = df_macro[df_macro['KEYSTAT_NAME'].str.contains('기준금리', na=False)].iloc[0]
        rate_val = rate_row['DATA_VALUE']
        m2_val = f"{float(m2_row['DATA_VALUE']):,}"
        
        cci_rows = df_macro[df_macro['KEYSTAT_NAME'].str.contains('선행', na=False)]
        if not cci_rows.empty: cci_val = cci_rows.iloc[0]['DATA_VALUE']
except Exception: pass

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

def get_top_themes():
    try:
        res = requests.get("https://finance.naver.com/sise/theme.naver", headers=headers, timeout=10)
        res.encoding = 'euc-kr'
        soup = BeautifulSoup(res.text, 'html.parser')
        theme_rows = soup.select('table.type_1 tr')
        themes = []
        for row in theme_rows:
            cols = row.find_all('td')
            if len(cols) >= 3:
                t_name, t_rate = cols[0].text.strip(), cols[1].text.strip()
                if t_name: themes.append({'theme': t_name, 'rate': t_rate})
        return themes[:5]
    except Exception: return []

top_themes = get_top_themes()

all_items_kr = {item.get('nm'): item.get('cd') for item in foreign_items + organ_items if item.get('nm')}
selected_kr_targets = {}
candidate_kr = both_names + [name for name in foreign_names if name not in both_names]
for name in candidate_kr:
    if len(selected_kr_targets) >= 5: break
    code = all_items_kr.get(name)
    if code: selected_kr_targets[name] = f"{code}.KS"

foreign_top_str = ", ".join(foreign_names[:5]) if foreign_names else "없음"
organ_top_str = ", ".join(organ_names[:5]) if organ_names else "없음"
both_names_str = ", ".join(both_names) if both_names else "단독 수급 우수 종목 대체선정"

kr_macro_summary = f"한국은행 기준금리: {rate_val}%, M2 통화량: {m2_val}십억원, 경기선행지수: {cci_val}, 외국인 순매수: {foreign_top_str}, 기관 순매수: {organ_top_str}, 양매수 종목: {both_names_str}"
print("  ⚡ [국장] 시장 상태(긍정/보통/부정) 및 AI 시황 진단 중...")
kr_market_status, kr_market_reason = generate_ai_market_status("대한민국 주식시장(국장)", kr_macro_summary, kr_news_keywords)

summary_box_kr = f"""
<div class="summary-card">
    <div class="summary-title">🔍 [국장 시장 스캔 요약 리포트]</div>
    <div class="summary-grid">
        <div class="summary-item">
            <div class="sub-title">👥 [수급 상위 스캔]</div>
            <ul>
                <li><b>외국인 순매수 Top 5:</b> {foreign_top_str}</li>
                <li><b>기관 순매수 Top 5:</b> {organ_top_str}</li>
                <li><b>양매수(교집합):</b> <span class="highlight">{both_names_str}</span></li>
            </ul>
        </div>
        <div class="summary-item">
            <div class="sub-title">🔥 [실시간 강세 테마 TOP 5]</div>
            <ol style="margin:0; padding-left:20px;">{"".join([f"<li><b>{idx}. {t['theme']}</b> ({t['rate']})</li>" for idx, t in enumerate(top_themes, 1)])}</ol>
        </div>
    </div>
    <div class="ai-opinion-box" style="margin-top:15px; background:#0f2942; border-color:#38bdf8;">
        <div class="ai-title" style="color:#38bdf8;">🌐 AI Market Trend & Issue Briefing (국장 시황 관전평)</div>
        <div class="ai-content"><b>[판단 사유]:</b> {kr_market_reason}</div>
    </div>
    <div class="summary-footer">📌 <b>[뉴스 & 정책 헤드라인 주요 키워드]:</b> {kr_news_keywords[:80]}...</div>
</div>
"""

stock_cards_kr_html = ""
for stock_name, symbol in selected_kr_targets.items():
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period="6mo", interval="1d")
        df_weekly = ticker.history(period="2y", interval="1wk")
        if df_daily.empty or df_weekly.empty: continue

        df_daily['MA20'] = df_daily['Close'].rolling(20).mean()
        df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
        df_weekly['MA20_W'] = df_weekly['Close'].rolling(20).mean()
        
        delta = df_daily['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df_daily['RSI'] = 100 - (100 / (1 + (gain / loss)))
        rsi_val = round(df_daily['RSI'].iloc[-1], 2)
        
        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_val = df_daily['MACD'].iloc[-1]
        signal_val = df_daily['Signal'].iloc[-1]
        
        tr = pd.concat([df_daily['High']-df_daily['Low'], np.abs(df_daily['High']-df_daily['Close'].shift()), np.abs(df_daily['Low']-df_daily['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        latest_close = int(df_daily['Close'].iloc[-1])
        ma20_d, ma60_d, ma20_w = int(df_daily['MA20'].iloc[-1]), int(df_daily['MA60'].iloc[-1]), int(df_weekly['MA20_W'].iloc[-1])
        
        df_recent10 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        raw_lines = []
        for idx, row in df_recent10.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            raw_lines.append(f"{date_str} | Open:{int(row['Open']):,}원 | High:{int(row['High']):,}원 | Low:{int(row['Low']):,}원 | Close:{int(row['Close']):,}원 | Vol:{int(row['Volume']):,}")
        raw_data_str = "\n".join(raw_lines)

        support_line = int(max(df_daily['Low'].tail(20).min(), ma20_d * 0.98))
        buy_range_low = int(min(latest_close, ma20_d))
        stop_loss = int(support_line - (1.5 * atr))
        target_price_1 = max(int(df_daily['High'].tail(60).max()), int(latest_close + (2 * atr)))

        rsi_status = f"과매수 ({rsi_val}) ⚠️" if rsi_val >= 70 else (f"과매도 ({rsi_val}) 🟢" if rsi_val <= 30 else f"중립 ({rsi_val}) ⚖️")
        macd_status = "골든크로스 📈" if macd_val > signal_val else "데드크로스 📉"
        ma_status = "정배열 (20>60) 🟢" if ma20_d > ma60_d else "역배열/혼조세 🔴"
        weekly_trend = "상승 추세 📈" if latest_close > ma20_w else "조정/하락 추세 📉"

        pure_code = symbol.split('.')[0]
        tradingview_url = f"https://www.tradingview.com/symbols/KRX-{pure_code}/"

        tech_summary_str = f"현재가 {latest_close:,}원, RSI {rsi_val}, MACD {macd_status}, 주봉 {weekly_trend}"
        
        print(f"  ⚡ [국장] {stock_name} 선정 이유 및 재검토 적용 10줄 분석 작성 중...")
        pick_reason = generate_ai_stock_reason(stock_name, kr_news_keywords, tech_summary_str)
        ai_comment = generate_ai_detailed_10line_analysis(stock_name, raw_data_str, rsi_val, macd_status, ma_status, target_price_1, stop_loss, "원")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close'], name='주가'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MA20'], line=dict(color='orange', width=1.2), name='20일선'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MA60'], line=dict(color='purple', width=1.2), name='60일선'), row=1, col=1)
        fig.add_hline(y=target_price_1, line_dash="dash", line_color="green", annotation_text=f"목표가: {target_price_1:,}원", row=1, col=1)
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"손절가: {stop_loss:,}원", row=1, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['RSI'], line=dict(color='blue', width=1.2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        fig.update_layout(
            height=420, 
            margin=dict(l=10, r=10, t=10, b=50), 
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        )
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        stock_cards_kr_html += f"""
        <div class="card">
            <div class="console-report">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="report-header">📊 [기계적 종목 분석] {stock_name} ({symbol})</div>
                    <a href="{tradingview_url}" target="_blank" class="tv-link-btn">📈 TradingView 실시간 차트 열기 ↗</a>
                </div>
                <div class="stock-reason-box">
                    💡 <b>AI & 수급 스캔 선정 이유:</b> {pick_reason}
                </div>
                <div class="report-divider"></div>
                <div class="report-line">• 현재가 (종가) &nbsp;&nbsp;: <span class="highlight-val">{latest_close:,} 원</span></div>
                <div class="report-line">• 주봉 추세 / 이평 : {weekly_trend} / {ma_status}</div>
                <div class="report-line">• RSI / MACD 지표 : {rsi_status} / {macd_status}</div>
                <div class="report-divider"></div>
                <div class="report-line">🎯 권장 매수 구간 : {buy_range_low:,} 원 ~ {latest_close:,} 원</div>
                <div class="report-line text-red">🛑 손절가 (Stop)  : {stop_loss:,} 원 (지지선 이탈시 손절)</div>
                <div class="report-line text-green">🚀 1차 목표가    : {target_price_1:,} 원</div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ Groq Comprehensive Technical Analysis</div>
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
print("🇺🇸 [PART 2] 미국 증시(미장) 뉴스 스캔 & 10개 종목 상세 분석 중...")
print("="*60)

us_news_keywords = get_yahoo_news_keywords()

vix_val = 18.5
try:
    sp500 = yf.Ticker("^GSPC").history(period="5d")['Close'].iloc[-1]
    nasdaq = yf.Ticker("^IXIC").history(period="5d")['Close'].iloc[-1]
    us10y = yf.Ticker("^TNX").history(period="5d")['Close'].iloc[-1]
    vix_val = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
    macro_us_status = f"S&P 500: {sp500:,.1f} | NASDAQ: {nasdaq:,.1f} | US 10Y: {us10y:.2f}% | VIX 지수: {vix_val:.2f}"
except Exception: macro_us_status = "미국 주요 지수 & VIX 스캔 완료"

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
    backup_pool = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'AMZN', 'GOOGL', 'META', 'AVGO', 'PLTR', 'NFLX', 'INTC', 'ARM', 'SMCI', 'COIN']
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
            selected_us_targets[info.get('shortName', sym)] = sym
    except Exception: continue

scanned_us_str = ", ".join([f"<b>{name}</b>({sym})" for name, sym in selected_us_targets.items()])

us_macro_summary = f"지수 및 금리: {macro_us_status}, 당일 거래량/상승률 상위 주도주 10종목: {', '.join(selected_us_targets.keys())}"
print("  ⚡ [미장] 시장 상태(긍정/보통/부정) 및 AI 시황 진단 중...")
us_market_status, us_market_reason = generate_ai_market_status("미국 주식시장(미장)", us_macro_summary, us_news_keywords)

summary_box_us = f"""
<div class="summary-card">
    <div class="summary-title">🔍 [미국 증시 실시간 동적 주도주 TOP 10 스캔 요약]</div>
    <div class="summary-grid">
        <div class="summary-item">
            <div class="sub-title">⚡ 스캔 필터링 기준</div>
            <ul>
                <li><b>유동성/모멘텀:</b> 당일 거래량 폭발 & 상승률 상위 종목</li>
                <li><b>우량주 조건:</b> 시가총액 $10B (약 13조 원) 이상 대장주</li>
            </ul>
        </div>
        <div class="summary-item">
            <div class="sub-title">🎯 시장 상황 기반 추출 종목 (10개)</div>
            <p style="margin:5px 0;">{scanned_us_str}</p>
        </div>
    </div>
    <div class="ai-opinion-box" style="margin-top:15px; background:#0f2942; border-color:#38bdf8;">
        <div class="ai-title" style="color:#38bdf8;">🌐 AI Market Trend & Issue Briefing (미장 시황 관전평)</div>
        <div class="ai-content"><b>[판단 사유]:</b> {us_market_reason}</div>
    </div>
    <div class="summary-footer">📌 <b>[글로벌 뉴스 주요 키워드]:</b> {us_news_keywords[:80]}...</div>
</div>
"""

stock_cards_us_html = ""
for stock_name, symbol in selected_us_targets.items():
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period="6mo", interval="1d")
        df_weekly = ticker.history(period="2y", interval="1wk")
        if df_daily.empty or df_weekly.empty: continue

        df_daily['MA20'] = df_daily['Close'].rolling(20).mean()
        df_daily['MA60'] = df_daily['Close'].rolling(60).mean()
        df_weekly['MA20_W'] = df_weekly['Close'].rolling(20).mean()
        
        delta = df_daily['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df_daily['RSI'] = 100 - (100 / (1 + (gain / loss)))
        rsi_val = round(df_daily['RSI'].iloc[-1], 2)
        
        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_val = df_daily['MACD'].iloc[-1]
        signal_val = df_daily['Signal'].iloc[-1]
        
        tr = pd.concat([df_daily['High']-df_daily['Low'], np.abs(df_daily['High']-df_daily['Close'].shift()), np.abs(df_daily['Low']-df_daily['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        latest_close = round(df_daily['Close'].iloc[-1], 2)
        ma20_d, ma60_d, ma20_w = round(df_daily['MA20'].iloc[-1], 2), round(df_daily['MA60'].iloc[-1], 2), round(df_weekly['MA20_W'].iloc[-1], 2)
        
        df_recent10 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        raw_lines = []
        for idx, row in df_recent10.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            raw_lines.append(f"{date_str} | Open:${row['Open']:.2f} | High:${row['High']:.2f} | Low:${row['Low']:.2f} | Close:${row['Close']:.2f} | Vol:{int(row['Volume']):,}")
        raw_data_str = "\n".join(raw_lines)

        support_line = round(max(df_daily['Low'].tail(20).min(), ma20_d * 0.98), 2)
        buy_range_low = round(min(latest_close, ma20_d), 2)
        stop_loss = round(support_line - (1.5 * atr), 2)
        target_price_1 = max(round(df_daily['High'].tail(60).max(), 2), round(latest_close + (2 * atr), 2))

        rsi_status = f"과매수 ({rsi_val}) ⚠️" if rsi_val >= 70 else (f"과매도 ({rsi_val}) 🟢" if rsi_val <= 30 else f"중립 ({rsi_val}) ⚖️")
        macd_status = "골든크로스 📈" if macd_val > signal_val else "데드크로스 📉"
        ma_status = "정배열 (20>60) 🟢" if ma20_d > ma60_d else "역배열/혼조세 🔴"
        weekly_trend = "상승 추세 📈" if latest_close > ma20_w else "조정/하락 추세 📉"

        tradingview_url = f"https://www.tradingview.com/symbols/{symbol}/"

        tech_summary_str = f"현재가 ${latest_close}, RSI {rsi_val}, MACD {macd_status}, 주봉 {weekly_trend}"
        
        print(f"  ⚡ [미장] {stock_name} 선정 이유 및 재검토 적용 10줄 분석 작성 중...")
        pick_reason = generate_ai_stock_reason(stock_name, us_news_keywords, tech_summary_str)
        ai_comment = generate_ai_detailed_10line_analysis(stock_name, raw_data_str, rsi_val, macd_status, ma_status, target_price_1, stop_loss, "$")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close'], name='주가'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MA20'], line=dict(color='orange', width=1.2), name='20일선'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MA60'], line=dict(color='purple', width=1.2), name='60일선'), row=1, col=1)
        fig.add_hline(y=target_price_1, line_dash="dash", line_color="green", annotation_text=f"목표가: ${target_price_1}", row=1, col=1)
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"손절가: ${stop_loss}", row=1, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['RSI'], line=dict(color='blue', width=1.2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        fig.update_layout(
            height=420, 
            margin=dict(l=10, r=10, t=10, b=50), 
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        )
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        stock_cards_us_html += f"""
        <div class="card">
            <div class="console-report">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="report-header">🇺🇸 [미국주식 AI 분석] {stock_name} ({symbol})</div>
                    <a href="{tradingview_url}" target="_blank" class="tv-link-btn">📈 TradingView 실시간 차트 열기 ↗</a>
                </div>
                <div class="stock-reason-box">
                    💡 <b>AI & 수급 스캔 선정 이유:</b> {pick_reason}
                </div>
                <div class="report-divider"></div>
                <div class="report-line">• 현재가 (종가) &nbsp;&nbsp;: <span class="highlight-val">${latest_close:,.2f}</span></div>
                <div class="report-line">• 주봉 추세 / 이평 : {weekly_trend} / {ma_status}</div>
                <div class="report-line">• RSI / MACD 지표 : {rsi_status} / {macd_status}</div>
                <div class="report-divider"></div>
                <div class="report-line">🎯 권장 매수 구간 : ${buy_range_low:,.2f} ~ ${latest_close:,.2f}</div>
                <div class="report-line text-red">🛑 손절가 (Stop)  : ${stop_loss:,.2f} (지지선 이탈시 손절)</div>
                <div class="report-line text-green">🚀 1차 목표가    : ${target_price_1:,.2f}</div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ Groq Detailed 10-Line Technical Analysis</div>
                <div class="ai-content" style="white-space: pre-line;">{ai_comment}</div>
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e: print(f"🚨 {stock_name} 생성 오류: {e}")

# =========================================================
# PART 3: GitHub Pages 배포
# =========================================================
html_style = """
<style>
    body { font-family: 'Consolas', 'Courier New', monospace, -apple-system; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
    .container { max-width: 900px; margin: 0 auto; }
    .nav-bar { display: flex; justify-content: center; gap: 15px; margin-bottom: 20px; }
    .nav-btn { padding: 8px 16px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 14px; }
    .btn-kr-active { background: #2563eb; color: #ffffff; }
    .btn-us-active { background: #2563eb; color: #ffffff; }
    .btn-inactive { background: #334155; color: #94a3b8; }
    .header { background: #1e293b; color: #38bdf8; padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center; border: 1px solid #334155; }
    .summary-card { background: #131d31; border: 1px solid #38bdf8; border-radius: 12px; padding: 18px; margin-bottom: 25px; }
    .summary-title { font-size: 17px; font-weight: bold; color: #38bdf8; margin-bottom: 12px; border-bottom: 1px dashed #334155; padding-bottom: 6px; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; font-size: 14px; line-height: 1.6; }
    .sub-title { font-weight: bold; color: #facc15; margin-bottom: 6px; }
    .summary-item ul { margin: 0; padding-left: 18px; }
    .highlight { color: #4ade80; font-weight: bold; }
    .summary-footer { margin-top: 12px; padding-top: 10px; border-top: 1px dashed #334155; font-size: 13px; color: #cbd5e1; }
    .card { background: #1e293b; padding: 20px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #334155; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); }
    .console-report { background: #090d16; padding: 18px; border-radius: 8px; border: 1px solid #334155; font-size: 15px; line-height: 1.7; }
    .stock-reason-box { background: #1e1b4b; border-left: 4px solid #818cf8; padding: 10px 12px; border-radius: 4px; margin: 12px 0; font-size: 14px; color: #e0e7ff; line-height: 1.5; font-family: -apple-system, sans-serif; }
    .report-header { font-size: 17px; font-weight: bold; color: #38bdf8; }
    .report-divider { border-top: 1px dashed #475569; margin: 10px 0; }
    .report-line { margin: 4px 0; }
    .highlight-val { color: #facc15; font-weight: bold; }
    .text-red { color: #f87171; font-weight: bold; }
    .text-green { color: #4ade80; font-weight: bold; }
    .tv-link-btn { background: #2563eb; color: #ffffff; padding: 4px 10px; border-radius: 4px; text-decoration: none; font-size: 12px; font-weight: bold; transition: 0.2s; }
    .tv-link-btn:hover { background: #1d4ed8; }
    .ai-opinion-box { background: #062a1c; border: 1px solid #22c55e; border-radius: 8px; padding: 16px; margin-top: 15px; }
    .ai-title { font-size: 14px; font-weight: bold; color: #4ade80; margin-bottom: 10px; font-family: -apple-system, sans-serif; }
    .ai-content { font-size: 14px; color: #f1f5f9; line-height: 1.75; font-family: -apple-system, sans-serif; }
    .chart-container { margin-top: 20px; border-radius: 8px; overflow: hidden; }
</style>
"""

full_html_kr = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>🇰🇷 AI 국장 분석 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-kr-active">🇰🇷 국장 대시보드 (현재)</a><a href="us_index.html" class="nav-btn btn-inactive">🇺🇸 미장 대시보드 바로가기</a></div><div class="header"><h1 style="margin:0 0 10px 0;">📊 AI 국장 매크로 & 주도주 분석 대시보드 <span style="font-size:18px;">[시장 상황: {kr_market_status}]</span></h1><p style="margin:0; color:#94a3b8; font-size:14px;">업데이트: {now_str} | 한국은행 기준금리: {rate_val}% | M2: {m2_val} 십억원 | 경기선행지수: {cci_val}</p></div>{summary_box_kr}{stock_cards_kr_html}</div></body></html>"""
full_html_us = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>🇺🇸 AI 미장 분석 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-inactive">🇰🇷 국장 대시보드 바로가기</a><a href="us_index.html" class="nav-btn btn-us-active">🇺🇸 미장 대시보드 (현재)</a></div><div class="header"><h1 style="margin:0 0 10px 0; color:#60a5fa;">🇺🇸 US Stock 실시간 동적 주도주 대시보드 <span style="font-size:18px;">[시장 상황: {us_market_status}]</span></h1><p style="margin:0; color:#94a3b8; font-size:14px;">업데이트: {now_str} | {macro_us_status}</p></div>{summary_box_us}{stock_cards_us_html}</div></body></html>"""

print("\n🌐 [PART 3] GitHub Pages 웹 서버로 업로드 중...")
try:
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO_NAME)
    
    try:
        c_kr = repo.get_contents("index.html")
        repo.update_file("index.html", f"Self-Correction & Multi-Indicator Analysis KR: {now_str}", full_html_kr, c_kr.sha)
    except Exception:
        repo.create_file("index.html", f"Initial Self-Correction KR: {now_str}", full_html_kr)
    print("✅ 국장 대시보드(index.html) 배포 성공!")

    try:
        c_us = repo.get_contents("us_index.html")
        repo.update_file("us_index.html", f"Self-Correction & Multi-Indicator Analysis US: {now_str}", full_html_us, c_us.sha)
    except Exception:
        repo.create_file("us_index.html", f"Initial Self-Correction US: {now_str}", full_html_us)
    print("✅ 미장 대시보드(us_index.html) 배포 성공!")

    print("\n" + "="*65)
    print("🎉 [재검토 프롬프트로 한자 완전 제거 + 단기/중장기 다각도 지표 분석] 배포 성공!")
    print("🔗 🇰🇷 국장 접속 주소: https://dhlee090512-arch.github.io/report/index.html")
    print("🔗 🇺🇸 미장 접속 주소: https://dhlee090512-arch.github.io/report/us_index.html")
    print("="*65)

except Exception as e:
    print(f"🚨 GitHub 배포 중 오류 발생: {e}")
