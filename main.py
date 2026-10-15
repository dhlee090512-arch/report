# main.py
import os
import re
import json
import datetime
import requests
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from github import Github, UnknownObjectException

from config import (
    GEMINI_API_KEY_1, GEMINI_API_KEY_2, GROQ_API_KEY_1, GROQ_API_KEY_2,
    GITHUB_TOKEN, GITHUB_REPO_NAME, CACHE_FILE_NAME,
    TOSS_CLIENT_ID, TOSS_CLIENT_SECRET,
    headers, kst_timezone, now_dt, now_str, today_date,
    ai_cache_store, is_cache_valid, save_ai_cache, save_entire_cache,
    should_refresh_daily_pivot, toss_request_with_proxy_failover, TEST_MODE
)
from llm_manager import (
    llm_mgr, fmt_price, fmt_num, adjust_to_tick_size,
    calculate_atr, validate_stop_loss_with_atr, calculate_wilder_rsi,
    extract_peaks_and_troughs, parse_price_from_text, sanitize_text
)
from market_data import (
    get_market_open_status, get_witching_day_alert, get_economic_calendar_events,
    get_kr_macro_data, get_us_macro_data, get_usd_krw_rate,
    get_naver_7days_news, get_yahoo_7days_news, scrape_saveticker_news
)
from html_templates import (
    render_macro_grid, build_kr_page, build_us_page, build_my_page, build_news_page
)

# =========================================================
# 🚨 시장 변동성 상태 머신 & 뱃지
# =========================================================
def get_index_change_rate(ticker_symbol):
    try:
        df = yf.Ticker(ticker_symbol).history(period="2d")
        if df is not None and len(df) >= 2:
            prev_close = float(df['Close'].iloc[-2])
            curr_close = float(df['Close'].iloc[-1])
            return ((curr_close - prev_close) / prev_close) * 100.0
        elif df is not None and len(df) == 1:
            prev_close = float(df['Open'].iloc[-1])
            curr_close = float(df['Close'].iloc[-1])
            return ((curr_close - prev_close) / prev_close) * 100.0
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
            banner_msg = f"🚨 <b>[시장 급락 경보 ({avg_chg:+.2f}%)]</b> 긴급 방어 지지선 및 손절선 풀 업데이트 완료"
        elif state["status"] == "RECOVERY_HANDLED":
            state["status"] = "HIGH_VOLATILITY_LOCKED"
            banner_msg = f"🚨 <b>[초고변동성 롤러코스터 경보 ({avg_chg:+.2f}%)]</b> 장중 잦은 급변동 발생. 신규 진입을 멈추고 관망 및 현금 비중 유지를 권장합니다."
    elif avg_chg >= -0.5 and state["status"] == "CRASH_HANDLED" and state["recovery_count"] == 0:
        state["status"] = "RECOVERY_HANDLED"
        state["recovery_count"] = 1
        is_emergency_refresh = True
        banner_msg = f"🟢 <b>[시장 급반등 확인 ({avg_chg:+.2f}%)]</b> 상방 목표가 및 추세 복구 풀 업데이트 완료"

    ai_cache_store[state_key] = state
    save_ai_cache(state_key, state)
    return is_emergency_refresh, banner_msg, defense_mode, avg_chg

def update_and_get_consecutive_days(symbol, is_new_pivot_cycle):
    tracker_key = "RECOMMEND_DAYS_TRACKER"
    tracker = ai_cache_store.get(tracker_key, {})
    item_info = tracker.get(symbol, {"days": 1, "last_pivot_date": ""})
    today_str = today_date.strftime("%Y-%m-%d")
    
    if is_new_pivot_cycle:
        last_date_str = item_info.get("last_pivot_date", "")
        if last_date_str:
            try:
                last_dt = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
                delta_days = (today_date - last_dt).days
                if 1 <= delta_days <= 3:
                    item_info["days"] = item_info.get("days", 1) + 1
                elif delta_days > 3:
                    item_info["days"] = 1
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
# 🤖 AI 분석 함수
# =========================================================
def analyze_7days_news_sentiment(market_type, news_text, force_refresh=False):
    cache_key = f"MARKET_{market_type}"
    if not force_refresh and not should_refresh_daily_pivot(market_type):
        cached = ai_cache_store[cache_key]
        return cached['status'], cached['briefing_html'], cached.get('updated_at', now_str)

    if not llm_mgr.is_available():
        if cache_key in ai_cache_store:
            cached_data = ai_cache_store[cache_key]
            return cached_data['status'], cached_data['briefing_html'], cached_data.get('updated_at', now_str)
        return "보통 🟡", "분석 데이터를 불러올 수 없습니다.", now_str

    prompt = f"""
    너는 수석 마켓 분석가이다. 아래 제공된 지난 7일간의 {market_type} 주요 뉴스 헤드라인 모음을 종합 분석하라.
    [지난 7일간 뉴스 헤드라인]
    {news_text}
    [출력 양식]
    상태: <긍정 OR 보통 OR 부정>
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
        status_match = re.search(r'상태:\s*(.*)', content)
        raw_status = status_match.group(1).strip() if status_match else "보통"
        status_val = "긍정 🟢" if "긍정" in raw_status else ("부정 🔴" if "부정" in raw_status else "보통 🟡")

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
        save_ai_cache(cache_key, {"status": status_val, "briefing_html": raw_briefing_html})
        return status_val, raw_briefing_html, now_str
    except Exception as e:
        return "보통 🟡", f"뉴스 분석 생성 안내: {e}", now_str

def generate_ai_stock_analysis(stock_name, symbol, news_keywords, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, latest_close, ma20_d, ma60_d, ma120_d, atr_val=0.0, supply_type="", currency_symbol="원", force_refresh=False):
    cache_key = f"STOCK_{symbol}"
    is_krw = True if currency_symbol in ["원", "KRW"] else False

    if not force_refresh and is_cache_valid(cache_key, max_hours=4):
        cached = ai_cache_store[cache_key]
        return cached.get('reason', ''), cached.get('basic_report', ''), cached.get('deep_report', ''), cached.get('parsed_prices', {}), cached.get('score_info', {"score": 75, "label": "적극 매수 🔵", "p_reason": "", "t_reason": "", "m_reason": ""}), cached.get('updated_at', now_str)

    if not llm_mgr.is_available():
        if cache_key in ai_cache_store:
            cached = ai_cache_store[cache_key]
            return cached.get('reason', ''), cached.get('basic_report', ''), cached.get('deep_report', ''), cached.get('parsed_prices', {}), cached.get('score_info', {"score": 75, "label": "적극 매수 🔵", "p_reason": "", "t_reason": "", "m_reason": ""}), cached.get('updated_at', now_str)
        return "수급/모멘텀 모니터링 종목", "AI 분석 준비 중", "", {"buy": None, "stop": None, "target1": None, "target2": None}, {"score": 0, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}, now_str

    if is_krw:
        ex_buy, ex_stop, ex_t1, ex_t2 = f"{int(latest_close*0.98)}", f"{int(latest_close*0.95)}", f"{int(latest_close*1.05)}", f"{int(latest_close*1.10)}"
        price_rule = "- 원화 가격이므로 콤마(,) 및 '원' 단위 없이 순수 정수 숫자로만 출력하라."
    else:
        ex_buy, ex_stop, ex_t1, ex_t2 = f"{latest_close*0.98:.2f}", f"{latest_close*0.95:.2f}", f"{latest_close*1.05:.2f}", f"{latest_close*1.10:.2f}"
        price_rule = f"- 현재 주가가 ${latest_close:.2f} 이므로, 반드시 달러($) 기호 없이 소수점 2자리 마침표(.)를 포함한 형태로만 출력하라."

    prompt = f"""
너는 20년 경력의 수석 기술적 분석 및 퀀트 펀드매니저이다.
단순 호재나 정배열만 보지 말고, [현재 주가가 진입하기에 매력적인 자리인가? (손익비/안전마진)]을 최우선으로 검증하여 -100점부터 +100점 사이의 [AI 복합 매매 스코어]와 상세 리포트를 도출하라.
[종목 기본 & 수급/뉴스 데이터]
- 종목명: {stock_name} ({symbol}) | 수급/테마 특징: {supply_type}
- 최근 시장 주요 뉴스 이슈/호재: {news_keywords}
[정량적 차트 지표]
- 현재가: {fmt_price(latest_close, is_krw)}
- 이동평균선: 20일선({fmt_price(ma20_d, is_krw)}), 60일선({fmt_price(ma60_d, is_krw)}), 120일선({fmt_price(ma120_d, is_krw)}) / 배열: {ma_status}
- 보조지표: Wilder RSI({rsi_val}) & RSI Signal({rsi_signal_val}) [{rsi_cross_status}], MACD({macd_status}), 볼린저밴드({bb_status}), 일목구름대({cloud_status})
- 매물대 & 파동: POC({fmt_price(poc_price, is_krw)}), 120일 최고({fmt_price(max_120, is_krw)}), 최저({fmt_price(min_120, is_krw)}) / 최근 파동: {peaks_and_troughs_summary}
[단기 캔들 15일]
{raw_data_str_15days}
[가격 출력 규칙]
{price_rule}
[출력 양식]
파싱_스코어점수: <+00 또는 -00>
파싱_스코어라벨: <적극 매수 🔵 OR 눌림목 매수 🟢 OR 관망 🟡 OR 비중 축소 🟠 OR 전량 매도 🔴>
파싱_가격매력도근거: <20일선/매물대 지지 및 손절 대비 손익비 평가 1줄>
파싱_기술지표근거: <RSI 및 이평선 배열, 캔들 지표 평가 1줄>
파싱_뉴스수급근거: <재료 지속성 및 외인/기관 수급 평가 1줄>
선정이유: <외인/기관 수급, 뉴스 호재 종합 요약 2~3줄>
파싱_눌림목가: <{ex_buy}>
파싱_손절가: <{ex_stop}>
파싱_1차익절가: <{ex_t1}>
파싱_2차익절가: <{ex_t2}>
핵심요약리포트:
📌 [차트 구조 진단]
- 내용 압축
🟢 [안전 매수/손절]
- 추천 진입 타점: {ex_buy}{currency_symbol} / 손절선: {ex_stop}{currency_symbol}
🚀 [분할 익절]
- 1차: {ex_t1}{currency_symbol} / 2차: {ex_t2}{currency_symbol}
상세리포트:
• 상세 진단 및 대응 가이드
[언어 제한] 한자 및 일본어 절대 금지. 순수 한글, 영문, 숫자만 사용할 것.
"""
    try:
        content = llm_mgr.generate_completion(prompt, temperature=0.3, max_tokens=1800)
        reason_match = re.search(r'선정이유:\s*(.*)', content)
        reason_val = reason_match.group(1).strip() if reason_match else f"{supply_type} 모멘텀 종목입니다."
        basic_match = re.search(r'핵심요약리포트:\s*([\s\S]*?)(?=상세리포트:|$)', content)
        basic_report_val = basic_match.group(1).strip() if basic_match else content
        deep_match = re.search(r'상세리포트:\s*([\s\S]*)', content)
        deep_report_val = deep_match.group(1).strip() if deep_match else ""

        score_match = re.search(r'파싱_스코어점수:\s*([+-]?\d+)', content)
        label_match = re.search(r'파싱_스코어라벨:\s*(.*)', content)
        p_match = re.search(r'파싱_가격매력도근거:\s*(.*)', content)
        t_match = re.search(r'파싱_기술지표근거:\s*(.*)', content)
        m_match = re.search(r'파싱_뉴스수급근거:\s*(.*)', content)

        score_info = {
            "score": int(score_match.group(1)) if score_match else 75,
            "label": label_match.group(1).strip() if label_match else "적극 매수 🔵",
            "p_reason": sanitize_text(p_match.group(1) if p_match else "지지선 대비 손익비 우수"),
            "t_reason": sanitize_text(t_match.group(1) if t_match else f"RSI {rsi_val} 탄력 유지"),
            "m_reason": sanitize_text(m_match.group(1) if m_match else "수급 모멘텀 지속")
        }

        ai_buy = parse_price_from_text(content, "파싱_눌림목가", is_krw, latest_close)
        ai_stop = parse_price_from_text(content, "파싱_손절가", is_krw, latest_close)
        ai_target1 = parse_price_from_text(content, "파싱_1차익절가", is_krw, latest_close)
        ai_target2 = parse_price_from_text(content, "파싱_2차익절가", is_krw, latest_close)
        if ai_stop and atr_val > 0 and ai_buy:
            ai_stop = validate_stop_loss_with_atr(ai_buy, ai_stop, atr_val, is_krw)

        parsed_prices = {"buy": ai_buy, "stop": ai_stop, "target1": ai_target1, "target2": ai_target2}
        save_ai_cache(cache_key, {
            "reason": sanitize_text(reason_val),
            "basic_report": sanitize_text(basic_report_val),
            "deep_report": sanitize_text(deep_report_val),
            "parsed_prices": parsed_prices,
            "score_info": score_info
        })
        return sanitize_text(reason_val), sanitize_text(basic_report_val), sanitize_text(deep_report_val), parsed_prices, score_info, now_str
    except Exception as e:
        print(f"⚠️ {stock_name} AI 리포트 생성 오류: {e}")
        return "AI 분석 호출 실패", f"🚨 AI 분석 오류 발생: {e}", "", {"buy": None, "stop": None, "target1": None, "target2": None}, {"score": 0, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}, now_str

def generate_ai_toss_3line_analysis(stock_name, symbol, avg_price, current_price, return_pct, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, is_krw=True, force_refresh=False):
    cache_key = f"TOSS_MY_{symbol}"
    currency_symbol = "원" if is_krw else "$"

    if not force_refresh and is_cache_valid(cache_key, max_hours=4):
        cached = ai_cache_store[cache_key]
        return cached.get('deep_report', ''), cached.get('stop_price'), cached.get('target_price'), cached.get('pyramid_price'), cached.get('pyramid_type'), cached.get('score_info', {"score": 50, "label": "관망 🟡", "p_reason": "", "t_reason": "", "m_reason": ""}), cached.get('updated_at', now_str)

    if not llm_mgr.is_available():
        if cache_key in ai_cache_store:
            cached = ai_cache_store[cache_key]
            return cached.get('deep_report', ''), cached.get('stop_price'), cached.get('target_price'), cached.get('pyramid_price'), cached.get('pyramid_type'), cached.get('score_info', {"score": 50, "label": "관망 🟡", "p_reason": "", "t_reason": "", "m_reason": ""}), cached.get('updated_at', now_str)
        return "[테스트 모드] AI 연동 미사용 상태입니다.", None, None, None, None, {"score": 0, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}, now_str

    avg_p_text = fmt_price(avg_price, is_krw, show_decimal=is_krw) if avg_price > 0 else "0원 (상장폐지/청산대기)"
    if is_krw:
        ex_pyramid, ex_stop, ex_target = f"{int(current_price*0.98)}", f"{int(current_price*0.95)}", f"{int(current_price*1.08)}"
        price_rule = "- 원화 가격이므로 콤마 및 '원' 단위 없이 순수 정수 숫자로만 출력하라."
    else:
        ex_pyramid, ex_stop, ex_target = f"{current_price*0.98:.2f}", f"{current_price*0.95:.2f}", f"{current_price*1.08:.2f}"
        price_rule = f"- 달러($) 기호 없이 소수점 2자리 마침표를 포함한 형태로만 출력하라."

    prompt = f"""
너는 20년 경력의 수석 포트폴리오 트레이딩 전문가이다.
[사용자의 실제 계좌 손익 상태 (수익권 {return_pct:+.2f}% vs 손실권)]과 [추세 및 차트 지지/저항 구조]를 결합하여 [포지션 파워 스코어 (-100 ~ +100)]와 8단계 결론 라벨이 포함된 상세가이드를 작성하라.
[종목 데이터]
- 종목명: {stock_name} ({symbol}) | 평단가: {avg_p_text} ({return_pct:+.2f}%) | 현재가: {fmt_price(current_price, is_krw)}
- 지표: RSI({rsi_val}), MACD({macd_status}), 이평선({ma_status}), 매물대POC({fmt_price(poc_price, is_krw)})
[15일 캔들]
{raw_data_str_15days}
[가격 규칙]
{price_rule}
[8단계 결론 라벨 중 택1]
1. [일부 익절 🟢 - 수익권 중 상단 저항 직면 / 분할 차익 실현]
2. [비중 축소 🟠 - 손실권 중 상단 저항 직면 / 손실 축소 필요]
3. [익절 🟢 - 수익권 중 지지선 이탈 / 이익 실현 필요]
4. [일부 손절 🔴 - 손실권 중 단기 지지 이탈 / 비중 축소 권장]
5. [손절 🔴 - 손실권 중 주요 추세선 붕괴 확인 / 전량 청산 권장]
6. [불타기 고려 🔵 - 상승 추세 및 눌림목 지지 확인]
7. [물타기 고려 🟠 - 장기 지지 및 바닥 반등 확인]
8. [관망 🟡 - 추세 유지 및 박스권 횡보 중 / 포지션 유지]
[출력 양식]
파싱_포지션점수: <+00 또는 -00>
파싱_포지션라벨: <선택된 라벨>
파싱_가격매력도근거: <평단 대비 현재가 수익률 위치 평가 1줄>
파싱_지지저항근거: <POC 및 지지/저항 1줄>
파싱_대응권고: <분할 대응 권고 1줄>
파싱_추매타입: <불타기 OR 물타기 OR 없음>
파싱_추매추천가: <{ex_pyramid} (없을 시 0)>
파싱_Trailing손절가: <{ex_stop}>
파싱_동적목표가: <{ex_target}>
상세가이드:
결론: [라벨 문구]
• 포지션 및 지지선 근거 서술
[언어 제한] 한자/일본어 절대 금지. 순수 한글, 영문, 숫자만 사용할 것.
"""
    try:
        content = llm_mgr.generate_completion(prompt, temperature=0.3, max_tokens=1400)
        stop_val = parse_price_from_text(content, "파싱_Trailing손절가", is_krw, current_price)
        target_val = parse_price_from_text(content, "파싱_동적목표가", is_krw, current_price)
        pyramid_val = parse_price_from_text(content, "파싱_추매추천가", is_krw, current_price)
        type_match = re.search(r'파싱_추매타입:\s*(불타기|물타기)', content)
        pyramid_type = type_match.group(1) if (type_match and pyramid_val and pyramid_val > 0) else None

        score_match = re.search(r'파싱_포지션점수:\s*([+-]?\d+)', content)
        label_match = re.search(r'파싱_포지션라벨:\s*(.*)', content)
        p_match = re.search(r'파싱_가격매력도근거:\s*(.*)', content)
        t_match = re.search(r'파싱_지지저항근거:\s*(.*)', content)
        m_match = re.search(r'파싱_대응권고:\s*(.*)', content)

        score_info = {
            "score": int(score_match.group(1)) if score_match else 50,
            "label": label_match.group(1).strip() if label_match else "관망 🟡",
            "p_reason": sanitize_text(p_match.group(1) if p_match else f"평단 대비 {return_pct:+.2f}% 위치"),
            "t_reason": sanitize_text(t_match.group(1) if t_match else "POC 지지 테스트"),
            "m_reason": sanitize_text(m_match.group(1) if m_match else "손절선 준수")
        }

        deep_match = re.search(r'상세가이드:\s*([\s\S]*)', content)
        deep_text = deep_match.group(1).strip() if deep_match else content

        save_ai_cache(cache_key, {
            "deep_report": sanitize_text(deep_text),
            "stop_price": stop_val,
            "target_price": target_val,
            "pyramid_price": pyramid_val if pyramid_type else None,
            "pyramid_type": pyramid_type,
            "score_info": score_info
        })
        return sanitize_text(deep_text), stop_val, target_val, (pyramid_val if pyramid_type else None), pyramid_type, score_info, now_str
    except Exception as e:
        print(f"⚠️ {stock_name} 마이 대시보드 AI 가이드 실패: {e}")
        return f"🚨 AI 분석 오류 발생: {e}", None, None, None, None, {"score": 0, "label": "관망 🟡", "p_reason": "-", "t_reason": "-", "m_reason": "-"}, now_str

def analyze_saveticker_macro_and_assets(news_text, force_refresh=False):
    cache_key = "SAVETICKER_INTEL_REPORT"
    if not force_refresh and is_cache_valid(cache_key, max_hours=3):
        return ai_cache_store[cache_key]

    if not llm_mgr.is_available():
        if cache_key in ai_cache_store: return ai_cache_store[cache_key]
        return {
            "verdict": "중립 🟡", "score": 0, "core_drivers": ["AI 데이터 수집 지연으로 기본 캐시를 유지합니다."],
            "asset_hedges": [{"name": "QQQ (나스닥 100)", "ticker": "QQQ", "type": "지수 ETF", "score": 70, "p_reason": "20일선 지지", "reason": "안정적 지수 추종 대응", "strategy": "분할 매수"}],
            "continuation_stocks": [{"name": "NVIDIA", "ticker": "NVDA", "score": 80, "p_reason": "AI 반도체 수요 견고", "reason": "공급 우위 지속", "strategy": "20일선 눌림목 지지 시 진입"}]
        }

    prompt = f"""
너는 글로벌 매크로 헤지펀드의 수석 전략가이다. 아래 세이브티커(SaveTicker) 실시간 뉴스들을 종합 분석하라.
[수집된 최신 속보 및 시황]
{news_text}
[규칙]
1. 하락장 판단 시 무리하게 개별주를 사지 말고 반드시 인버스 상품(SQQQ 등)을 추천할 것.
2. 각 상품별로 -100~+100점의 스코어 및 가격 매력도 근거 1줄 작성.
[출력 양식: 마크다운 없이 순수 JSON만 출력]
{{
  "verdict": "강한 상승 🟢",
  "score": 50,
  "core_drivers": ["동인1", "동인2", "동인3"],
  "asset_hedges": [{{"name": "상품명", "ticker": "SQQQ", "type": "인버스", "score": 75, "p_reason": "손익비 평가 1줄", "reason": "근거", "strategy": "전략"}}],
  "continuation_stocks": [{{"name": "기업명", "ticker": "NVDA", "score": 85, "p_reason": "손익비 평가 1줄", "reason": "근거", "strategy": "전략"}}]
}}
"""
    try:
        raw_res = llm_mgr.generate_completion(prompt, temperature=0.2, max_tokens=1800)
        match_json = re.search(r'(\{[\s\S]*\})', raw_res)
        json_clean = match_json.group(1) if match_json else raw_res
        data = json.loads(json_clean)
        save_ai_cache(cache_key, data)
        return data
    except Exception as e:
        print(f"⚠️ SaveTicker AI 분석 파싱 오류 ({e}) -> 기본값 설정")
        fallback_data = {
            "verdict": "중립 🟡", "score": 5, "core_drivers": ["거시 불확실성 지속 및 섹터별 차별화 장세 진행"],
            "asset_hedges": [
                {"name": "Invesco QQQ Trust", "ticker": "QQQ", "type": "나스닥 100", "score": 65, "p_reason": "20일선 지지선 분할 진입 시 손익비 양호", "reason": "주요 빅테크 실적 방어력 유효", "strategy": "20일선 지지선 분할 매수"},
                {"name": "SPDR Gold Shares", "ticker": "GLD", "type": "금 안전자산", "score": 75, "p_reason": "박스권 하단 지지 확인", "reason": "지정학적 리스크 헷지", "strategy": "박스권 하단 매수"}
            ],
            "continuation_stocks": [
                {"name": "NVIDIA", "ticker": "NVDA", "score": 82, "p_reason": "단기 급등 후 눌림목 형성", "reason": "AI 수요 지속", "strategy": "눌림목 진입 대기"}
            ]
        }
        return fallback_data

def verify_asset_chart(ticker_symbol):
    try:
        df = yf.Ticker(ticker_symbol).history(period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 20: return None
        last_close = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else last_close
        daily_chg = ((last_close - prev_close) / prev_close) * 100.0
        ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
        rsi_series, _ = calculate_wilder_rsi(df['Close'], period=14, signal_period=9)
        rsi_val = round(float(rsi_series.iloc[-1]), 1)
        return {
            "last_close": last_close, "daily_chg": daily_chg, "ma20": ma20, "rsi": rsi_val,
            "is_overheated": (daily_chg >= 9.0) or (rsi_val >= 78.0), "above_ma20": (last_close >= ma20)
        }
    except Exception:
        return None

# =========================================================
# 🎯 토스 잔고 조회
# =========================================================
def get_toss_holdings():
    if not TOSS_CLIENT_ID or not TOSS_CLIENT_SECRET:
        print("⚠️ Toss API Key 미설정 ➔ 샘플 데이터 사용")
        return get_mock_holdings(), False

    try:
        token_url = "https://openapi.tossinvest.com/oauth2/token"
        token_data = {"grant_type": "client_credentials", "client_id": TOSS_CLIENT_ID, "client_secret": TOSS_CLIENT_SECRET}
        token_res = toss_request_with_proxy_failover("POST", token_url, data=token_data, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=12)
        
        if token_res.status_code == 200:
            access_token = token_res.json().get("access_token")
            base_headers = {"Authorization": f"Bearer {access_token}", "x-api-key": TOSS_CLIENT_ID, "Content-Type": "application/json"}
            acc_res = toss_request_with_proxy_failover("GET", "https://openapi.tossinvest.com/api/v1/accounts", headers=base_headers, timeout=12)
            account_seq = 1
            if acc_res.status_code == 200:
                acc_list = acc_res.json().get("result", [])
                if isinstance(acc_list, list) and len(acc_list) > 0:
                    account_seq = acc_list[0].get("accountSeq", 1)

            holdings_headers = base_headers.copy()
            holdings_headers["X-Tossinvest-Account"] = str(account_seq)
            res = toss_request_with_proxy_failover("GET", "https://openapi.tossinvest.com/api/v1/holdings", headers=holdings_headers, timeout=12)
            if res.status_code == 200:
                result_obj = res.json().get("result", {})
                items = result_obj.get("items", []) if isinstance(result_obj, dict) else []
                holdings = []
                for item in items:
                    raw_sym = str(item.get("symbol") or item.get("stockCode") or "").strip()
                    name = str(item.get("name") or item.get("stockName") or raw_sym).strip()
                    qty = float(item.get("quantity") or item.get("holdingQuantity") or 0)
                    avg_p = float(item.get("averagePurchasePrice") or item.get("avgPrice") or 0)
                    last_p = float(item.get("lastPrice") or 0)
                    mv = item.get("marketValue") or {}
                    eval_amt = float(mv.get("amountAfterCost") or mv.get("amount") or (last_p * qty))
                    pl = item.get("profitLoss") or {}
                    profit_loss = float(pl.get("amountAfterCost") or pl.get("amount") or 0)
                    ret_rate = float(pl.get("rateAfterCost") or pl.get("rate") or 0.0) * 100.0
                    market = str(item.get("marketCountry", "KR")).upper()
                    currency = str(item.get("currency", "KRW")).upper()

                    if avg_p <= 0: profit_loss, ret_rate = 0.0, 0.0
                    if qty > 0:
                        holdings.append({
                            "ticker": raw_sym, "name": name, "avg_price": avg_p, "current_price": last_p,
                            "eval_amount": eval_amt, "profit_loss": profit_loss, "return_pct": ret_rate,
                            "quantity": qty, "market": market, "currency": currency
                        })
                if holdings:
                    print(f"🎉 토스증권 API 연동 성공! 실제 보유 종목 총 {len(holdings)}개 수신 완료")
                    return holdings, True
    except Exception as e:
        print(f"⚠️ 토스 API 호출 오류: {e}")
    return get_mock_holdings(), False

def get_mock_holdings():
    return [
        {"ticker": "005930.KS", "name": "삼성전자", "avg_price": 72000, "current_price": 74500, "eval_amount": 3725000, "profit_loss": 125000, "return_pct": 3.47, "quantity": 50, "market": "KR", "currency": "KRW"},
        {"ticker": "000660.KS", "name": "SK하이닉스", "avg_price": 175000, "current_price": 182000, "eval_amount": 3640000, "profit_loss": 140000, "return_pct": 4.0, "quantity": 20, "market": "KR", "currency": "KRW"},
        {"ticker": "NVDA", "name": "NVIDIA", "avg_price": 115.0, "current_price": 128.5, "eval_amount": 1927.5, "profit_loss": 202.5, "return_pct": 11.74, "quantity": 15, "market": "US", "currency": "USD"},
        {"ticker": "PLTR", "name": "Palantir", "avg_price": 24.5, "current_price": 27.2, "eval_amount": 2720.0, "profit_loss": 270.0, "return_pct": 11.02, "quantity": 100, "market": "US", "currency": "USD"}
    ]

# =========================================================
# 실행 흐름 (국장 -> 미장 -> 마이 대시보드 -> 뉴스 -> GitHub 배포)
# =========================================================
kr_macro = get_kr_macro_data()
us_macro = get_us_macro_data()
usd_krw_rate = get_usd_krw_rate()

# PART 1: 🇰🇷 국장
print("\n" + "="*60)
print("🇰🇷 [PART 1] 한국 증시 스캔 & 복합 스코어 리포트 생성 중...")
print("="*60)
kr_is_open, kr_open_msg = get_market_open_status("KR")
kr_witching_alerts = get_witching_day_alert("KR")
kr_econ_events = get_economic_calendar_events("KR")
kr_emergency, kr_vol_banner, kr_defense_mode, kr_avg_chg = check_market_volatility_trigger("KR")
kr_7d_news = get_naver_7days_news(now_dt)
kr_market_status, kr_sentiment_briefing, kr_briefing_time = analyze_7days_news_sentiment("대한민국 주식시장(국장)", kr_7d_news, force_refresh=kr_emergency)

kr_banner_items = kr_witching_alerts + kr_econ_events
if kr_vol_banner: kr_banner_items.insert(0, kr_vol_banner)
if not kr_is_open: kr_banner_items.insert(0, f"<b>[오늘 휴장일]</b> {kr_open_msg} (시세는 직전 거래일 종가 기준입니다)")
kr_banner_html = f'<div class="event-banner"><div class="event-banner-title">🚨 [시장 변동성 주의] 주요 일정 & 만기일 캘린더</div><div class="event-banner-content">{"<br>".join(kr_banner_items)}</div></div>' if kr_banner_items else ""

kr_needs_refresh = should_refresh_daily_pivot("대한민국 주식시장(국장)") or kr_emergency
kr_selected_cache_key = "SELECTED_KR_TARGETS"

if not kr_needs_refresh and kr_selected_cache_key in ai_cache_store:
    selected_kr_targets = ai_cache_store[kr_selected_cache_key].get("targets", {})
else:
    def get_naver_multi_sise():
        candidate_map = {}
        for biz, sosok in [("dealForeign", "0"), ("dealForeign", "1"), ("dealOrgan", "0"), ("dealOrgan", "1"), ("topAmount", "0"), ("topAmount", "1"), ("fluctuation", "0"), ("fluctuation", "1")]:
            try:
                res = requests.get(f"https://m.stock.naver.com/api/json/sise/siseListJson.nhn?bizType={biz}&sosok={sosok}", headers=headers, timeout=6)
                if res.status_code == 200:
                    for item in res.json().get('result', {}).get('itemList', [])[:25]:
                        name, cd = item.get('nm'), item.get('cd')
                        if name and cd and name not in candidate_map:
                            candidate_map[name] = f"{cd}.KS" if sosok == "0" else f"{cd}.KQ"
            except Exception: pass
        return candidate_map

    raw_kr_candidates = get_naver_multi_sise()
    scored_kr_stocks = []
    for name, symbol in list(raw_kr_candidates.items())[:60]:
        try:
            df_hist = yf.Ticker(symbol).history(period="3mo", interval="1d")
            if df_hist is None or df_hist.empty or len(df_hist) < 25: continue
            last_close = float(df_hist['Close'].iloc[-1])
            if last_close < 1000 or (df_hist['Volume'].tail(5).mean() * last_close) < 10_000_000_000: continue
            candle_range = float(df_hist['High'].iloc[-1]) - float(df_hist['Low'].iloc[-1])
            upper_tail = float(df_hist['High'].iloc[-1]) - max(last_close, float(df_hist['Open'].iloc[-1]))
            if candle_range > 0 and (upper_tail / candle_range) > 0.5: continue

            score = 0
            tag_reasons = []
            if last_close >= df_hist['Close'].rolling(20).mean().iloc[-1]: score += 25
            if df_hist['Close'].rolling(5).mean().iloc[-1] >= df_hist['Close'].rolling(20).mean().iloc[-1] >= df_hist['Close'].rolling(60).mean().iloc[-1]:
                score += 25
                tag_reasons.append("이평선 정배열 안착 🟢")
            vol_surge = (df_hist['Volume'].iloc[-1] / (df_hist['Volume'].tail(20).mean() + 1e-9))
            if vol_surge >= 2.0:
                score += 30
                tag_reasons.append(f"거래량 평소 대비 {vol_surge:.1f}배 급증 🔥")
            elif vol_surge >= 1.3: score += 15
            if last_close > float(df_hist['Open'].iloc[-1]):
                score += 20
                tag_reasons.append("장대양봉 종가 고가 마감 📈")
            feature_str = " / ".join(tag_reasons) if tag_reasons else "연속 추세 우상향 지속주"
            scored_kr_stocks.append((score, name, symbol, feature_str))
        except Exception: continue

    scored_kr_stocks.sort(key=lambda x: x[0], reverse=True)
    selected_kr_targets = {item[1]: (item[2], item[3]) for item in scored_kr_stocks[:5]}
    backup_kr = [("한미반도체", "042700.KS", "AI 반도체 수급 주도주"), ("알테오젠", "196170.KQ", "바이오 플랫폼 기관 순매수"), ("효성중공업", "298040.KS", "전력기기 수혜주"), ("HD현대일렉트릭", "267260.KS", "전력 인프라 성장주"), ("레인보우로보틱스", "277810.KQ", "로봇 모멘텀주")]
    for b_name, b_sym, b_feat in backup_kr:
        if len(selected_kr_targets) >= 5: break
        if b_name not in selected_kr_targets: selected_kr_targets[b_name] = (b_sym, b_feat)
    save_ai_cache(kr_selected_cache_key, {"targets": selected_kr_targets})

stock_cards_kr_html = ""
for stock_name, (symbol, supply_type) in selected_kr_targets.items():
    try:
        pure_code = symbol.split('.')[0]
        df_daily = yf.Ticker(symbol).history(period="1y", interval="1d")
        if df_daily is None or df_daily.empty or len(df_daily) < 1: continue

        badge_html = update_and_get_consecutive_days(symbol, kr_needs_refresh)
        kr_stock_chg = ((df_daily['Close'].iloc[-1] - df_daily['Close'].iloc[-2]) / df_daily['Close'].iloc[-2]) * 100.0 if len(df_daily) >= 2 else 0.0
        defense_badge = get_crash_defense_badge(kr_stock_chg, kr_avg_chg, kr_defense_mode)

        df_daily['MA20'] = df_daily['Close'].rolling(20, min_periods=1).mean()
        df_daily['MA60'] = df_daily['Close'].rolling(60, min_periods=1).mean()
        df_daily['MA120'] = df_daily['Close'].rolling(120, min_periods=1).mean()
        std20 = df_daily['Close'].rolling(20, min_periods=1).std().fillna(0)
        df_daily['BB_Upper'] = df_daily['MA20'] + (std20 * 2)
        df_daily['BB_Lower'] = df_daily['MA20'] - (std20 * 2)

        high9, low9 = df_daily['High'].rolling(9, min_periods=1).max(), df_daily['Low'].rolling(9, min_periods=1).min()
        high26, low26 = df_daily['High'].rolling(26, min_periods=1).max(), df_daily['Low'].rolling(26, min_periods=1).min()
        high52, low52 = df_daily['High'].rolling(52, min_periods=1).max(), df_daily['Low'].rolling(52, min_periods=1).min()
        df_daily['Tenkan'] = (high9 + low9) / 2
        df_daily['Kijun'] = (high26 + low26) / 2
        df_daily['Senkou_A'] = ((df_daily['Tenkan'] + df_daily['Kijun']) / 2).shift(26).fillna(df_daily['Close'])
        df_daily['Senkou_B'] = ((high52 + low52) / 2).shift(26).fillna(df_daily['Close'])

        df_recent120 = df_daily.tail(120).copy()
        try:
            num_bins = min(15, len(df_recent120))
            price_bins = pd.cut(df_recent120['Close'], bins=num_bins)
            vol_by_price = df_recent120.groupby(price_bins, observed=False)['Volume'].sum()
            poc_bin = vol_by_price.idxmax() if not vol_by_price.empty else None
            poc_price = int(poc_bin.mid) if poc_bin is not None and pd.notnull(poc_bin) else int(df_recent120['Close'].mean())
        except Exception:
            poc_price = int(df_recent120['Close'].mean())

        max_120, min_120 = int(df_recent120['High'].max()), int(df_recent120['Low'].min())
        peaks_and_troughs_summary = extract_peaks_and_troughs(df_daily.tail(60), is_krw=True)
        atr_val = calculate_atr(df_daily, period=14)
        df_daily = df_daily.ffill().bfill()
        df_daily['RSI'], df_daily['RSI_Signal'] = calculate_wilder_rsi(df_daily['Close'], period=14, signal_period=9)
        rsi_val, rsi_signal_val = round(float(df_daily['RSI'].iloc[-1]), 2), round(float(df_daily['RSI_Signal'].iloc[-1]), 2)
        rsi_cross_status = "RSI 상향 돌파 및 상승 모멘텀 유지 📈" if rsi_val > rsi_signal_val else ("RSI 데드크로스 경고 🔴" if rsi_val < rsi_signal_val and rsi_val >= 60 else "RSI 조정 📉")

        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_status = "골든크로스 📈" if float(df_daily['MACD'].iloc[-1]) > float(df_daily['Signal'].iloc[-1]) else "데드크로스 📉"

        latest_close = int(df_daily['Close'].iloc[-1])
        ma20_d, ma60_d, ma120_d = int(df_daily['MA20'].iloc[-1]), int(df_daily['MA60'].iloc[-1]), int(df_daily['MA120'].iloc[-1])
        bb_up, bb_low = int(df_daily['BB_Upper'].iloc[-1]), int(df_daily['BB_Lower'].iloc[-1])
        cloud_top = max(int(df_daily['Senkou_A'].iloc[-1]), int(df_daily['Senkou_B'].iloc[-1]))
        short_trend = "단기 상승 추세 📈" if latest_close >= ma20_d else "단기 하락 추세 📉"
        mid_trend = "중기 상승 추세 📈" if latest_close >= ma60_d else "중기 하락 추세 📉"
        bb_status = "상한선 돌파/근접 🚀" if latest_close >= bb_up * 0.99 else ("하한선 근접/지지 🟢" if latest_close <= bb_low * 1.01 else "밴드 내 안정 ⚖️")
        cloud_status = "구름대 위 상승 국면 🟢" if latest_close > cloud_top else "구름대 내부/하단 🟡"

        df_recent15 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15).copy()
        raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:{int(row['Open']):,}원 | High:{int(row['High']):,}원 | Low:{int(row['Low']):,}원 | Close:{int(row['Close']):,}원 | Vol:{int(row['Volume']):,}" for idx, row in df_recent15.iterrows()]
        raw_data_str_15days = "\n".join(raw_lines)
        rsi_status = f"과매수 ({rsi_val}) ⚠️" if rsi_val >= 70 else (f"과매도 ({rsi_val}) 🟢" if rsi_val <= 30 else f"중립 ({rsi_val}) ⚖️")
        ma_status = "정배열 (20>60>120) 🟢" if (ma20_d > ma60_d and ma60_d > ma120_d) else "역배열/혼조세 🔴"

        pick_reason, basic_ai_report, deep_ai_report, ai_prices, score_info, stock_ai_time = generate_ai_stock_analysis(
            stock_name, symbol, kr_7d_news, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, latest_close, ma20_d, ma60_d, ma120_d, atr_val, supply_type, "원", force_refresh=kr_emergency
        )

        buy_price_str = f"{int(round(ai_prices['buy'])):,}원" if ai_prices.get('buy') else "⚠️ 산출 실패"
        stop_loss_str = f"{int(round(ai_prices['stop'])):,}원" if ai_prices.get('stop') else "⚠️ 산출 실패"
        target_price_str = f"{int(round(ai_prices['target1'])):,}원" if ai_prices.get('target1') else "⚠️ 산출 실패"

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
        fig.add_hline(y=poc_price, line_dash="dot", line_color="#facc15", annotation_text=f"POC: {fmt_price(poc_price, True)}", row=1, col=1)
        colors = ['#f87171' if c < o else '#4ade80' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='거래량'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#38bdf8', width=1.2), name='RSI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI_Signal'], line=dict(color='#facc15', width=1.0, dash='dot'), name='RSI Signal'), row=3, col=1)
        fig.update_layout(height=500, margin=dict(l=5, r=5, t=35, b=30), xaxis_rangeslider_visible=False, template="plotly_dark", dragmode=False)
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'scrollZoom': False, 'displayModeBar': False})

        score_color = "#38bdf8" if score_info["score"] >= 70 else ("#4ade80" if score_info["score"] >= 30 else ("#facc15" if score_info["score"] >= -29 else ("#fb923c" if score_info["score"] >= -69 else "#f87171")))
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
                        • <b>가격 매력도:</b> {score_info['p_reason']}<br>• <b>기술 지표:</b> {score_info['t_reason']}<br>• <b>뉴스/수급:</b> {score_info['m_reason']}
                    </div>
                </div>
                <div class="stock-reason-box">💡 <b>선정 이유:</b><br>{pick_reason}</div>
                <div class="report-divider"></div>
                <div class="report-line">• 종가 기준 현재가 : <span class="highlight-val">{fmt_price(latest_close, True)}</span></div>
                <div class="report-line">• 추세 진단 : {short_trend} / {mid_trend}</div>
                <div class="report-line">• 차트 구조 : {bb_status} / {cloud_status}</div>
                <div class="report-line">• 집중 매물대 (POC) : <span class="highlight-val">{fmt_price(poc_price, True)}</span></div>
                <div class="report-line">• RSI / MACD : {rsi_status} / {macd_status}</div>
                <div class="report-line" style="color:#38bdf8; font-weight:bold;">🎯 AI 추천 진입가 : {buy_price_str} <span class="sub-desc">(눌림목 지지 타점)</span></div>
                <div class="report-line text-red">🛑 AI 산출 손절가 : {stop_loss_str} <span class="sub-desc">(ATR 버퍼 검증 지지선)</span></div>
                <div class="report-line text-green">🚀 AI 산출 1차 익절가 : {target_price_str} <span class="sub-desc">(손익비 1:1.5 저항선)</span></div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ AI 핵심 매매 전략 <span class="sub-desc">({stock_ai_time})</span></div>
                <div class="ai-content" style="white-space: pre-line;">{basic_ai_report}</div>
                {f'<details class="deep-report-accordion"><summary class="deep-report-btn">🔍 AI 심도 분석 & 세부 근거 더보기 ▼</summary><div class="deep-report-content" style="white-space: pre-line;">{deep_ai_report}</div></details>' if deep_ai_report else ''}
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e: print(f"🚨 {stock_name} 생성 오류: {e}")

# PART 2: 🇺🇸 미장
print("\n" + "="*60)
print("🇺🇸 [PART 2] 미국 증시 스캔 & 3중 상폐 필터 & 복합 스코어 리포트 생성 중...")
print("="*60)
us_is_open, us_open_msg = get_market_open_status("US")
us_witching_alerts = get_witching_day_alert("US")
us_econ_events = get_economic_calendar_events("US")
us_emergency, us_vol_banner, us_defense_mode, us_avg_chg = check_market_volatility_trigger("US")
us_7d_news = get_yahoo_7days_news()
us_market_status, us_sentiment_briefing, us_briefing_time = analyze_7days_news_sentiment("미국 주식시장(미장)", us_7d_news, force_refresh=us_emergency)

us_banner_items = us_witching_alerts + us_econ_events
if us_vol_banner: us_banner_items.insert(0, us_vol_banner)
if not us_is_open: us_banner_items.insert(0, f"<b>[오늘 휴장일]</b> {us_open_msg} (시세는 직전 거래일 종가 기준입니다)")
us_banner_html = f'<div class="event-banner"><div class="event-banner-title">🚨 [시장 변동성 주의] 주요 일정 & 만기일 캘린더</div><div class="event-banner-content">{"<br>".join(us_banner_items)}</div></div>' if us_banner_items else ""

us_needs_refresh = should_refresh_daily_pivot("미국 주식시장(미장)") or us_emergency
us_selected_cache_key = "SELECTED_US_TARGETS"

if not us_needs_refresh and us_selected_cache_key in ai_cache_store:
    selected_us_targets = ai_cache_store[us_selected_cache_key].get("targets", {})
else:
    def get_us_active_stocks():
        if TEST_MODE: return ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'AMZN', 'GOOGL', 'META', 'AVGO', 'PLTR']
        scanned = []
        for url in ["https://finance.yahoo.com/markets/stocks/most-active/", "https://finance.yahoo.com/markets/stocks/gainers/"]:
            try:
                soup = BeautifulSoup(requests.get(url, headers=headers, timeout=10).text, 'html.parser')
                for a in soup.find_all('a'):
                    href = a.get('href', '')
                    if '/quote/' in href:
                        sym = href.split('/quote/')[1].split('?')[0].split('/')[0].upper()
                        if sym.isalpha() and len(sym) <= 5 and sym not in scanned: scanned.append(sym)
            except Exception: pass
        for b in ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'AMZN', 'GOOGL', 'META', 'AVGO', 'PLTR', 'COST', 'NFLX', 'CRM']:
            if b not in scanned: scanned.append(b)
        return scanned

    selected_us_targets = {}
    for sym in get_us_active_stocks():
        if len(selected_us_targets) >= 10: break
        try:
            tk = yf.Ticker(sym)
            df_check = tk.history(period="5d", interval="1d")
            if df_check is None or df_check.empty or len(df_check) < 2: continue
            if float(df_check['Volume'].iloc[-1]) <= 1000: continue
            if (today_date - df_check.index[-1].date()).days > 7: continue
            if float(df_check['Close'].iloc[-1]) <= 0.5: continue
            info = tk.info
            if info.get('marketCap', 0) >= 10_000_000_000:
                selected_us_targets[info.get('shortName', sym)] = (sym, "🔥 Wall Street 거래대금 상위 및 빅테크/AI 핵심주")
        except Exception: continue
    save_ai_cache(us_selected_cache_key, {"targets": selected_us_targets})

stock_cards_us_html = ""
for stock_name, (symbol, supply_type) in selected_us_targets.items():
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period="1y", interval="1d")
        if df_daily is None or len(df_daily) < 1: continue

        badge_html = update_and_get_consecutive_days(symbol, us_needs_refresh)
        us_stock_chg = ((df_daily['Close'].iloc[-1] - df_daily['Close'].iloc[-2]) / df_daily['Close'].iloc[-2]) * 100.0 if len(df_daily) >= 2 else 0.0
        defense_badge = get_crash_defense_badge(us_stock_chg, us_avg_chg, us_defense_mode)

        df_daily['MA20'] = df_daily['Close'].rolling(20, min_periods=1).mean()
        df_daily['MA60'] = df_daily['Close'].rolling(60, min_periods=1).mean()
        df_daily['MA120'] = df_daily['Close'].rolling(120, min_periods=1).mean()
        std20 = df_daily['Close'].rolling(20, min_periods=1).std().fillna(0)
        df_daily['BB_Upper'] = df_daily['MA20'] + (std20 * 2)
        df_daily['BB_Lower'] = df_daily['MA20'] - (std20 * 2)

        high9, low9 = df_daily['High'].rolling(9, min_periods=1).max(), df_daily['Low'].rolling(9, min_periods=1).min()
        high26, low26 = df_daily['High'].rolling(26, min_periods=1).max(), df_daily['Low'].rolling(26, min_periods=1).min()
        high52, low52 = df_daily['High'].rolling(52, min_periods=1).max(), df_daily['Low'].rolling(52, min_periods=1).min()
        df_daily['Tenkan'] = (high9 + low9) / 2
        df_daily['Kijun'] = (high26 + low26) / 2
        df_daily['Senkou_A'] = ((df_daily['Tenkan'] + df_daily['Kijun']) / 2).shift(26).fillna(df_daily['Close'])
        df_daily['Senkou_B'] = ((high52 + low52) / 2).shift(26).fillna(df_daily['Close'])

        df_recent120 = df_daily.tail(120).copy()
        try:
            num_bins = min(15, len(df_recent120))
            price_bins = pd.cut(df_recent120['Close'], bins=num_bins)
            vol_by_price = df_recent120.groupby(price_bins, observed=False)['Volume'].sum()
            poc_bin = vol_by_price.idxmax() if not vol_by_price.empty else None
            poc_price = round(float(poc_bin.mid), 2) if poc_bin is not None and pd.notnull(poc_bin) else round(float(df_recent120['Close'].mean()), 2)
        except Exception:
            poc_price = round(float(df_recent120['Close'].mean()), 2)

        max_120, min_120 = round(float(df_recent120['High'].max()), 2), round(float(df_recent120['Low'].min()), 2)
        peaks_and_troughs_summary = extract_peaks_and_troughs(df_daily.tail(60), is_krw=False)
        atr_val = calculate_atr(df_daily, period=14)
        df_daily = df_daily.ffill().bfill()
        df_daily['RSI'], df_daily['RSI_Signal'] = calculate_wilder_rsi(df_daily['Close'], period=14, signal_period=9)
        rsi_val, rsi_signal_val = round(float(df_daily['RSI'].iloc[-1]), 2), round(float(df_daily['RSI_Signal'].iloc[-1]), 2)
        rsi_cross_status = "RSI 상향 돌파 및 상승 모멘텀 유지 📈" if rsi_val > rsi_signal_val else ("RSI 데드크로스 경고 🔴" if rsi_val < rsi_signal_val and rsi_val >= 60 else "RSI 조정 📉")

        exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
        df_daily['MACD'] = exp1 - exp2
        df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
        macd_status = "골든크로스 📈" if float(df_daily['MACD'].iloc[-1]) > float(df_daily['Signal'].iloc[-1]) else "데드크로스 📉"

        latest_close = round(float(df_daily['Close'].iloc[-1]), 2)
        ma20_d, ma60_d, ma120_d = round(float(df_daily['MA20'].iloc[-1]), 2), round(float(df_daily['MA60'].iloc[-1]), 2), round(float(df_daily['MA120'].iloc[-1]), 2)
        bb_up, bb_low = round(float(df_daily['BB_Upper'].iloc[-1]), 2), round(float(df_daily['BB_Lower'].iloc[-1]), 2)
        cloud_top = max(round(float(df_daily['Senkou_A'].iloc[-1]), 2), round(float(df_daily['Senkou_B'].iloc[-1]), 2))
        short_trend = "단기 상승 추세 📈" if latest_close >= ma20_d else "단기 하락 추세 📉"
        mid_trend = "중기 상승 추세 📈" if latest_close >= ma60_d else "중기 하락 추세 📉"
        bb_status = "상한선 돌파/근접 🚀" if latest_close >= bb_up * 0.99 else ("하한선 근접/지지 🟢" if latest_close <= bb_low * 1.01 else "밴드 내 안정 ⚖️")
        cloud_status = "구름대 위 상승 국면 🟢" if latest_close > cloud_top else "구름대 내부/하단 🟡"

        df_recent15 = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15).copy()
        raw_lines = [f"{idx.strftime('%Y-%m-%d')} | Open:${row['Open']:.2f} | High:${row['High']:.2f} | Low:${row['Low']:.2f} | Close:${row['Close']:.2f} | Vol:{int(row['Volume']):,}" for idx, row in df_recent15.iterrows()]
        raw_data_str_15days = "\n".join(raw_lines)
        rsi_status = f"과매수 ({rsi_val}) ⚠️" if rsi_val >= 70 else (f"과매도 ({rsi_val}) 🟢" if rsi_val <= 30 else f"중립 ({rsi_val}) ⚖️")
        ma_status = "정배열 (20>60>120) 🟢" if (ma20_d > ma60_d and ma60_d > ma120_d) else "역배열/혼조세 🔴"

        pick_reason, basic_ai_report, deep_ai_report, ai_prices, score_info, stock_ai_time = generate_ai_stock_analysis(
            stock_name, symbol, us_7d_news, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, latest_close, ma20_d, ma60_d, ma120_d, atr_val, supply_type, "$", force_refresh=us_emergency
        )

        buy_price_str = f"${ai_prices['buy']:.2f}" if ai_prices.get('buy') else "⚠️ 산출 실패"
        stop_loss_str = f"${ai_prices['stop']:.2f}" if ai_prices.get('stop') else "⚠️ 산출 실패"
        target_price_str = f"${ai_prices['target1']:.2f}" if ai_prices.get('target1') else "⚠️ 산출 실패"

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
        fig.add_hline(y=poc_price, line_dash="dot", line_color="#facc15", annotation_text=f"POC: {fmt_price(poc_price, False)}", row=1, col=1)
        colors = ['#f87171' if c < o else '#4ade80' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='거래량'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#38bdf8', width=1.2), name='RSI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI_Signal'], line=dict(color='#facc15', width=1.0, dash='dot'), name='RSI Signal'), row=3, col=1)
        fig.update_layout(height=500, margin=dict(l=5, r=5, t=35, b=30), xaxis_rangeslider_visible=False, template="plotly_dark", dragmode=False)
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'scrollZoom': False, 'displayModeBar': False})

        score_color = "#38bdf8" if score_info["score"] >= 70 else ("#4ade80" if score_info["score"] >= 30 else ("#facc15" if score_info["score"] >= -29 else ("#fb923c" if score_info["score"] >= -69 else "#f87171")))
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
                        • <b>가격 매력도:</b> {score_info['p_reason']}<br>• <b>기술 지표:</b> {score_info['t_reason']}<br>• <b>뉴스/수급:</b> {score_info['m_reason']}
                    </div>
                </div>
                <div class="stock-reason-box">💡 <b>선정 이유:</b><br>{pick_reason}</div>
                <div class="report-divider"></div>
                <div class="report-line">• 종가 기준 현재가 : <span class="highlight-val">{fmt_price(latest_close, False)}</span></div>
                <div class="report-line">• 추세 진단 : {short_trend} / {mid_trend}</div>
                <div class="report-line">• 차트 구조 : {bb_status} / {cloud_status}</div>
                <div class="report-line">• 집중 매물대 (POC) : <span class="highlight-val">{fmt_price(poc_price, False)}</span></div>
                <div class="report-line">• RSI / MACD : {rsi_status} / {macd_status}</div>
                <div class="report-line" style="color:#38bdf8; font-weight:bold;">🎯 AI 추천 진입가 : {buy_price_str} <span class="sub-desc">(눌림목 지지 타점)</span></div>
                <div class="report-line text-red">🛑 AI 산출 손절가 : {stop_loss_str} <span class="sub-desc">(ATR 버퍼 검증 지지선)</span></div>
                <div class="report-line text-green">🚀 AI 산출 1차 익절가 : {target_price_str} <span class="sub-desc">(손익비 1:1.5 저항선)</span></div>
            </div>
            <div class="ai-opinion-box">
                <div class="ai-title">⚡ AI 핵심 매매 전략 <span class="sub-desc">({stock_ai_time})</span></div>
                <div class="ai-content" style="white-space: pre-line;">{basic_ai_report}</div>
                {f'<details class="deep-report-accordion"><summary class="deep-report-btn">🔍 AI 심도 분석 & 세부 근거 더보기 ▼</summary><div class="deep-report-content" style="white-space: pre-line;">{deep_ai_report}</div></details>' if deep_ai_report else ''}
            </div>
            <div class="chart-container">{chart_html}</div>
        </div>
        """
    except Exception as e: print(f"🚨 {stock_name} 생성 오류: {e}")

# PART 3: 🎯 마이 대시보드
print("\n" + "="*60)
print("🎯 [PART 3] 토스 실계좌 잔고 직결 & 포지션 파워 스코어 & 8단계 라벨 보존...")
print("="*60)
toss_holdings, is_real_toss = get_toss_holdings()

if is_real_toss:
    currently_held_symbols = set(h['ticker'] for h in toss_holdings)
    deleted_cache_count = 0
    for key in list(ai_cache_store.keys()):
        if key.startswith("TOSS_MY_") and key.replace("TOSS_MY_", "") not in currently_held_symbols:
            del ai_cache_store[key]
            deleted_cache_count += 1
    if deleted_cache_count > 0: save_entire_cache(ai_cache_store)

my_stock_cards_html = ""
total_eval_my = 0.0
total_profit_my = 0.0

for h in toss_holdings:
    try:
        ticker, stock_name, avg_price, quantity = h['ticker'], h['name'], h['avg_price'], h['quantity']
        current_price, eval_amount_raw, profit_loss_raw, return_pct = h['current_price'], h['eval_amount'], h['profit_loss'], h['return_pct']
        pure_code = ticker.split('.')[0]
        is_krw = True if (h['market'] == 'KR' or h['currency'] == 'KRW') else False
        fx = usd_krw_rate if not is_krw else 1.0
        eval_amount_krw, profit_loss_krw = eval_amount_raw * fx, profit_loss_raw * fx
        emergency_flag = kr_emergency if is_krw else us_emergency

        yf_ticker = f"{pure_code}.KS" if (is_krw and not ticker.endswith((".KS", ".KQ"))) else pure_code
        df_daily = None
        try:
            df_daily = yf.Ticker(yf_ticker).history(period="1y", interval="1d")
            if (df_daily is None or df_daily.empty or len(df_daily) < 1) and is_krw:
                df_daily = yf.Ticker(f"{pure_code}.KQ").history(period="1y", interval="1d")
        except Exception: df_daily = None

        if df_daily is None or df_daily.empty or len(df_daily) == 0:
            short_trend, mid_trend = "단기 추세 분석 중 ⚖️", "중기 추세 분석 중 ⚖️"
            bb_status, cloud_status = "밴드 안정 ⚖️", "구름대 유효 🟢"
            poc_price, max_120, min_120 = current_price, current_price, current_price
            rsi_status, macd_status = "중립 (50) ⚖️", "중립 ⚖️"
            ma_status = "이평선 안정화 진행 중 ⚖️"
            rsi_val, rsi_signal_val, rsi_cross_status = 50.0, 50.0, "모멘텀 안정"
            peaks_and_troughs_summary = "마디점 안정화 진행 중"
            raw_data_str_15days = f"최신 토스 현재가: {fmt_price(current_price, is_krw)}"
            defense_badge = ""
        else:
            my_stock_chg = ((df_daily['Close'].iloc[-1] - df_daily['Close'].iloc[-2]) / df_daily['Close'].iloc[-2]) * 100.0 if len(df_daily) >= 2 else 0.0
            defense_badge = get_crash_defense_badge(my_stock_chg, (kr_avg_chg if is_krw else us_avg_chg), (kr_defense_mode if is_krw else us_defense_mode))

            df_daily['MA20'] = df_daily['Close'].rolling(20, min_periods=1).mean()
            df_daily['MA60'] = df_daily['Close'].rolling(60, min_periods=1).mean()
            df_daily['MA120'] = df_daily['Close'].rolling(120, min_periods=1).mean()
            std20 = df_daily['Close'].rolling(20, min_periods=1).std().fillna(0)
            df_daily['BB_Upper'] = df_daily['MA20'] + (std20 * 2)
            df_daily['BB_Lower'] = df_daily['MA20'] - (std20 * 2)

            df_recent120 = df_daily.tail(120).copy()
            try:
                num_bins = min(15, len(df_recent120))
                price_bins = pd.cut(df_recent120['Close'], bins=num_bins)
                vol_by_price = df_recent120.groupby(price_bins, observed=False)['Volume'].sum()
                poc_bin = vol_by_price.idxmax() if not vol_by_price.empty else None
                poc_price = float(poc_bin.mid) if poc_bin is not None and pd.notnull(poc_bin) else float(df_recent120['Close'].mean())
            except Exception:
                poc_price = float(df_recent120['Close'].mean())

            max_120, min_120 = float(df_recent120['High'].max()), float(df_recent120['Low'].min())
            peaks_and_troughs_summary = extract_peaks_and_troughs(df_daily.tail(60), is_krw=is_krw)
            df_daily = df_daily.ffill().bfill()
            df_daily['RSI'], df_daily['RSI_Signal'] = calculate_wilder_rsi(df_daily['Close'], period=14, signal_period=9)
            rsi_val, rsi_signal_val = round(float(df_daily['RSI'].iloc[-1]), 2), round(float(df_daily['RSI_Signal'].iloc[-1]), 2)
            rsi_cross_status = "RSI 상향 돌파 📈" if rsi_val > rsi_signal_val else ("RSI 데드크로스 🔴" if rsi_val < rsi_signal_val and rsi_val >= 60 else "RSI 조정 📉")

            exp1 = df_daily['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df_daily['Close'].ewm(span=26, adjust=False).mean()
            df_daily['MACD'] = exp1 - exp2
            df_daily['Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
            macd_status = "골든크로스 📈" if float(df_daily['MACD'].iloc[-1]) > float(df_daily['Signal'].iloc[-1]) else "데드크로스 📉"

            ma20_d, ma60_d = float(df_daily['MA20'].iloc[-1]), float(df_daily['MA60'].iloc[-1])
            bb_up, bb_low = float(df_daily['BB_Upper'].iloc[-1]), float(df_daily['BB_Lower'].iloc[-1])
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
            rsi_status = f"과매수 ({fmt_num(rsi_val)}) ⚠️" if rsi_val >= 70 else (f"과매도 ({fmt_num(rsi_val)}) 🟢" if rsi_val <= 30 else f"중립 ({rsi_val}) ⚖️")
            ma_status = "정배열 지지 🟢" if current_price >= ma20_d else "역배열/혼조세 🔴"

        deep_ai_guide, my_stop_val, my_target_val, my_pyramid_val, my_pyramid_type, my_score_info, my_guide_time = generate_ai_toss_3line_analysis(
            stock_name, ticker, avg_price, current_price, return_pct, raw_data_str_15days, rsi_val, rsi_signal_val, rsi_cross_status, macd_status, ma_status, bb_status, cloud_status, poc_price, max_120, min_120, peaks_and_troughs_summary, is_krw, force_refresh=emergency_flag
        )

        eval_formatted, profit_formatted = f"{int(round(eval_amount_krw)):,}원", f"({profit_loss_krw:+,.0f}원)"
        avg_price_formatted = fmt_price(avg_price, is_krw, show_decimal=is_krw) if avg_price > 0 else "0원 (상장폐지/청산대기)"
        current_price_formatted = fmt_price(current_price, is_krw, show_decimal=not is_krw)
        poc_formatted = fmt_price(poc_price, is_krw, show_decimal=not is_krw)
        my_stop_str = fmt_price(my_stop_val, is_krw, show_decimal=not is_krw) if my_stop_val else "⚠️ 산출 실패"
        my_target_str = fmt_price(my_target_val, is_krw, show_decimal=not is_krw) if my_target_val else "⚠️ 산출 실패"

        pyramid_row_html = f'<div class="report-line" style="color:#38bdf8; font-weight:bold;">🎯 AI 추천 추매가({my_pyramid_type}) : {fmt_price(my_pyramid_val, is_krw, show_decimal=not is_krw)} <span class="sub-desc">(눌림목/반등 타점)</span></div>' if (my_pyramid_val and my_pyramid_type) else ""
        country_badge = "🇰🇷" if is_krw else "🇺🇸"
        my_score_color = "#38bdf8" if my_score_info["score"] >= 70 else ("#4ade80" if my_score_info["score"] >= 30 else ("#facc15" if my_score_info["score"] >= -29 else ("#fb923c" if my_score_info["score"] >= -69 else "#f87171")))

        my_stock_cards_html += f"""
        <div class="card">
            <div class="console-report">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div class="report-header">{country_badge} {stock_name} ({pure_code}) - {fmt_num(quantity)}주 {defense_badge}</div>
                    <a href="https://www.tradingview.com/symbols/{f'KRX-{pure_code}' if is_krw else ticker}/" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
                </div>
                <div style="font-size:20px; font-weight:bold; margin-top:6px; color:#f8fafc;">
                    {eval_formatted} <span class="{'text-green' if profit_loss_krw>=0 else 'text-red'}">{profit_formatted}</span>
                </div>
                <div style="background:#111827; border:1px solid #374151; border-radius:8px; padding:10px 14px; margin:10px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:13px; color:#94a3b8; font-weight:bold;">🎯 포지션 파워 스코어</span>
                        <span style="font-size:15px; font-weight:bold; color:{my_score_color};">{my_score_info['score']:+d}점 [{my_score_info['label']}]</span>
                    </div>
                    <div style="font-size:12px; color:#cbd5e1; line-height:1.6; border-top:1px dashed #374151; padding-top:6px;">
                        • <b>가격 매력도:</b> {my_score_info['p_reason']}<br>• <b>저항/지지:</b> {my_score_info['t_reason']}<br>• <b>대응 권고:</b> {my_score_info['m_reason']}
                    </div>
                </div>
                <div class="report-divider"></div>
                <div class="report-line">주당 평단가 : <span class="highlight-val">{avg_price_formatted}</span> (<span class="{'text-green' if return_pct>=0 else 'text-red'}">{return_pct:+.2f}%</span>) &nbsp;&nbsp;|&nbsp;&nbsp; 주당 현재가 : <span class="highlight-val">{current_price_formatted}</span></div>
                <div class="report-line">추세 진단 : {short_trend} &nbsp;&nbsp;|&nbsp;&nbsp; {mid_trend}</div>
                <div class="report-line">차트 구조 : {bb_status} &nbsp;&nbsp;|&nbsp;&nbsp; {cloud_status}</div>
                <div class="report-line">집중 매물대 (POC) : <span class="highlight-val">{poc_formatted}</span></div>
                <div class="report-line">RSI / MACD : {rsi_status} / {macd_status}</div>
                {pyramid_row_html}
                <div class="report-line text-red">🛑 AI 권장 손절선(Trailing) : {my_stop_str}</div>
                <div class="report-line text-green">🚀 AI 동적 목표가 : {my_target_str}</div>
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
        total_eval_my += eval_amount_krw
        total_profit_my += profit_loss_krw
    except Exception as e:
        print(f"⚠️ {h.get('name', '종목')} 처리 중 예외 발생: {e}")

total_cost_my = total_eval_my - total_profit_my
total_return_pct_my = (total_profit_my / total_cost_my * 100) if total_cost_my > 0 else 0

# PART 4: 📰 뉴스 인텔리전스
print("\n" + "="*60)
print("📰 [PART 4] SaveTicker 뉴스 인텔리전스 & 거시/ETF/원자재/주도주 선별 중...")
print("="*60)
saveticker_news_list = scrape_saveticker_news()
news_digest_text = "\n".join([f"[{item['time']}] {item['title']} - {item['desc']}" for item in saveticker_news_list])
saveticker_intel = analyze_saveticker_macro_and_assets(news_digest_text, force_refresh=us_emergency)

asset_cards_html = ""
for item in saveticker_intel.get("asset_hedges", [])[:4]:
    sym = item.get("ticker", "").upper()
    tech = verify_asset_chart(sym)
    tech_html = ""
    status_tag = '<span class="badge-item" style="background:#15803d;">진입 유효 🟢</span>'
    if tech:
        if tech["is_overheated"]: status_tag = '<span class="badge-item" style="background:#dc2626;">단기 과열 경고 (추격 금지) ⚠️</span>'
        elif tech["above_ma20"]: status_tag = '<span class="badge-item" style="background:#2563eb;">20일선 위 추세 유지 📈</span>'
        tech_html = f'<div class="report-line">• 현재가: <span class="highlight-val">${tech["last_close"]:.2f}</span> ({tech["daily_chg"]:+.2f}%) &nbsp;|&nbsp; 20일선: ${tech["ma20"]:.2f} &nbsp;|&nbsp; Wilder RSI: {tech["rsi"]}</div>'

    score_val = item.get('score', 70)
    score_color = "#38bdf8" if score_val >= 70 else ("#4ade80" if score_val >= 30 else "#facc15")
    asset_cards_html += f"""
    <div class="card">
        <div class="console-report">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div class="report-header">{item.get('name')} ({sym}) - <span style="color:#a855f7;">{item.get('type')}</span> {status_tag}</div>
                <a href="https://www.tradingview.com/symbols/{sym}/" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
            </div>
            <div style="background:#111827; border:1px solid #374151; border-radius:8px; padding:10px 14px; margin:10px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-size:13px; color:#94a3b8; font-weight:bold;">🎯 뉴스 헷지 파워 스코어</span>
                    <span style="font-size:15px; font-weight:bold; color:{score_color};">{score_val:+d}점 [헷지 진입 유효 🔵]</span>
                </div>
                <div style="font-size:12px; color:#cbd5e1; line-height:1.6; border-top:1px dashed #374151; padding-top:6px;">
                    • <b>가격 매력도:</b> {item.get('p_reason', '지지선 기반 손익비 양호')}<br>• <b>재료 지속성:</b> {item.get('reason')}
                </div>
            </div>
            <div class="stock-reason-box">💡 <b>선정 근거 & 매크로 트리거:</b><br>{item.get('reason')}</div>
            <div class="report-divider"></div>
            {tech_html}
            <div class="report-line" style="color:#38bdf8; font-weight:bold;">🎯 실전 대응/헷지 가이드 : {item.get('strategy')}</div>
        </div>
    </div>
    """

stock_picks_html = ""
for item in saveticker_intel.get("continuation_stocks", [])[:4]:
    sym = item.get("ticker", "").upper()
    tech = verify_asset_chart(sym)
    tech_html = ""
    status_tag = '<span class="badge-item" style="background:#15803d;">추세 지속 모멘텀 🚀</span>'
    if tech:
        if tech["is_overheated"]: status_tag = '<span class="badge-item" style="background:#b91c1c;">단기 1차 급등 반영 (눌림목 대기) 🟠</span>'
        elif not tech["above_ma20"]: status_tag = '<span class="badge-item" style="background:#64748b;">이평선 회복 확인 필요 🟡</span>'
        tech_html = f'<div class="report-line">• 현재가: <span class="highlight-val">${tech["last_close"]:.2f}</span> ({tech["daily_chg"]:+.2f}%) &nbsp;|&nbsp; 20일선: ${tech["ma20"]:.2f} &nbsp;|&nbsp; Wilder RSI: {tech["rsi"]}</div>'

    score_val = item.get('score', 80)
    score_color = "#38bdf8" if score_val >= 70 else ("#4ade80" if score_val >= 30 else "#facc15")
    stock_picks_html += f"""
    <div class="card">
        <div class="console-report">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div class="report-header">{item.get('name')} ({sym}) {status_tag}</div>
                <a href="https://www.tradingview.com/symbols/{sym}/" target="_blank" class="tv-link-btn">📈 TradingView 차트 ↗</a>
            </div>
            <div style="background:#111827; border:1px solid #374151; border-radius:8px; padding:10px 14px; margin:10px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-size:13px; color:#94a3b8; font-weight:bold;">🎯 뉴스 모멘텀 스코어</span>
                    <span style="font-size:15px; font-weight:bold; color:{score_color};">{score_val:+d}점 [추세 지속 유망 🟢]</span>
                </div>
                <div style="font-size:12px; color:#cbd5e1; line-height:1.6; border-top:1px dashed #374151; padding-top:6px;">
                    • <b>가격 매력도:</b> {item.get('p_reason', '눌림목 지지선 형성')}<br>• <b>재료 지속성:</b> {item.get('reason')}
                </div>
            </div>
            <div class="stock-reason-box">💡 <b>뉴스 지속성 평가:</b><br>{item.get('reason')}</div>
            <div class="report-divider"></div>
            {tech_html}
            <div class="report-line text-green">🎯 시간차 방어 진입 가이드 : {item.get('strategy')}</div>
        </div>
    </div>
    """

news_feed_rows = [f'<div style="margin-bottom:12px; border-bottom:1px dashed #334155; padding-bottom:8px;"><span style="color:#38bdf8; font-size:12px; font-weight:bold;">[{n["time"]}]</span> <b>{n["title"]}</b><div style="color:#94a3b8; font-size:13px; margin-top:3px;">{n["desc"]}</div></div>' for n in saveticker_news_list]
news_feed_html = "".join(news_feed_rows)
drivers_html = "<br>".join([f"&nbsp;&nbsp;• {d}" for d in saveticker_intel.get("core_drivers", [])])

# PART 5: 페이지 빌드
macro_html_kr = render_macro_grid(kr_macro, is_krw=True)
macro_html_us = render_macro_grid(us_macro, is_krw=False)

full_html_kr = build_kr_page(now_str, kr_open_msg, kr_market_status, kr_banner_html, macro_html_kr, kr_briefing_time, kr_sentiment_briefing, stock_cards_kr_html)
full_html_us = build_us_page(now_str, us_open_msg, us_market_status, us_banner_html, macro_html_us, us_briefing_time, us_sentiment_briefing, stock_cards_us_html)
full_html_my = build_my_page(now_str, fmt_price(total_eval_my, True), total_profit_my, total_return_pct_my, usd_krw_rate, my_stock_cards_html)
full_html_news = build_news_page(now_str, saveticker_intel, drivers_html, asset_cards_html, stock_picks_html, saveticker_news_list, news_feed_html)

# PART 6: GitHub Pages 배포
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

print("\n🌐 [PART 6] GitHub Pages (index, us_index, index3, index4) 및 AI 캐시 동기화 업로드 중...")
try:
    if not GITHUB_TOKEN:
        raise ValueError("GH_TOKEN이 설정되지 않았습니다.")

    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO_NAME)
    
    upload_to_github_safely(repo, "index.html", f"Deploy KR Report: {now_str}", full_html_kr)
    upload_to_github_safely(repo, "us_index.html", f"Deploy US Report: {now_str}", full_html_us)
    upload_to_github_safely(repo, "index3.html", f"Deploy My Dashboard: {now_str}", full_html_my)
    upload_to_github_safely(repo, "index4.html", f"Deploy News Intelligence: {now_str}", full_html_news)
    
    if os.path.exists(CACHE_FILE_NAME):
        with open(CACHE_FILE_NAME, "r", encoding="utf-8") as f:
            cache_json_str = f.read()
        upload_to_github_safely(repo, "ai_cache.json", f"Update AI Cache: {now_str}", cache_json_str)

    print("\n" + "="*65)
    print("🎉 [최종 완료] 3단계 AI Failover (Gemini 1/2 -> Groq 1/2) 동기화 배포 완료!")
    print(f"🔗 🇰🇷 국장: https://{repo.owner.login}.github.io/{repo.name}/index.html")
    print(f"🔗 🇺🇸 미장: https://{repo.owner.login}.github.io/{repo.name}/us_index.html")
    print(f"🔗 🎯 마이: https://{repo.owner.login}.github.io/{repo.name}/index3.html")
    print(f"🔗 📰 뉴스: https://{repo.owner.login}.github.io/{repo.name}/index4.html")
    print("="*65)

except Exception as e:
    print(f"🚨 GitHub 연결 과정 중 오류 발생: {e}")
