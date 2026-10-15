# html_templates.py

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
    .macro-card { 
        background: #182232; 
        border: 1px solid #334155; 
        border-radius: 10px; 
        padding: 14px 12px; 
        text-align: center; 
        text-decoration: none; 
        color: inherit; 
        display: block;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }
    .macro-card:hover { transform: translateY(-3px); border-color: #38bdf8; box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2); }
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
    .tv-link-btn { background: #2563eb; color: #ffffff; padding: 4px 9px; border-radius: 4px; text-decoration: none; font-size: 12px; font-weight: bold; display: inline-block; }
    .sub-desc { font-size: 12px; color: #94a3b8; font-weight: normal; }
    .badge-item { color: #ffffff; font-size: 11.5px; padding: 2px 7px; border-radius: 12px; margin-left: 6px; font-weight: bold; display: inline-block; }
    
    .ai-opinion-box { background: #062a1c; border: 1px solid #22c55e; border-radius: 8px; padding: 16px; margin-top: 14px; }
    .ai-title { font-size: 14.5px; font-weight: bold; color: #4ade80; margin-bottom: 8px; }
    .ai-content { font-size: 14px; color: #f1f5f9; line-height: 1.7; }
    
    .deep-report-accordion { margin-top: 12px; border-top: 1px dashed rgba(34, 197, 94, 0.4); padding-top: 10px; }
    .deep-report-btn { 
        background: #0d4a32; 
        color: #86efac; 
        padding: 9px 12px; 
        border-radius: 6px; 
        font-size: 13px; 
        font-weight: bold; 
        cursor: pointer; 
        user-select: none;
        list-style: none;
        text-align: center;
        border: 1px solid #15803d;
        transition: background 0.2s;
    }
    .deep-report-btn:hover { background: #15803d; }
    .deep-report-btn::-webkit-details-marker { display: none; }
    .deep-report-content { 
        background: #041f15; 
        border-radius: 6px; 
        padding: 13px; 
        margin-top: 8px; 
        font-size: 13.5px; 
        color: #e2e8f0; 
        line-height: 1.75; 
        border: 1px solid #166534; 
    }

    .chart-container { margin-top: 16px; border-radius: 8px; overflow: hidden; }

    #btn-back-to-top {
        position: fixed;
        bottom: 25px;
        right: 25px;
        display: none;
        width: 46px;
        height: 46px;
        border-radius: 50%;
        background: #2563eb;
        color: #ffffff;
        border: none;
        outline: none;
        cursor: pointer;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        z-index: 9999;
        transition: opacity 0.3s, transform 0.2s, background 0.2s;
        align-items: center;
        justify-content: center;
    }
    #btn-back-to-top:hover {
        background: #1d4ed8;
        transform: translateY(-3px);
    }

    @media (max-width: 768px) {
        body { padding: 10px 8px; }
        .container { width: 100%; }
        .nav-bar { gap: 5px; margin-bottom: 12px; }
        .nav-btn { flex: 1 1 45%; text-align: center; font-size: 13px; padding: 9px 2px; }
        
        .header h1 { font-size: 19px; }
        .header p { font-size: 13.5px !important; }
        
        .macro-grid { grid-template-columns: 1fr; gap: 6px; }
        .macro-title { font-size: 13.5px; }
        .macro-value { font-size: 20px; }
        .macro-sub { font-size: 12.5px; }
        
        .card { padding: 14px 10px; margin-bottom: 20px; }
        .report-header { font-size: 17.5px; }
        .console-report { font-size: 15px; line-height: 1.75; padding: 14px 12px; }
        
        #btn-back-to-top {
            bottom: 20px;
            right: 18px;
            width: 42px;
            height: 42px;
            font-size: 18px;
        }
    }
</style>
"""

top_button_component = """
<button id="btn-back-to-top" title="최상단으로 이동">▲</button>
<script>
    const topBtn = document.getElementById("btn-back-to-top");
    window.addEventListener("scroll", () => {
        if (window.scrollY > 300) {
            topBtn.style.display = "flex";
        } else {
            topBtn.style.display = "none";
        }
    });
    topBtn.addEventListener("click", () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
    });
</script>
"""

def render_macro_grid(macro_data, is_krw=True):
    currency_title = "원화 통화량 (M2)" if is_krw else "달러 통화량 (US M2)"
    cli_title = "한국 경기선행지수 (CLI)" if is_krw else "미국 경기선행지수 (CLI)"
    vix_title = "한국 VKOSPI" if is_krw else "미국 VIX 지수"
    source_sub = "네이버 금융 원본 연동" if is_krw else "TradingView 원본 연동"

    return f"""
<div class="macro-grid">
    <a href="{macro_data['m2_url']}" target="_blank" class="macro-card">
        <div class="macro-title">💵 {currency_title} ↗</div>
        <div class="macro-value">{macro_data['m2']}</div>
        <div class="macro-sub">{macro_data['m2_date']}</div>
    </a>
    <a href="{macro_data['cli_url']}" target="_blank" class="macro-card">
        <div class="macro-title">🌐 {cli_title} ↗</div>
        <div class="macro-value">{macro_data['cli']}</div>
        <div class="macro-sub">{macro_data['cli_date']}</div>
    </a>
    <a href="{macro_data['vix_url']}" target="_blank" class="macro-card">
        <div class="macro-title">⚡ {vix_title} ↗</div>
        <div class="macro-value" style="color:#facc15;">{macro_data['vix']}</div>
        <div class="macro-sub">{source_sub}</div>
    </a>
</div>
"""

def build_kr_page(now_str, kr_open_msg, kr_market_status, kr_banner_html, macro_html_kr, kr_briefing_time, kr_sentiment_briefing, stock_cards_kr_html):
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>🇰🇷 AI 국장 분석 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-active">🇰🇷 국장 대시보드</a><a href="us_index.html" class="nav-btn btn-inactive">🇺🇸 미장 대시보드</a><a href="index3.html" class="nav-btn btn-inactive">🎯 마이 대시보드</a><a href="index4.html" class="nav-btn btn-inactive">📰 뉴스 인텔리전스</a></div><div class="header"><h1>📊 AI 국장 주도주 대시보드 <span style="font-size:17px;">[{kr_market_status}]</span></h1><p style="margin:0; color:#94a3b8; font-size:13px;">상태: {kr_open_msg} | 업데이트: {now_str}</p></div>{kr_banner_html}{macro_html_kr}<div class="news-briefing-card"><div class="news-title">📰 [최근 7일간 뉴스 AI 종합 분석 브리핑] <span style="font-size:11.5px; color:#94a3b8; font-weight:normal;">({kr_briefing_time})</span></div>{kr_sentiment_briefing}</div>{stock_cards_kr_html}</div>{top_button_component}</body></html>"""

def build_us_page(now_str, us_open_msg, us_market_status, us_banner_html, macro_html_us, us_briefing_time, us_sentiment_briefing, stock_cards_us_html):
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>🇺🇸 AI 미장 분석 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-inactive">🇰🇷 국장 대시보드</a><a href="us_index.html" class="nav-btn btn-active">🇺🇸 미장 대시보드</a><a href="index3.html" class="nav-btn btn-inactive">🎯 마이 대시보드</a><a href="index4.html" class="nav-btn btn-inactive">📰 뉴스 인텔리전스</a></div><div class="header"><h1>🇺🇸 AI US Stock 주도주 대시보드 <span style="font-size:17px;">[{us_market_status}]</span></h1><p style="margin:0; color:#94a3b8; font-size:13px;">상태: {us_open_msg} | 업데이트: {now_str}</p></div>{us_banner_html}{macro_html_us}<div class="news-briefing-card"><div class="news-title">📰 [최근 7일간 뉴스 AI 종합 분석 브리핑] <span style="font-size:11.5px; color:#94a3b8; font-weight:normal;">({us_briefing_time})</span></div>{us_sentiment_briefing}</div>{stock_cards_us_html}</div>{top_button_component}</body></html>"""

def build_my_page(now_str, total_eval_str, total_profit_my, total_return_pct_my, usd_krw_rate, my_stock_cards_html):
    profit_color = '#f87171' if total_profit_my >= 0 else '#60a5fa'
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>🎯 마이 포트폴리오 대시보드</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-inactive">🇰🇷 국장 대시보드</a><a href="us_index.html" class="nav-btn btn-inactive">🇺🇸 미장 대시보드</a><a href="index3.html" class="nav-btn btn-active">🎯 마이 대시보드</a><a href="index4.html" class="nav-btn btn-inactive">📰 뉴스 인텔리전스</a></div><div class="header"><h1>🎯 마이 포트폴리오 실계좌 대시보드</h1><p style="margin:0; color:#94a3b8; font-size:13px;">업데이트: {now_str}</p></div><div style="background:linear-gradient(135deg, #1e293b, #334155); padding:16px; border-radius:12px; margin-bottom:22px; display:flex; justify-content:space-around; text-align:center; border:1px solid #334155; flex-wrap:wrap; gap:8px;"><div><div style="font-size:0.75rem; color:#94a3b8;">총 평가 금액 (원화)</div><div style="font-size:1.4rem; font-weight:bold; margin-top:3px; color:#f8fafc;">{total_eval_str}</div><div style="font-size:0.75rem; color:#60a5fa; margin-top:2px;">환율: {usd_krw_rate:,.1f}원</div></div><div><div style="font-size:0.75rem; color:#94a3b8;">총 평가 손익</div><div style="font-size:1.4rem; font-weight:bold; margin-top:3px; color:{profit_color};">{total_profit_my:+,.0f}원</div></div><div><div style="font-size:0.75rem; color:#94a3b8;">전체 수익률</div><div style="font-size:1.4rem; font-weight:bold; margin-top:3px; color:{profit_color};">{total_return_pct_my:+.2f}%</div></div></div><h2 style="font-size:1.15rem; color:#38bdf8; margin-bottom:12px;">📊 토스증권 연동 보유 종목 정밀 분석 & AI 가이드</h2>{my_stock_cards_html}</div>{top_button_component}</body></html>"""

def build_news_page(now_str, saveticker_intel, drivers_html, asset_cards_html, stock_picks_html, saveticker_news_list, news_feed_html):
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>📰 SaveTicker 뉴스 인텔리전스</title>{html_style}</head><body><div class="container"><div class="nav-bar"><a href="index.html" class="nav-btn btn-inactive">🇰🇷 국장 대시보드</a><a href="us_index.html" class="nav-btn btn-inactive">🇺🇸 미장 대시보드</a><a href="index3.html" class="nav-btn btn-inactive">🎯 마이 대시보드</a><a href="index4.html" class="nav-btn btn-active">📰 뉴스 인텔리전스</a></div><div class="header"><h1>📰 SaveTicker 뉴스 인텔리전스 대시보드</h1><p style="margin:0; color:#94a3b8; font-size:13px;">거시 시장 방향성 / 다중 자산(ETF·원자재·인버스) / 실전 주도주 | 업데이트: {now_str}</p></div><div class="card"><div class="console-report"><div class="report-header">🧭 글로벌 거시 시장 종합 나침반 : <span class="highlight-val">{saveticker_intel.get('verdict')}</span> (센티먼트 점수: {saveticker_intel.get('score', 0):+d}점)</div><div class="report-divider"></div><div class="report-line"><b>📌 시장을 움직이는 핵심 매크로 요인:</b><br>{drivers_html}</div></div></div><h2 style="font-size:1.15rem; color:#38bdf8; margin-bottom:12px;">🛡️ 시장 대응 맞춤 자산 (지수 추종 / 인버스 헷지 / 원자재 ETF)</h2>{asset_cards_html}<h2 style="font-size:1.15rem; color:#4ade80; margin-bottom:12px;">🚀 뉴스 모멘텀 지속 유망주 (시간차 방어 및 눌림목)</h2>{stock_picks_html}<details class="deep-report-accordion" style="margin-top:20px;"><summary class="deep-report-btn">📋 SaveTicker 수집 원문 뉴스 피드 전체보기 ({len(saveticker_news_list)}건) ▼</summary><div class="deep-report-content">{news_feed_html}</div></details></div>{top_button_component}</body></html>"""
