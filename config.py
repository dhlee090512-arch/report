# config.py
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
import datetime
import warnings

warnings.filterwarnings('ignore')

TEST_MODE = False

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
# 🌐 [오라클 프록시 설정 적용]
# =========================================================
PROXY_POOL = [
    "http://proxy:password@161.33.159.78:3128"
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
                print(f"⚠️ 프록시 #{i+1} ({p_url.split('@')[-1]}) 대역폭 소진 (402) -> 다음 프록시로 페일오버")
                continue
            return res
        except Exception as e:
            print(f"⚠️ 프록시 #{i+1} ({p_url.split('@')[-1]}) 통신 실패: {e}")
            continue

    print("⚠️ 모든 프록시 실패 -> 로컬/클라우드 직접 연결 시도")
    if method.upper() == "POST":
        return requests.post(url, headers=headers_req, data=data_req, timeout=timeout_req)
    else:
        return requests.get(url, headers=headers_req, timeout=timeout_req)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Chrome/120.0.0.0)',
    'Referer': 'https://finance.naver.com/'
}

kst_timezone = datetime.timezone(datetime.timedelta(hours=9))
now_dt = datetime.datetime.now(kst_timezone)
now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")
today_date = now_dt.date()

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
        print(f"⚠️ 전체 캐시 파일 저장 실패: {e}")

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
        elapsed_hours = (now_dt - cached_dt).total_seconds() / 3600.0
        return elapsed_hours < max_hours
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
        target_hour, target_minute = (8, 30) if ("국장" in market_type or "한국" in market_type) else (22, 0)
        today_pivot = now_dt.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if now_dt >= today_pivot:
            return cached_dt < today_pivot
        else:
            yesterday_pivot = today_pivot - datetime.timedelta(days=1)
            return cached_dt < yesterday_pivot
    except Exception as e:
        print(f"⚠️ 기준시점 판별 예외 ({e}) -> 신규 생성 진행")
        return True
