from datetime import datetime




def get_today_str() -> str:
    """Get current date in a human-readable format. / 현재 날짜를 사람이 읽기 쉬운 형태로 반환합니다."""
    return datetime.now().strftime("%a %b %-d, %Y")  
