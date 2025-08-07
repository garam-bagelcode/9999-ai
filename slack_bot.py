# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Slack Bot 예제
봇을 태그(@bot_name)했을 때 메시지를 출력하는 기능
"""

import os
import time
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from RagQuery import answer_query
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# 로깅 설정 개선 - DEBUG 레벨로 변경
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Slack 앱 초기화
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# 모든 이벤트를 캐치하는 미들웨어 (디버깅용)
@app.middleware
def log_all_events(body, next):
    """모든 이벤트를 로그에 기록하는 미들웨어"""
    print(f"🔍 이벤트 수신: {body.get('type', 'unknown')}")
    if 'event' in body:
        event_type = body['event'].get('type', 'unknown')
        print(f"   └─ 이벤트 타입: {event_type}")
        if 'text' in body['event']:
            text = body['event']['text'][:100]
            print(f"   └─ 텍스트: {text}")
    logger.debug(f"Raw event received: {body}")
    next()

@app.event("app_mention")
def handle_app_mention_events(body, logger):
    """
    봇이 태그(@bot_name)되었을 때 실행되는 함수
    """
    try:
        print("🎯 app_mention 핸들러 실행됨!")
        
        # 이벤트 정보 추출
        event = body["event"]
        user_id = event["user"]
        channel_id = event["channel"]
        text = event["text"]
        timestamp = event["ts"]
        
        # 메시지 정보 출력
        print("=" * 50)
        print("🤖 슬랙봇이 태그되었습니다!")
        print(f"📅 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(timestamp)))}")
        print(f"👤 사용자 ID: {user_id}")
        print(f"📍 채널 ID: {channel_id}")
        print(f"💬 메시지: {text}")
        print("=" * 50)
        
        # 로거에도 기록
        logger.info(f"Bot mentioned by {user_id} in {channel_id}: {text}")
        
        # 응답 메시지 보내기 CUSTOM
        answer, docs = answer_query(text, "faiss_index")
        
        app.client.chat_postMessage(
            channel=channel_id,
            text=f"{answer}"
        )
        
    except Exception as e:
        print(f"❌ 메시지 처리 중 오류 발생: {e}")
        logger.error(f"Error handling app mention: {e}")

# 모든 메시지 감지 (디버깅용) - 항상 활성화
@app.event("message")
def handle_message_events(body, logger):
    """
    모든 메시지를 감지하는 함수 (디버깅용)
    """
    try:
        print("📨 message 이벤트 핸들러 실행됨!")
        event = body["event"]
        
        # 이벤트 상세 정보 출력
        print(f"   ├─ 채널: {event.get('channel', 'unknown')}")
        print(f"   ├─ 사용자: {event.get('user', 'unknown')}")
        print(f"   ├─ 봇ID: {event.get('bot_id', 'none')}")
        print(f"   └─ 텍스트: {event.get('text', 'no text')[:100]}")
        
        if "text" in event and "user" in event:
            # 봇 자신의 메시지는 제외
            if event.get("bot_id"):
                print("   └─ 봇 메시지이므로 무시")
                return
            
            print(f"📨 새 메시지 감지: {event['text'][:50]}{'...' if len(event['text']) > 50 else ''}")
            logger.info(f"Message received in channel {event.get('channel')}: {event['text'][:100]}")
            
            # 멘션이 포함되어 있는지 확인
            if f"<@{app.client.auth_test()['user_id']}>" in event['text']:
                answer, docs = answer_query(text, "faiss_index")
                app.client.chat_postMessage(
                    channel=channel_id,
                    text=f"{answer.encode("utf-8").decode("utf-8")}"
                )
        
    except Exception as e:
        print(f"❌ 메시지 이벤트 처리 중 오류: {e}")
        logger.error(f"Error handling message event: {e}")

def main():
    """
    슬랙봇 실행 함수
    """
    print("🚀 슬랙봇을 시작합니다...")
    print("🔧 설정 확인:")
    print(f"   - Bot Token: {'✅ 설정됨' if os.environ.get('SLACK_BOT_TOKEN') else '❌ 없음'}")
    print(f"   - App Token: {'✅ 설정됨' if os.environ.get('SLACK_APP_TOKEN') else '❌ 없음'}")
    
    # 봇 정보 확인
    try:
        from slack_sdk import WebClient
        client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
        auth_info = client.auth_test()
        print(f"   - 봇 이름: {auth_info['user']}")
        print(f"   - 봇 ID: {auth_info['user_id']}")
    except Exception as e:
        print(f"   - 봇 정보 확인 실패: {e}")
    
    print("📋 기능:")
    print("   - 봇을 태그(@bot_name)하면 메시지가 출력됩니다")
    print("   - 'hello'가 포함된 메시지에 반응합니다")
    print("   - 모든 메시지가 로그에 기록됩니다 (디버깅용)")
    print("🛑 종료하려면 Ctrl+C를 누르세요.")
    print("=" * 60)
    
    try:
        # Socket Mode로 봇 실행
        handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
        print("🔌 Socket Mode 연결 중...")
        print("📡 이벤트 대기 중... (아무 메시지나 보내보세요)")
        handler.start()
    except KeyboardInterrupt:
        print("\n🛑 슬랙봇을 종료합니다.")
    except Exception as e:
        print(f"❌ 슬랙봇 실행 중 오류: {e}")
        logger.error(f"Bot startup error: {e}")

if __name__ == "__main__":
    # 환경변수 확인
    if not os.environ.get("SLACK_BOT_TOKEN"):
        print("❌ SLACK_BOT_TOKEN 환경변수가 설정되지 않았습니다.")
        print("📝 .env 파일에 다음을 추가하세요:")
        print("SLACK_BOT_TOKEN=xoxb-your-bot-token")
        print("SLACK_APP_TOKEN=xapp-your-app-token")
    elif not os.environ.get("SLACK_APP_TOKEN"):
        print("❌ SLACK_APP_TOKEN 환경변수가 설정되지 않았습니다.")
        print("📝 .env 파일에 다음을 추가하세요:")
        print("SLACK_BOT_TOKEN=xoxb-your-bot-token")
        print("SLACK_APP_TOKEN=xapp-your-app-token")
    else:
        main()