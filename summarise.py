import requests
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from dateutil.parser import parse

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

def get_db_connection():
    try:
        conn = psycopg2.connect(
            user="avnadmin",
            password="AVNS_BjR949P9WbH1581lKaa",
            host="pg-3261b9de-shashupreethims-9126.l.aivencloud.com",
            port=22737,
            database="defaultdb",
            sslmode='require'
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        return None

def summarize_conversation(conversation_text):
    """Creates a concise conversation summary using Groq's current models."""
    try:
        if not conversation_text or len(conversation_text.strip()) < 10:
            return "Not enough conversation data to summarize."

        print(f"Creating concise summary (length: {len(conversation_text)} characters)...")

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # Updated to use a current Groq model (as of March 2025)
        # Replace "llama3-70b-8192" with the recommended model from Groq docs
        payload = {
            "model": "llama3-70b-8192",  # Updated model name
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant that creates brief, personal summaries of conversations. Use 'you' perspective when describing the conversation."
                },
                {
                    "role": "user",
                    "content": f"Summarize this conversation from my perspective:\n\n{conversation_text}\n\nKeep it detailed and highlight:\n1. Main topic and key points\n2. Any specific details or numbers mentioned\n3. mention the names mentioned\n 4. No bad words should be mentioned"
                }
            ],
            "temperature": 0.3,
            "max_tokens": 300,
            "top_p": 0.8,
            "stream": False
        }

        print(f"Payload being sent to Groq API: {payload}")
        print("Sending request to Groq API...")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        # Debug information
        print(f"API Response Status: {response.status_code}")
        print(f"API Response Headers: {response.headers}")
        
        if response.status_code != 200:
            try:
                error_details = response.json()
                print(f"API Error Details: {error_details}")
            except ValueError:
                print(f"API Error Response: {response.text}")
            return f"Failed to generate summary. Status code: {response.status_code}"
            
        result = response.json()
        if "choices" in result and result["choices"]:
            summary_text = result["choices"][0].get("message", {}).get("content", "")
            if summary_text:
                return summary_text.strip()
        
        return "Could not generate summary from API response."
            
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {str(e)}")
        return f"Failed to get summary from API: {str(e)}"
    except Exception as e:
        print(f"Error in summarize_conversation: {str(e)}")
        return f"Summary generation failed: {str(e)}"

@app.route('/api/summarize-conversation', methods=['GET'])
def summarize_conversation_endpoint():
    app_user_id = request.args.get('app_user_id')
    other_user_id = request.args.get('other_user_id')
    date_str = request.args.get('date')

    print(f"Fetching conversations for date: {date_str}")

    if not all([app_user_id, other_user_id, date_str]):
        return jsonify({
            'error': 'Missing required parameters (app_user_id, other_user_id, date)',
            'success': False
        }), 400

    try:
        # Parse target date
        target_date = parse(date_str).date()
        today = datetime.now().date()
        
        if target_date > today:
            return jsonify({
                'error': 'Cannot fetch conversations from future dates',
                'success': False
            }), 400

        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed', 'success': False}), 500

        try:
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            # Get all conversations between these users
            cursor.execute(
                """
                SELECT id, conversation_data, recorded_at
                FROM conversations
                WHERE app_user_id = %s AND other_user_id = %s
                ORDER BY recorded_at ASC
                """,
                (app_user_id, other_user_id)
            )

            conversations = cursor.fetchall()
            full_conversation_text = []
            latest_conversation_id = None
            original_messages = []
            
            for row in conversations:
                if row['conversation_data'] and isinstance(row['conversation_data'], dict):
                    messages = row['conversation_data'].get('messages', [])
                    for message in messages:
                        # Get message timestamp and parse it
                        msg_timestamp = parse(message.get('timestamp', ''))
                        msg_date = msg_timestamp.date()
                        
                        # Only process messages from the target date
                        if msg_date == target_date:
                            latest_conversation_id = row['id']
                            speaker = message.get('speaker', 'Unknown')
                            text = message.get('text', '')
                            
                            if text:
                                formatted_message = f"[{msg_timestamp.isoformat()}] {speaker}: {text}"
                                full_conversation_text.append(formatted_message)
                                original_messages.append({
                                    'speaker': speaker,
                                    'text': text,
                                    'timestamp': msg_timestamp.isoformat()
                                })

            if not full_conversation_text:
                return jsonify({
                    'success': False,
                    'summary': f'No conversations found for {target_date}',
                    'conversation_count': 0,
                    'date': target_date.isoformat()
                })

            conversation_text = "\n".join(full_conversation_text)
            summary = summarize_conversation(conversation_text)
            
            return jsonify({
                'success': True,
                'summary': summary,
                'conversation_length': len(conversation_text),
                'conversation_id': latest_conversation_id,
                'total_messages': len(full_conversation_text),
                'original_messages': original_messages,
                'date': target_date.isoformat(),
                'conversation_count': len(original_messages)
            })

        finally:
            if cursor:
                cursor.close()

    except ValueError as ve:
        return jsonify({
            'error': f'Invalid date format. Please use YYYY-MM-DD: {str(ve)}',
            'success': False
        }), 400
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'error': f'Failed to process request: {str(e)}',
            'success': False
        }), 500
    finally:
        if conn:
            conn.close()

def test_groq_api():
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Updated to use a current Groq model (as of March 2025)
    test_payload = {
        "model": "llama3-70b-8192",  # Updated model name
        "messages": [{"role": "user", "content": "Test message"}]
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=test_payload,
            headers=headers,
            timeout=10
        )

        print(f"Test API Status: {response.status_code}")
        print(f"Test API Response: {response.text}")
        if response.status_code != 200:
            try:
                error_details = response.json()
                print(f"Test API Error Details: {error_details}")
            except ValueError:
                print(f"Test API Error Response: {response.text}")
            return False
        return True

    except Exception as e:
        print(f"Test API Error: {e}")
        return False

# Call this function before your app.run()
test_groq_api()

if __name__ == '__main__':
    app.run(debug=True, port=5002)