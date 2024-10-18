from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3  # You can replace this with your preferred database (PostgreSQL)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Function to get past conversations (simulating with static data for now)
def get_past_conversations():
    return [
        {'summary': 'Senior Data Scientist Application'},
        {'summary': '3D Body Scan Avatar'},
        {'summary': 'Depth Measurement Estimation'},
        {'summary': 'Fashion Try-On Tool'},
        {'summary': 'Database for Call Insights'}
    ]

# Landing page route
@app.route('/')
def landing_page():
    # Display the 5 commonly asked questions
    common_questions = [
        "How do I check inventory?",
        "Show me the sales data for the last month",
        "What is the most popular product?",
        "What are the recent order statistics?",
        "Show the customer feedback"
    ]

    user_initial = session.get('user_initial', 'U')
    return render_template('index.html', common_questions=common_questions, past_conversations=get_past_conversations(), user_initial=user_initial)

# Handle query submission
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    
    # Here you would translate the question to an SQL query
    sql_query = translate_to_sql(question)  # Placeholder function
    result = run_sql_query(sql_query)  # Placeholder for DB call
    
    # Store question and result in the database (table for memory)
    store_question_answer(question, result)
    
    return render_template('results.html', question=question, result=result)

# Redirect to sign-in page on user click
@app.route('/sign-in')
def sign_in():
    return redirect(url_for('sign_in_page'))

# Placeholder function for SQL translation
def translate_to_sql(question):
    # Implement the logic to translate natural language to SQL
    return f"SELECT * FROM example_table WHERE question='{question}'"

# Placeholder for running the SQL query
def run_sql_query(query):
    # Example function to run SQL query
    return "Query Result"

# Memory saving logic
def store_question_answer(question, answer):
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS conversation_memory 
                      (id INTEGER PRIMARY KEY, question TEXT, answer TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute("INSERT INTO conversation_memory (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    app.run(debug=True, port=5001)
