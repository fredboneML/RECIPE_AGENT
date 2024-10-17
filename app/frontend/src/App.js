import { useNavigate } from 'react-router-dom';
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([]);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState('');
  const [userInitial, setUserInitial] = useState('U');
  const [showPopup, setShowPopup] = useState(false);
  const navigate = useNavigate();
  const textareaRef = useRef(null);
  const commonQuestions = [
    "Top 10 topics",
    "Overall sentiment trend",
    "Companies with highest average positive sentiment",
    "Companies showing an increasing trend in negative call sentiments"
  ];

  useEffect(() => {
    fetch('/api/get_conversations')
      .then(response => response.json())
      .then(data => setConversations(data));
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [query]);

  const handleSubmit = () => {
    fetch('/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    })
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          console.error("Query error:", data.message);
        } else {
          setResult(data.result);
        }
      });
  };

  const handleQuestionClick = (question) => {
    setQuery(question);
  };

  const handleSignOff = () => {
    navigate('/login');
  };

  return (
    <div className="app">
      <div className="sidebar">
        <img src="/logo.png" alt="Company Logo" className="small-logo" />
        <div className="new-conversation">
          <button>New Conversation</button>
        </div>
        <div className="conversations">
          {conversations.map((conv, index) => (
            <div key={index} className="conversation-summary">
              {conv.summary}
            </div>
          ))}
        </div>
      </div>

      <div className="content">
        <div className="top-bar">
          <div className="user-initial" onClick={() => setShowPopup(!showPopup)}>
            <button>{userInitial}</button>
          </div>
          {showPopup && (
            <div className="popup">
              <button onClick={handleSignOff}>Sign Off</button>
            </div>
          )}
        </div>

        <div className="landing">
          <img src="/logo.png" alt="Company Logo" className="large-logo" />
          <div className="common-questions">
            {commonQuestions.map((question, index) => (
              <div
                key={index}
                className="question-box"
                onClick={() => handleQuestionClick(question)}
              >
                {question}
              </div>
            ))}
          </div>
          <div className="query-input">
            <textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask your question..."
            />
            <button onClick={handleSubmit}>Send</button>
          </div>
          {result && (
            <div className="query-result">
              <h3>Your Question:</h3>
              <p>{query}</p>
              <h3>Result:</h3>
              <p>{result}</p>
              <button onClick={() => alert("Ask a follow-up question!")}>Follow Up?</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
