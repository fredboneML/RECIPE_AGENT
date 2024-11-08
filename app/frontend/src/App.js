import { useNavigate } from 'react-router-dom';
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([]);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState('');
  const [userInitial, setUserInitial] = useState('');
  const [showPopup, setShowPopup] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRequestCanceled, setIsRequestCanceled] = useState(false);
  const navigate = useNavigate();
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);
  
  // Dynamically determine the backend URL
  const backendUrl = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000'
    : `http://${window.location.hostname}:8000`;

  const commonQuestions = [
    "Top 10 topics",
    "How does the sentiment of calls last week compare to the previous week?",
    "Top 10 topics leading to higher customer satisfaction",
    "Top 10 Companies showing an increasing trend in negative call sentiments"
  ];

  useEffect(() => {
    const token = localStorage.getItem('token');
    const userName = localStorage.getItem('userName');
    if (!token) {
      navigate('/login');
    } else {
      if (userName) {
        setUserInitial(userName.charAt(0).toUpperCase());
      }
      fetch(`${backendUrl}/api/get_conversations`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept': 'application/json',
        }
      })
        .then(response => {
          if (!response.ok) {
            throw new Error('Failed to fetch conversations');
          }
          return response.json();
        })
        .then(data => setConversations(data))
        .catch(error => {
          console.error('Error fetching conversations:', error);
          if (error.message === 'Unauthorized') {
            localStorage.removeItem('token');
            localStorage.removeItem('userName');
            navigate('/login');
          }
        });
    }
  }, [navigate, backendUrl]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [query]);

  const handleSubmit = async () => {
    if (!query.trim() || isProcessing) return;
    
    setIsProcessing(true);
    setResult('');
    const token = localStorage.getItem('token');
    
    // Create new AbortController
    abortControllerRef.current = new AbortController();
    
    try {
      const response = await fetch(`${backendUrl}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          'Accept': 'application/json',
        },
        body: JSON.stringify({ query }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error('Failed to submit query');
      }

      const data = await response.json();
      
      if (!isRequestCanceled) {
        if (data.error) {
          console.error("Query error:", data.message);
        } else {
          setResult(data.result);
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request was canceled');
      } else {
        console.error('Error submitting query:', error);
        if (error.message === 'Unauthorized') {
          localStorage.removeItem('token');
          localStorage.removeItem('userName');
          navigate('/login');
        }
      }
    } finally {
      if (!isRequestCanceled) {
        setIsProcessing(false);
      }
      setIsRequestCanceled(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleStopRequest = () => {
    if (abortControllerRef.current) {
      setIsRequestCanceled(true);
      abortControllerRef.current.abort();
      setIsProcessing(false);
    }
  };

  const handleQuestionClick = (question) => {
    setQuery(question);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const handleSignOff = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('userName');
    navigate('/login');
  };

  const handleNewConversation = () => {
    setQuery('');
    setResult('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  return (
    <div className="app">
      <div className="sidebar">
        <img src="/logo.png" alt="Company Logo" className="small-logo" />
        <div className="new-conversation">
          <button onClick={handleNewConversation}>New Conversation</button>
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

          {isProcessing && (
            <div className="processing-message">
              Processing your request...
            </div>
          )}

          <div className="input-container">
            <div className="query-input">
              <textarea
                ref={textareaRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask your question... (Press Enter to send, Shift+Enter for new line)"
              />
              <button 
                onClick={isProcessing ? handleStopRequest : handleSubmit}
                className={isProcessing ? 'stop-button' : ''}
              >
                {isProcessing ? 'Stop' : 'Send'}
              </button>
            </div>

            {result && (
              <div className="query-result">
                <h3>Your Question:</h3>
                <p>{query}</p>
                <h3>Result:</h3>
                <pre>{typeof result === 'object' ? JSON.stringify(result, null, 4) : result}</pre>
                <button onClick={() => alert("Ask a follow-up question!")}>Follow Up?</button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;