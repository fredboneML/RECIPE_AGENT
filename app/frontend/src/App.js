import { useNavigate } from 'react-router-dom';
import React, { useState, useEffect, useRef } from 'react';

import './App.css';

function App() {
  const [isAuthChecking, setIsAuthChecking] = useState(true);
  const [conversations, setConversations] = useState([]);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState('');
  const [userInitial, setUserInitial] = useState('');
  const [showPopup, setShowPopup] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRequestCanceled, setIsRequestCanceled] = useState(false);
  const [categories, setCategories] = useState({});
  const [isLoadingQuestions, setIsLoadingQuestions] = useState(true);
  
  const navigate = useNavigate();
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);
  
  const backendUrl = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000'
    : `http://${window.location.hostname}:8000`;

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: "smooth", 
        block: "end" 
      });
    }
  };

  // Add effect to scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);


  useEffect(() => {
    const verifyAuth = async () => {
      try {
        console.log('=== App.js - Authentication Check Start ===');
        const userName = localStorage.getItem('userName');
        const tenantCode = localStorage.getItem('tenantCode');
  
        console.log('App.js - Checking stored credentials:', { userName, tenantCode });
  
        if (!userName || !tenantCode) {
          console.warn('App.js - Missing credentials, redirecting to login');
          localStorage.clear();
          navigate('/login', { replace: true });
          return;
        }
  
        setUserInitial(userName.charAt(0).toUpperCase());
        
        console.log('App.js - Fetching conversations...');
        const conversationsResponse = await fetch(`${backendUrl}/api/conversations`, {
          headers: {
            'Content-Type': 'application/json',
            'X-Tenant-Code': tenantCode,
            'Accept': 'application/json',
          }
        });
  
        if (!conversationsResponse.ok) {
          throw new Error('Auth verification failed');
        }
  
        const conversationsData = await conversationsResponse.json();
        setConversations(conversationsData);
        await fetchInitialQuestions();
  
      } catch (error) {
        console.error('App.js - Auth verification failed:', error);
        localStorage.clear();
        navigate('/login', { replace: true });
      } finally {
        setIsAuthChecking(false);
      }
    };
  
    verifyAuth();
  }, [navigate]); // Only depend on navigate
  
  
  //  Keep the textarea height adjustment useEffect separate
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [query]);



// In App.js, update the API call functions to include tenant code header:
const getHeaders = () => ({
  'Content-Type': 'application/json',
  'X-Tenant-Code': localStorage.getItem('tenantCode'),
  'Accept': 'application/json',
});

const fetchInitialQuestions = async () => {
  try {
    const response = await fetch(`${backendUrl}/api/initial-questions`, {
      headers: getHeaders()
    });
    
    if (response.ok) {
      const data = await response.json();
      if (data.success) {
        setCategories(data.categories);
      }
    }
  } catch (error) {
    console.error('Error fetching initial questions:', error);
  } finally {
    setIsLoadingQuestions(false);
  }
};


  const fetchConversations = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/conversations`, {
        headers: getHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setConversations(data);
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
      if (error.message === 'Unauthorized') {
        localStorage.removeItem('userName');
        localStorage.removeItem('tenantCode');
        navigate('/login');
      }
    }
  };

  const fetchConversationMessages = async (conversationId) => {
    try {
      const response = await fetch(`${backendUrl}/api/conversations/${conversationId}`, {
        headers: getHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setMessages(data);
        setTimeout(scrollToBottom, 100); // Add scroll after loading messages
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  
  const handleSubmit = async () => {
    if (!query.trim() || isProcessing) return;
    
    setIsProcessing(true);
    
    abortControllerRef.current = new AbortController();
    
    try {
      const response = await fetch(`${backendUrl}/api/query`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({ 
          query,
          conversation_id: currentConversation?.id 
        }),
        signal: abortControllerRef.current.signal
      });
  
      if (!response.ok) {
        throw new Error('Failed to submit query');
      }
  
      const data = await response.json();
      
      if (!isRequestCanceled) {
        if (data.error) {
          const errorMessage = {
            query,
            error: data.error,
            reformulated_question: data.reformulated_question,
            followup_questions: data.followup_questions || [],
            timestamp: new Date()
          };
          console.log("Adding error message to UI:", errorMessage);
          setMessages(prev => [...prev, errorMessage]);
        } else {
          const responseText = data.result || data.response || "";
          setResult(responseText);
          const newMessage = {
            query,
            response: responseText,
            followup_questions: data.followup_questions || [],
            timestamp: new Date()
          };
          
          console.log("Adding new message to UI:", newMessage);
          setMessages(prev => [...prev, newMessage]);
          
          if (!currentConversation) {
            setCurrentConversation({ id: data.conversation_id });
            fetchConversations();
          }
          
          // Force scroll after a small delay to ensure rendering is complete
          setTimeout(scrollToBottom, 100);
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request was canceled');
      } else {
        console.error('Error submitting query:', error);
        if (error.message === 'Unauthorized') {
          localStorage.removeItem('userName');
          localStorage.removeItem('tenantCode');
          navigate('/login');
        }
      }
    } finally {
      if (!isRequestCanceled) {
        setIsProcessing(false);
      }
      setIsRequestCanceled(false);
      setQuery('');
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
    localStorage.clear(); // Clear all storage instead of individual items
    navigate('/login', { replace: true });
  };

  const handleNewConversation = () => {
    setCurrentConversation(null);
    setMessages([]);
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
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-summary ${currentConversation?.id === conv.id ? 'active' : ''}`}
              onClick={() => {
                setCurrentConversation(conv);
                fetchConversationMessages(conv.id);
              }}
            >
              <div className="conversation-title">{conv.title}</div>
              <div className="conversation-date">
                {new Date(conv.timestamp).toLocaleDateString()}
              </div>
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
          {!currentConversation && !messages.length && (
            <>
              <img src="/logo.png" alt="Company Logo" className="large-logo" />
              {isLoadingQuestions ? (
                <div className="loading-questions">Loading suggestions...</div>
              ) : (
                <div className="question-categories">
                  {Object.entries(categories).map(([category, data]) => (
                    <div key={category} className="category-section">
                      <h3 className="category-title">{category}</h3>
                      <p className="category-description">{data.description}</p>
                      <div className="common-questions">
                        {data.questions.map((question, index) => (
                          <div
                            key={index}
                            className="question-box"
                            onClick={() => handleQuestionClick(question)}
                          >
                            {question}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          <div className="messages-container">
            {messages.map((message, index) => (
              <div key={index} className="message-group">
                {/* User message */}
                {message.query && (
                  <div className="message user">
                    <div className="message-content user-message">
                      {message.query}
                    </div>
                  </div>
                )}
                
                {/* Assistant response */}
                {message.response && (
                  <div className="message assistant">
                    <div className="message-content assistant-message">
                      {message.response}
                    </div>
                    {message.followup_questions && message.followup_questions.length > 0 && (
                      <div className="followup-suggestions">
                        <h4>Follow-up Questions:</h4>
                        <div className="common-questions">
                          {message.followup_questions.map((question, idx) => (
                            <div
                              key={idx}
                              className="question-box"
                              onClick={() => handleQuestionClick(question)}
                            >
                              {question}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
                
                {/* Error message with reformulated question */}
                {message.error && (
                  <div className="error-suggestion">
                    <h4>Query Suggestion:</h4>
                    {message.reformulated_question && (
                      <div
                        className="question-box"
                        onClick={() => handleQuestionClick(message.reformulated_question)}
                      >
                        {message.reformulated_question}
                      </div>
                    )}
                    {message.followup_questions && message.followup_questions.length > 0 && (
                      <div className="followup-suggestions">
                        <h4>Try these instead:</h4>
                        <div className="common-questions">
                          {message.followup_questions.map((question, idx) => (
                            <div
                              key={idx}
                              className="question-box"
                              onClick={() => handleQuestionClick(question)}
                            >
                              {question}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
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
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;