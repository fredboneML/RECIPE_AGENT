import { useNavigate } from 'react-router-dom';
import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import tokenManager from './tokenManager';

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
  const [uploadedDocument, setUploadedDocument] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  
  const navigate = useNavigate();
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  
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
        
        // First verify health check
        console.log('App.js - Checking health status...');
        const healthResponse = await tokenManager.get('/health');
        if (!healthResponse.ok) {
          throw new Error('Health check failed');
        }
        
        console.log('App.js - Fetching conversations...');
        const conversationsResponse = await tokenManager.get('/api/conversations');
  
        if (!conversationsResponse.ok) {
          throw new Error('Auth verification failed');
        }
  
        const conversationsData = await conversationsResponse.json();
        setConversations(conversationsData);
  
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

  useEffect(() => {
    tokenManager.scheduleRefresh();
  }, []);

  // Replace getHeaders with tokenManager logic
  const getHeaders = () => ({
    'Content-Type': 'application/json',
    'X-Tenant-Code': localStorage.getItem('tenantCode'),
    'Accept': 'application/json',
    'Authorization': `Bearer ${localStorage.getItem('token')}`,
  });

  const fetchConversations = async () => {
    try {
      const response = await tokenManager.get('/api/conversations');
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
      const response = await tokenManager.get(`/api/conversations/${conversationId}`);
      if (response.ok) {
        const data = await response.json();
        setMessages(data);
        setTimeout(scrollToBottom, 100);
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
      // Combine query with uploaded document text if available
      let finalQuery = query;
      if (uploadedDocument && uploadedDocument.extractedText) {
        finalQuery = `${query}\n\n[Extracted from document: ${uploadedDocument.filename}]\n${uploadedDocument.extractedText}`;
      }
      
      const response = await tokenManager.post('/api/query', {
        query: finalQuery,
        conversation_id: currentConversation?.id
      }, {
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
            comparison_table: data.comparison_table || null,
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
      setUploadedDocument(null); // Clear uploaded document after submission
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
    setUploadedDocument(null);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Don't set Content-Type header - let the browser set it automatically with boundary
      const response = await tokenManager.post('/api/upload-document', formData);

      if (!response.ok) {
        let errorMessage = 'Failed to upload document';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || JSON.stringify(errorData);
        } catch (e) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      
      if (data.success) {
        setUploadedDocument({
          filename: data.filename,
          extractedText: data.extracted_text
        });
        
        // Optionally, pre-fill the query with a message about the uploaded document
        setQuery(`[Document uploaded: ${data.filename}]\n\n`);
        
        if (textareaRef.current) {
          textareaRef.current.focus();
        }
      }
    } catch (error) {
      console.error('Error uploading document:', error);
      alert(`Error uploading document: ${error.message}`);
    } finally {
      setIsUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleRemoveDocument = () => {
    setUploadedDocument(null);
    setQuery('');
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
              <h1 className="landing-title">Recipes Search Agent</h1>
              <p className="landing-subtitle">Upload a customer brief or write your search description in the field bellow</p>
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
                    <div className="message-content assistant-message" dangerouslySetInnerHTML={{
                      __html: message.response
                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                        .replace(/\n/g, '<br/>')
                    }} />
                    
                    {/* Comparison Table */}
                    {message.comparison_table && message.comparison_table.has_data && (
                      <div className="comparison-table-container">
                        <h4>Recipe Comparison</h4>
                        <div className="table-wrapper">
                          <table className="comparison-table">
                            <thead>
                              <tr className="recipe-names">
                                {message.comparison_table.recipes.map((recipe, idx) => (
                                  <th key={idx} colSpan="2" className="recipe-header">
                                    {recipe.recipe_name || recipe.recipe_id}
                                  </th>
                                ))}
                              </tr>
                              <tr className="column-headers">
                                {message.comparison_table.recipes.map((recipe, idx) => (
                                  <React.Fragment key={idx}>
                                    <th className="characteristic-header">Characteristic</th>
                                    <th className="value-header">Value</th>
                                  </React.Fragment>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {message.comparison_table.recipes[0]?.characteristics.map((_, charIndex) => (
                                <tr key={charIndex}>
                                  {message.comparison_table.recipes.map((recipe, recipeIndex) => (
                                    <React.Fragment key={recipeIndex}>
                                      <td className="characteristic-cell">
                                        {recipe.characteristics[charIndex]?.charactDescr || ''}
                                      </td>
                                      <td className="value-cell">
                                        {recipe.characteristics[charIndex]?.valueCharLong || ''}
                                      </td>
                                    </React.Fragment>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                    
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
            {uploadedDocument && (
              <div className="uploaded-document-indicator">
                <span className="document-name">📎 {uploadedDocument.filename}</span>
                <button 
                  className="remove-document-btn"
                  onClick={handleRemoveDocument}
                  title="Remove document"
                >
                  ✕
                </button>
              </div>
            )}
            <div className="query-input">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".pdf,.doc,.docx,.ppt,.pptx,.jpg,.jpeg,.png,.gif,.bmp,.tiff,.webp,.svg,.html,.htm,.txt,.rtf,.odt"
                style={{ display: 'none' }}
              />
              <button 
                className="upload-button"
                onClick={handleUploadClick}
                disabled={isUploading || isProcessing}
                title="Document uploaded"
              >
                <img 
                  src="/upload.png" 
                  alt="Upload" 
                  className="upload-icon"
                />
              </button>
              <textarea
                ref={textareaRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question... (Shift+Enter to continue, Ctrl+V to paste)"
                disabled={isUploading}
              />
              <button 
                onClick={isProcessing ? handleStopRequest : handleSubmit}
                className={isProcessing ? 'stop-button' : ''}
                disabled={isUploading}
              >
                {isProcessing ? 'Stop' : isUploading ? 'Uploading...' : 'Send'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;