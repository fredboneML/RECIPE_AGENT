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
  const [uploadedDocuments, setUploadedDocuments] = useState([]); // Support up to 5 documents
  const [isUploading, setIsUploading] = useState(false);
  const [countries, setCountries] = useState([]);
  const [selectedCountries, setSelectedCountries] = useState([]);
  const [selectedVersion, setSelectedVersion] = useState('P');
  const [showCountryDropdown, setShowCountryDropdown] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  
  const navigate = useNavigate();
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const countryDropdownRef = useRef(null);
  const dropZoneRef = useRef(null);

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
        const conversationsResponse = await tokenManager.get('/conversations');
  
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

  // Load countries from JSON file
  useEffect(() => {
    const loadCountries = async () => {
      try {
        const response = await fetch('/countries.json');
        if (response.ok) {
          const data = await response.json();
          setCountries(data.countries || []);
        }
      } catch (error) {
        console.error('Error loading countries:', error);
      }
    };
    loadCountries();
  }, []);

  // Close country dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (countryDropdownRef.current && !countryDropdownRef.current.contains(event.target)) {
        setShowCountryDropdown(false);
      }
    };

    if (showCountryDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showCountryDropdown]);

  // Replace getHeaders with tokenManager logic
  const getHeaders = () => ({
    'Content-Type': 'application/json',
    'X-Tenant-Code': localStorage.getItem('tenantCode'),
    'Accept': 'application/json',
    'Authorization': `Bearer ${localStorage.getItem('token')}`,
  });

  const fetchConversations = async () => {
    try {
      const response = await tokenManager.get('/conversations');
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
      const response = await tokenManager.get(`/conversations/${conversationId}`);
      if (response.ok) {
        const data = await response.json();
        setMessages(data);
        setTimeout(scrollToBottom, 100);
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  
  const handleSubmit = async (overrideQuery) => {
    const submitQuery = (typeof overrideQuery === 'string') ? overrideQuery : query;
    if (!submitQuery.trim() || isProcessing) return;
    
    setIsProcessing(true);
    setQuery(submitQuery);
    
    abortControllerRef.current = new AbortController();
    
    try {
      // Combine query with all uploaded documents' text if available
      let finalQuery = submitQuery;
      if (uploadedDocuments.length > 0) {
        const documentsText = uploadedDocuments
          .map(doc => `[Extracted from document: ${doc.filename}]\n${doc.extractedText}`)
          .join('\n\n');
        finalQuery = `${submitQuery}\n\n${documentsText}`;
      }
      
      const response = await tokenManager.post('/query', {
        query: finalQuery,
        conversation_id: currentConversation?.id,
        country_filter: selectedCountries.length === 0 ? null : selectedCountries,
        version_filter: selectedVersion === 'All' ? null : selectedVersion
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
            query: submitQuery,
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
            query: submitQuery,
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
      setUploadedDocuments([]); // Clear uploaded documents after submission
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
    // Auto-submit for "top N recipes" follow-up questions (all languages)
    const topNPatterns = [
      "show me the next top",           // English
      "montrez-moi les",                // French
      "zeige mir die nächsten top",     // German
      "toon mij de volgende top",       // Dutch
      "mostrami le prossime",           // Italian
      "muéstrame las siguientes",       // Spanish
    ];
    const isTopNQuestion = topNPatterns.some(p => question.toLowerCase().startsWith(p));
    if (isTopNQuestion) {
      handleSubmit(question);
      return;
    }
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
    setUploadedDocuments([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const MAX_DOCUMENTS = 5;

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (!files.length) return;

    // Check if adding these files would exceed the limit
    const remainingSlots = MAX_DOCUMENTS - uploadedDocuments.length;
    if (remainingSlots <= 0) {
      alert(`Maximum ${MAX_DOCUMENTS} documents allowed. Please remove some documents first.`);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    // Only process up to the remaining slots
    const filesToProcess = files.slice(0, remainingSlots);
    if (files.length > remainingSlots) {
      alert(`Only ${remainingSlots} more document(s) can be added. Processing first ${remainingSlots} file(s).`);
    }

    setIsUploading(true);
    
    try {
      const newDocuments = [];
      
      for (const file of filesToProcess) {
      const formData = new FormData();
      formData.append('file', file);

      // Don't set Content-Type header - let the browser set it automatically with boundary
      const response = await tokenManager.post('/upload-document', formData);

      if (!response.ok) {
        let errorMessage = 'Failed to upload document';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || JSON.stringify(errorData);
        } catch (e) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
          throw new Error(`${file.name}: ${errorMessage}`);
      }

      const data = await response.json();
      
      if (data.success) {
          newDocuments.push({
          filename: data.filename,
          extractedText: data.extracted_text
        });
        }
      }
      
      // Add new documents to existing ones
      setUploadedDocuments(prev => [...prev, ...newDocuments]);
        
      // Update query with all document names
      const allDocNames = [...uploadedDocuments, ...newDocuments].map(d => d.filename);
      setQuery(`[Documents uploaded: ${allDocNames.join(', ')}]\n\n`);
        
        if (textareaRef.current) {
          textareaRef.current.focus();
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

  const handleRemoveDocument = (indexToRemove) => {
    setUploadedDocuments(prev => {
      const updated = prev.filter((_, index) => index !== indexToRemove);
      // Update query with remaining document names
      if (updated.length > 0) {
        const docNames = updated.map(d => d.filename);
        setQuery(`[Documents uploaded: ${docNames.join(', ')}]\n\n`);
      } else {
        setQuery('');
      }
      return updated;
    });
  };

  // Drag and drop handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    // Only set dragging to false if we're leaving the drop zone itself
    if (!dropZoneRef.current?.contains(e.relatedTarget)) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (!files.length) return;

    // Check if adding these files would exceed the limit
    const remainingSlots = MAX_DOCUMENTS - uploadedDocuments.length;
    if (remainingSlots <= 0) {
      alert(`Maximum ${MAX_DOCUMENTS} documents allowed. Please remove some documents first.`);
      return;
    }

    // Only process up to the remaining slots
    const filesToProcess = files.slice(0, remainingSlots);
    if (files.length > remainingSlots) {
      alert(`Only ${remainingSlots} more document(s) can be added. Processing first ${remainingSlots} file(s).`);
    }

    // Process the dropped files using the same logic as file upload
    setIsUploading(true);
    
    try {
      const newDocuments = [];
      
      for (const file of filesToProcess) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await tokenManager.post('/upload-document', formData);

        if (!response.ok) {
          let errorMessage = 'Failed to upload document';
          try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorData.message || JSON.stringify(errorData);
          } catch (e) {
            errorMessage = `HTTP ${response.status}: ${response.statusText}`;
          }
          throw new Error(`${file.name}: ${errorMessage}`);
        }

        const data = await response.json();
        
        if (data.success) {
          newDocuments.push({
            filename: data.filename,
            extractedText: data.extracted_text
          });
        }
      }
      
      // Add new documents to existing ones
      setUploadedDocuments(prev => [...prev, ...newDocuments]);
        
      // Update query with all document names
      const allDocNames = [...uploadedDocuments, ...newDocuments].map(d => d.filename);
      setQuery(`[Documents uploaded: ${allDocNames.join(', ')}]\n\n`);
        
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    } catch (error) {
      console.error('Error uploading dropped files:', error);
      alert(`Error uploading document: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleCountryToggle = (countryName) => {
    setSelectedCountries(prev => {
      // If "All" is selected, clear all other selections (means search all countries)
      if (countryName === 'All') {
        return prev.includes('All') ? [] : ['All'];
      }
      // If selecting a specific country, remove "All" from selection
      const withoutAll = prev.filter(c => c !== 'All');
      if (prev.includes(countryName)) {
        return withoutAll.filter(c => c !== countryName);
      } else {
        return [...withoutAll, countryName];
      }
    });
  };

  const handleSelectAllCountries = () => {
    if (selectedCountries.length === countries.length) {
      setSelectedCountries([]);
    } else {
      setSelectedCountries(countries.map(c => c.name));
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
          <div className="country-filter">
            <label htmlFor="country-select">Country:</label>
            <div className="country-multiselect-wrapper" ref={countryDropdownRef}>
              <button
                type="button"
                className={`country-multiselect-button ${showCountryDropdown ? 'open' : ''}`}
                onClick={() => setShowCountryDropdown(!showCountryDropdown)}
              >
                {selectedCountries.length === 0 
                  ? 'All Countries' 
                  : selectedCountries.length === 1
                  ? selectedCountries[0]
                  : `${selectedCountries.length} selected`}
                <span className="country-dropdown-arrow">▼</span>
              </button>
              {showCountryDropdown && (
                <div className="country-multiselect-dropdown">
                  <div className="country-multiselect-header">
                    <button
                      type="button"
                      className="country-select-all-btn"
                      onClick={handleSelectAllCountries}
                    >
                      {selectedCountries.length === countries.length ? 'Deselect All' : 'Select All'}
                    </button>
                    {selectedCountries.length > 0 && (
                      <button
                        type="button"
                        className="country-clear-btn"
                        onClick={() => setSelectedCountries([])}
                      >
                        Clear
                      </button>
                    )}
                  </div>
                  <div className="country-multiselect-list">
                    {countries.map((country) => (
                      <label
                        key={country.code}
                        className="country-multiselect-option"
                      >
                        <input
                          type="checkbox"
                          checked={selectedCountries.includes(country.name)}
                          onChange={() => handleCountryToggle(country.name)}
                        />
                        <span>{country.name}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
          <div className="version-filter">
            <label htmlFor="version-select">Version:</label>
            <select
              id="version-select"
              value={selectedVersion}
              onChange={(e) => setSelectedVersion(e.target.value)}
              className="version-dropdown"
            >
              <option value="All">All</option>
              <option value="P">P</option>
              <option value="L">L</option>
              <option value="Missing">Missing</option>
            </select>
          </div>
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
                    {/* Check if this is a recipe list (starts with numbered recipes) */}
                    {message.response.match(/^\d+\.\s+[A-Z0-9_]+.*Score:\s*\d/) ? (
                      <div className="recipe-list-container">
                        <div className="recipe-list-header">
                          <h4>Top Recipes</h4>
                        </div>
                        <div className="recipe-list-content">
                          {message.response.split('\n').map((line, idx) => {
                            if (!line.trim()) return null;
                            return (
                              <div key={idx} className="recipe-list-item">
                                {line}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ) : (
                      <div className="message-content assistant-message" dangerouslySetInnerHTML={{
                        __html: message.response
                          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                          .replace(/\n/g, '<br/>')
                      }} />
                    )}
                    
                    {/* Comparison Table - 60 Specified Fields Format */}
                    {message.comparison_table && message.comparison_table.has_data && (
                      <div className="comparison-table-container">
                        <h4>Recipe Comparison - 60 Specified Fields</h4>
                        <div className="table-wrapper">
                          {message.comparison_table.field_definitions && message.comparison_table.field_definitions.length > 0 ? (
                            /* New format: Code | Field Name | Recipe1 | Recipe2 | Recipe3 */
                            <table className="comparison-table structured">
                              <thead>
                                <tr>
                                  <th className="field-code-header sticky-col">Code</th>
                                  <th className="field-name-header sticky-col-2">Field Name</th>
                                  {message.comparison_table.recipes.map((recipe, idx) => (
                                    <th key={idx} className="recipe-value-header">
                                      {idx + 1}. {recipe.recipe_name || recipe.recipe_id}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {message.comparison_table.field_definitions.map((field, fieldIdx) => {
                                  const hasAnyValue = message.comparison_table.recipes.some(r => 
                                    r.values && r.values[fieldIdx] && String(r.values[fieldIdx]).trim() !== ''
                                  );
                                  return (
                                    <tr key={fieldIdx} className={hasAnyValue ? 'has-value' : 'no-value'}>
                                      <td className="field-code sticky-col">{field.code}</td>
                                      <td className="field-name sticky-col-2">{field.display_name}</td>
                                      {message.comparison_table.recipes.map((recipe, recipeIdx) => (
                                        <td key={recipeIdx} className="recipe-value">
                                          {recipe.values && recipe.values[fieldIdx] 
                                            ? String(recipe.values[fieldIdx])
                                            : '-'}
                                        </td>
                                      ))}
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          ) : (
                            /* Legacy fallback */
                            <table className="comparison-table legacy">
                              <thead>
                                <tr className="recipe-names">
                                  {message.comparison_table.recipes.map((recipe, idx) => (
                                    <th key={idx} colSpan="2" className="recipe-header">
                                      {idx + 1}. {recipe.recipe_name || recipe.recipe_id}
                                    </th>
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
                          )}
                        </div>
                        <div className="table-legend">
                          <span className="legend-has-value">● Fields with values</span>
                          <span className="legend-no-value">○ Fields without values</span>
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

          <div 
            className={`input-container ${isDragging ? 'drag-over' : ''}`}
            ref={dropZoneRef}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {uploadedDocuments.length > 0 && (
              <div className="uploaded-documents-container">
                {uploadedDocuments.map((doc, index) => (
                  <div key={index} className="uploaded-document-indicator">
                    <span className="document-name">📎 {doc.filename}</span>
                <button 
                  className="remove-document-btn"
                      onClick={() => handleRemoveDocument(index)}
                  title="Remove document"
                >
                  ✕
                </button>
                  </div>
                ))}
                {uploadedDocuments.length < MAX_DOCUMENTS && (
                  <div className="documents-count">
                    {uploadedDocuments.length}/{MAX_DOCUMENTS} documents
                  </div>
                )}
              </div>
            )}
            {isDragging && (
              <div className="drag-overlay">
                <div className="drag-overlay-content">
                  <div className="drag-icon">📄</div>
                  <div className="drag-text">Drop files here to upload</div>
                </div>
              </div>
            )}
            <div className="query-input">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".pdf,.doc,.docx,.ppt,.pptx,.jpg,.jpeg,.png,.gif,.bmp,.tiff,.webp,.svg,.html,.htm,.txt,.rtf,.odt"
                multiple
                style={{ display: 'none' }}
              />
              <button 
                className="upload-button"
                onClick={handleUploadClick}
                disabled={isUploading || isProcessing || uploadedDocuments.length >= MAX_DOCUMENTS}
                title={uploadedDocuments.length >= MAX_DOCUMENTS ? `Maximum ${MAX_DOCUMENTS} documents reached` : "Upload documents (up to 5)"}
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