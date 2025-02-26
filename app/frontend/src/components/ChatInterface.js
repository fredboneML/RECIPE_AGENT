const handleFollowupClick = (question) => {
  // Check if this is a "start new conversation" suggestion when context is exceeded
  if (contextExceeded && 
      (question.toLowerCase().includes("start a new conversation") || 
       question.toLowerCase().includes("start een nieuw gesprek"))) {
    // This will have the same effect as clicking the New Conversation button
    handleNewConversation();
    return;
  }
  
  // Normal follow-up question handling
  setUserInput(question);
  handleSendMessage(question);
};

{followupQuestions && followupQuestions.length > 0 && (
  <div className="followup-questions">
    <h4>{contextExceeded ? 
      (isDutch ? "Aanbevolen acties:" : "Recommended actions:") : 
      (isDutch ? "Suggesties voor vervolgvragen:" : "Suggested follow-up questions:")}
    </h4>
    <ul>
      {followupQuestions.map((question, index) => (
        <li 
          key={index} 
          onClick={() => handleFollowupClick(question)}
          className={contextExceeded && 
            (question.toLowerCase().includes("start a new conversation") || 
             question.toLowerCase().includes("start een nieuw gesprek")) 
            ? "new-conversation-action" : ""}
        >
          {question}
        </li>
      ))}
    </ul>
  </div>
)} 